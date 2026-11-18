"""Code-writing proposers: an LLM writes the candidate as Python source.

The proposer never has to be right. Every program it writes is scored by the
disposer (a VerificationHandle) inside the sandbox; we only keep what scores.

Vanilla FunSearch issues thousands of one-at-a-time calls. Touchstone batches
instead: each call shows the current best programs with their scores and asks
for several diverse new ones, so a handful of calls is enough for a
demonstration.

Three interchangeable backends, all configured by environment variables
(see README.md / .env.example):

  anthropic  the official `anthropic` SDK (Messages API)
  openai     the official `openai` SDK; also any OpenAI-compatible server
             (vLLM, llama.cpp, Ollama, OpenRouter, ...) via a base URL
  command    any CLI that reads a prompt on stdin and prints a reply on stdout
             (e.g. a local agent CLI). Useful for mixing model families.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import time
from typing import Protocol

from ..config import env

_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """All fenced code blocks, unfiltered."""
    return [m.group(1).strip() for m in _BLOCK.finditer(text or "")]


def extract_functions(text: str, fn_name: str) -> list[str]:
    """Fenced blocks that define `def {fn_name}`; falls back to the raw reply
    when a model answers with bare code and no fence."""
    blocks = [c for c in parse_code_blocks(text) if f"def {fn_name}" in c]
    if not blocks and f"def {fn_name}" in (text or "") and "```" not in (text or ""):
        blocks = [text.strip()]
    return blocks


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, system: str) -> str: ...


class AnthropicClient:
    """Claude via the official SDK. Credentials resolve the SDK's usual way
    (ANTHROPIC_API_KEY, ...). Current Claude models reject `temperature`, so
    island diversity comes from per-island strategy hints, not sampling."""

    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None,
                 timeout: float | None = None, fallbacks: str | None = None):
        import anthropic  # optional dependency: pip install -e ".[anthropic]"

        self.model = model or env("TOUCHSTONE_ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = effort or env("TOUCHSTONE_ANTHROPIC_EFFORT", "medium")
        self.fallbacks = fallbacks or env("TOUCHSTONE_ANTHROPIC_FALLBACKS", "default")
        self._client = anthropic.Anthropic(
            timeout=float(timeout or env("TOUCHSTONE_LLM_TIMEOUT", "300")))

    def complete(self, prompt: str, system: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
        )
        if self.fallbacks and self.fallbacks != "off":
            # Server-side refusal fallback (beta): if the model declines, the API
            # re-runs the request on a fallback model inside the same call.
            resp = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": self.fallbacks}, **kwargs)
        else:
            resp = self._client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIClient:
    """OpenAI or any OpenAI-compatible endpoint via the official `openai` SDK."""

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 temperature: float | None = None, timeout: float | None = None):
        import openai  # optional dependency: pip install -e ".[openai]"

        self.model = model or env("TOUCHSTONE_OPENAI_MODEL")
        if not self.model:
            raise ValueError("set TOUCHSTONE_OPENAI_MODEL (e.g. a model id your endpoint serves)")
        t = temperature if temperature is not None else env("TOUCHSTONE_OPENAI_TEMPERATURE")
        self.temperature = float(t) if t is not None else None
        self._client = openai.OpenAI(
            base_url=base_url or env("TOUCHSTONE_OPENAI_BASE_URL"),
            timeout=float(timeout or env("TOUCHSTONE_LLM_TIMEOUT", "300")))

    def complete(self, prompt: str, system: str) -> str:
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            **kwargs)
        return resp.choices[0].message.content or ""


class CommandClient:
    """Pipe the prompt to any command on stdin; read the reply from stdout.
    The system prompt is prepended to the user prompt."""

    name = "command"
    _PREAMBLE = ("Reply in a single turn containing ONLY the requested ```python``` "
                 "code blocks. Do not run tools or commands.\n\n")

    def __init__(self, command: str | None = None, timeout: float | None = None):
        cmd = command or env("TOUCHSTONE_LLM_COMMAND")
        if not cmd:
            raise ValueError("set TOUCHSTONE_LLM_COMMAND (a CLI that reads stdin)")
        self.argv = shlex.split(cmd)
        self.timeout = float(timeout or env("TOUCHSTONE_LLM_TIMEOUT", "300"))

    def complete(self, prompt: str, system: str) -> str:
        proc = subprocess.run(self.argv, input=f"{self._PREAMBLE}{system}\n\n{prompt}",
                              capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or f"rc={proc.returncode}")[:200])
        return proc.stdout


def client_from_env(provider: str | None = None) -> LLMClient:
    provider = (provider or env("TOUCHSTONE_LLM_PROVIDER", "anthropic")).lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "command":
        return CommandClient()
    raise ValueError(f"unknown TOUCHSTONE_LLM_PROVIDER {provider!r} "
                     "(expected anthropic | openai | command)")


class CodeProposer:
    """Wraps an LLMClient into the proposer interface used by the loops:
    `propose_raw(prompt, system, fn_name) -> list[source]`. Never raises; on
    failure it returns [] and records why in `last_meta`."""

    def __init__(self, client: LLMClient, role: str = "", role_hint: str = ""):
        self.client = client
        self.role = role or client.name
        self.role_hint = role_hint
        self.last_meta: dict = {}

    def propose_raw(self, prompt: str, system: str, fn_name: str = "priority") -> list[str]:
        if self.role_hint:
            system = f"{system}\n\nISLAND STRATEGY ({self.role}): {self.role_hint}"
        t0 = time.time()
        try:
            text = self.client.complete(prompt, system)
            codes = extract_functions(text, fn_name)
            self.last_meta = {"driver": self.client.name, "role": self.role, "ok": True,
                              "elapsed_s": round(time.time() - t0, 2),
                              "n_extracted": len(codes)}
            return codes
        except Exception as e:  # a failed call costs a generation, never the run
            self.last_meta = {"driver": self.client.name, "role": self.role, "ok": False,
                              "elapsed_s": round(time.time() - t0, 2),
                              "error": repr(e)[:200]}
            return []
