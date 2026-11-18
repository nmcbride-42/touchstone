import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs network + LLM credentials "
                            "(set TOUCHSTONE_RUN_NETWORK_TESTS=1 to run)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TOUCHSTONE_RUN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set TOUCHSTONE_RUN_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
