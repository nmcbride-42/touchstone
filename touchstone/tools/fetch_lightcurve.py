"""Fetch + phase-fold a TESS light curve, cache (phase, flux) for Touchstone.

Needs `lightkurve` (optional extra: `pip install -e ".[fetch]"`) and network
access to MAST. Touchstone itself only reads the cached .npz with numpy, which
keeps the core dependency-light. The repo already ships the one curve the
examples use (data/lc_329089161.npz), so this tool is only needed to add more.

    python -m touchstone.tools.fetch_lightcurve \
        --tic 329089161 --period <days> --t0 <BTJD> --bins 140

Output: $TOUCHSTONE_DATA_DIR/lc_<tic>.npz (default ./data) with arrays
phase[-0.5,0.5], flux (median-normalized, binned), plus scalars
tic/period/t0/n_raw. The raw FITS downloads go to <data dir>/cache/.
"""
import argparse

import numpy as np

from ..config import data_dir

DATA = data_dir()


def fetch_all(tic):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
    ts, fs = [], []
    for i in range(len(sr)):
        try:
            lc = sr[i].download(download_dir=str(DATA / "cache")).remove_nans()
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            fs.append(f / np.median(f))            # per-sector median normalize
            ts.append(t)
        except Exception as e:
            print(f"  sector {i} dl failed: {type(e).__name__}")
    if not ts:
        raise RuntimeError("no sectors downloaded")
    return np.concatenate(ts), np.concatenate(fs)


def fold(t, f, period, t0, bins):
    phase = (((t - t0) / period + 0.5) % 1.0) - 0.5     # [-0.5, 0.5], eclipse ~0
    order = np.argsort(phase)
    phase, f = phase[order], f[order]
    # normalize by the OUT-OF-ECLIPSE baseline so baseline -> 1.0 (robust to
    # per-sector offsets and to the global median being skewed)
    base = np.median(f[np.abs(phase) > 0.15])
    if base <= 0:
        base = np.median(f)
    f = f / base
    # drop physically-impossible points (faint-star negative-flux artifacts);
    # keep eclipses as deep as 70% — generous, removes only garbage
    keep = (f > 0.30) & (f < 1.15)
    phase, f = phase[keep], f[keep]
    # bin: median flux per phase bin -> clean curve, noise down
    edges = np.linspace(-0.5, 0.5, bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)
    bp, bf = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 3:
            bp.append(0.5 * (edges[b] + edges[b + 1]))
            bf.append(float(np.median(f[m])))
    return np.array(bp), np.array(bf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--bins", type=int, default=140)
    a = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    print(f"fetching TIC {a.tic} ...")
    t, f = fetch_all(a.tic)
    print(f"  {len(t)} raw cadences; folding at P={a.period}d t0={a.t0}")
    phase, flux = fold(t, f, a.period, a.t0, a.bins)
    out = DATA / f"lc_{a.tic}.npz"
    np.savez(out, phase=phase, flux=flux, tic=a.tic, period=a.period,
             t0=a.t0, n_raw=len(t))
    print(f"  folded -> {len(phase)} phase bins; "
          f"flux range [{flux.min():.4f}, {flux.max():.4f}]")
    print(f"  min flux at phase {phase[int(np.argmin(flux))]:+.3f} (eclipse)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
