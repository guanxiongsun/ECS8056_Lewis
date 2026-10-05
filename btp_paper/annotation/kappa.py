#!/usr/bin/env python3
"""Agreement for the twin-scene re-annotation (BtP plan B6 step 4).

    .venv/bin/python docs/btp_paper/annotation/kappa.py reannotation_L.csv reannotation_S.csv

Reports, per question, raw agreement, Cohen's kappa and Gwet's AC1 (stable when one category
dominates, as plausibility does) between the two annotators, each with a 95% bootstrap CI over
scenes, plus each annotator's agreement with the original hand labels for arrangement and
plausibility.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORIG = HERE.parents[2] / "google_drive" / "v2" / "constructed" / "validation_sample.csv"
QS = ["two_instances", "gripper_ok", "arrangement", "plausible"]
TO_NEW = {"same_side_left": "both left", "opposite": "one on each side", "same_side_right": "both right",
          "unclear": "unclear"}


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def ac1(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pi = [(np.mean(a == c) + np.mean(b == c)) / 2 for c in cats]
    pe = sum(p * (1 - p) for p in pi) / max(1, len(cats) - 1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(a, b, f, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    vals = np.array([f(a[i], b[i]) for i in idx])
    return np.nanpercentile(vals, [2.5, 97.5])


def summary(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    k_lo, k_hi = boot(a, b, kappa)
    g_lo, g_hi = boot(a, b, ac1)
    return (f"n={len(a)}; agreement {np.mean(a == b):.1%}; kappa {kappa(a, b):.2f} [{k_lo:.2f}, {k_hi:.2f}]; "
            f"AC1 {ac1(a, b):.2f} [{g_lo:.2f}, {g_hi:.2f}]")


def main(paths):
    sample = pd.read_csv(HERE / "sample.csv")
    runs = []
    for p in paths:
        d = pd.read_csv(p, dtype=str).drop(columns=["position"], errors="ignore").merge(sample, on="scene_key")
        missing = d[QS].isna().any(axis=1).sum()
        print(f"{p}: annotator {d['annotator'].iloc[0]}, {len(d)} scenes, {missing} incomplete")
        runs.append(d.dropna(subset=QS).set_index("construct_id"))
    orig = pd.read_csv(ORIG).set_index("construct_id")
    if len(runs) == 2:
        a, b = runs
        common = a.index.intersection(b.index)
        print(f"\nBetween annotators ({a['annotator'].iloc[0]} vs {b['annotator'].iloc[0]}):")
        for q in QS:
            print(f"  {q:14s} {summary(a.loc[common, q], b.loc[common, q])}")
    for r in runs:
        who = r["annotator"].iloc[0]
        o = orig.loc[r.index]
        print(f"\n{who} vs the original hand labels:")
        print(f"  arrangement    {summary(r['arrangement'], o['human_configuration'].map(TO_NEW))}")
        print(f"  plausible      {summary(r['plausible'], o['human_paste_plausible'])}")
        print(f"  two_instances  share 'yes' {np.mean(r['two_instances'] == 'yes'):.1%} (original: 100% yes)")
        print(f"  gripper_ok     share 'yes' {np.mean(r['gripper_ok'] == 'yes'):.1%} (original: 100% yes)")


if __name__ == "__main__":
    sys.argv[1:] or sys.exit(__doc__)
    main(sys.argv[1:])
