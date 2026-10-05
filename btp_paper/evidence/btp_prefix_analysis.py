#!/usr/bin/env python3
"""Ledger §12.6: autoregressive prefix control (BtP plan B2).

    .venv/bin/python docs/btp_paper/evidence/btp_prefix_analysis.py [--precision bf16]

Input: outputs/btp/a100/prefix_<precision>_a100.shard*.csv (scripts/run_gpu.py prefix). Each scene x
wording x image transform has three rows (a = "left", b = "right", n = term-free), each holding dy read
after every greedy dx token (ct1_from_a/b/n), after its own (ct1_own), and as mixtures over dx
(cm1_npi: weights p(dx | neutral); cm1_own: the prompt's own p(dx)).

Decomposition of the oriented left-right dy difference (grid steps; + = "left" more image-left):
  total        dy(left | dx_left)  - dy(right | dx_right)            what greedy decoding executes
  CDE(t)       dy(left | t)        - dy(right | t), t in {dx_left, dx_right, dx_neutral}
  direct       mean of CDE(dx_left) and CDE(dx_right); interaction = their difference
  indirect     total - direct (through the dx token)
  marg_direct  cm1_npi(left) - cm1_npi(right)                       dx drawn from the neutral prompt
  samp_total   cm1_own(left) - cm1_own(right)                       total effect under sampling
Means with base-frame bootstrap CIs (4,000 draws); split direction with dx fixed at dx_neutral.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import btp_paper.evidence.btp_evidence as E  # noqa: E402

R = E.R
REPO = E.REPO
STEP = float(E.STEP_255[1])
WORDINGS = ("baseline", "prenominal", "object", "table_side")
SEED, N_BOOT = 20261003, 4000


def wide(run: pd.DataFrame, wording: str, transform: str) -> pd.DataFrame:
    sub = run[(run["condition"] == f"prefix_{wording}") & (run["image_transform"] == transform)]
    cols = ["ct1_own", "ct1_from_a", "ct1_from_b", "ct1_from_n", "cm1_npi", "cm1_own", "dx_token"]
    w = sub.pivot_table(index="scene_id", columns="role", values=cols, aggfunc="first")
    w.columns = [f"{c}_{r}" for c, r in w.columns]
    return w


def effects(w: pd.DataFrame) -> pd.DataFrame:
    # Positive when "left" (role a) moves further image-left (+dy) than "right" (role b).
    out = pd.DataFrame(index=w.index)
    out["total"] = (w["ct1_own_a"] - w["ct1_own_b"]) / STEP
    for src in ("a", "b", "n"):
        out[f"cde_{src}"] = (w[f"ct1_from_{src}_a"] - w[f"ct1_from_{src}_b"]) / STEP
    out["direct"] = (out["cde_a"] + out["cde_b"]) / 2
    out["interaction"] = out["cde_a"] - out["cde_b"]
    out["indirect"] = out["total"] - out["direct"]
    out["marg_direct"] = (w["cm1_npi_a"] - w["cm1_npi_b"]) / STEP
    out["samp_total"] = (w["cm1_own_a"] - w["cm1_own_b"]) / STEP
    out["dx_differs"] = (w["dx_token_a"] != w["dx_token_b"]).astype(float)
    return out


def boot_mean(df: pd.DataFrame, col: str, frames: pd.Series, rng) -> list:
    codes, uniq = pd.factorize(frames.loc[df.index])
    sums = np.bincount(codes, weights=df[col].to_numpy(), minlength=len(uniq))
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT).astype(float)
    means = np.einsum("nf,f->n", w, sums) / np.einsum("nf,f->n", w, counts)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def split_direction(run, wording, transform, geo, col):
    rows = run[(run["condition"] == f"prefix_{wording}") & (run["image_transform"] == transform) &
               (run["role"].isin(["a", "b"]))].copy()
    rows["condition"] = R.BASELINE
    rows["cf1"] = rows[col].astype(float)
    for i in range(7):            # the prefix log holds dy only; evaluate() reads c1 from `value`
        rows[f"c{i}"] = rows["cf1"] if i == 1 else 0.0
    e = E.evaluate(rows, geo, value="cf1", floor=STEP, unit=STEP, n_boot=0, images=transform)
    return {"splits": e["splits"], "right_way": e["splits_right_way"], "p": e["splits_p"],
            "opp_both_correct": e["opp_both_correct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="bf16")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data = E.load_all(os.path.join(REPO, "google_drive", "v2"), os.path.join(REPO, "outputs", "runs"))
    geo = data["geo"]
    frames = geo["base_scene_id"]
    run = R.load_run(os.path.join(REPO, "outputs", "btp", "a100", f"prefix_{args.precision}_a100.csv"))
    res = {"precision": args.precision, "unit": f"grid steps of {STEP:.7f}", "n_rows": int(len(run))}

    # Cross-check: the own-prefix rows reproduce the A100 ladder run of the same prompts (bf16 only).
    if args.precision == "bf16":
        lad = R.load_run(os.path.join(REPO, "outputs", "btp", "a100", "ladder_bf16_a100.csv"))
        rep = R.load_run(os.path.join(REPO, "outputs", "btp", "a100", "replay_bf16_a100.csv"))
        pre = run[run["role"].isin(["a", "b"])]
        checks = []
        for w in WORDINGS:
            ref = (rep[rep["condition"] == R.BASELINE] if w == "baseline" else lad[lad["condition"] == f"ladder_{w}"])
            ref = ref[ref["image_transform"] == "original"][["scene_id", "role", "cf1"]]
            m = pre[(pre["condition"] == f"prefix_{w}") & (pre["image_transform"] == "original")][
                ["scene_id", "role", "ct1_own"]].merge(ref, on=["scene_id", "role"])
            checks.append(float(((m["ct1_own"] - m["cf1"]).abs() / STEP).max()))
        res["max_abs_dev_vs_generate_runs_steps"] = max(checks)

    for transform in ("original", "mirror"):
        for w in WORDINGS:
            eff = effects(wide(run, w, transform))
            if eff.empty:
                continue
            cell = {"n_scenes": int(len(eff)), "dx_differs": float(eff["dx_differs"].mean())}
            for col in ("total", "direct", "indirect", "interaction", "cde_n", "marg_direct", "samp_total"):
                cell[col] = {"mean": float(eff[col].mean()), "median": float(eff[col].median()),
                             "ci_mean": boot_mean(eff, col, frames, rng)}
            same_sign = np.sign(eff["direct"]) == np.sign(eff["total"])
            cell["direct_keeps_total_sign"] = float(same_sign[eff["total"] != 0].mean())
            cell["share_of_total_mean_direct"] = (float(eff["direct"].mean() / eff["total"].mean())
                                                  if eff["total"].mean() != 0 else float("nan"))
            cell["o2_own"] = split_direction(run, w, transform, geo, "ct1_own")
            cell["o2_fixed_dx_neutral"] = split_direction(run, w, transform, geo, "ct1_from_n")
            res[f"{transform}|{w}"] = cell
    out = os.path.join(HERE, f"prefix_{args.precision}.json")
    with open(out, "w") as f:
        json.dump(E.jclean(res), f, indent=2)
    for k, v in res.items():
        if isinstance(v, dict) and "total" in v:
            print(f"{k:22s} total {v['total']['mean']:+.2f} [{v['total']['ci_mean'][0]:+.2f},{v['total']['ci_mean'][1]:+.2f}]  "
                  f"direct {v['direct']['mean']:+.2f} [{v['direct']['ci_mean'][0]:+.2f},{v['direct']['ci_mean'][1]:+.2f}]  "
                  f"indirect {v['indirect']['mean']:+.2f}  interaction {v['interaction']['mean']:+.2f}  "
                  f"marg {v['marg_direct']['mean']:+.2f}  dx differs {v['dx_differs']:.0%}  "
                  f"O2 own {v['o2_own']['right_way']}/{v['o2_own']['splits']}  fixed {v['o2_fixed_dx_neutral']['right_way']}/{v['o2_fixed_dx_neutral']['splits']}")
    print("cross-check (own prefix vs generate runs), max |dev| steps:", res.get("max_abs_dev_vs_generate_runs_steps"))


if __name__ == "__main__":
    main()
