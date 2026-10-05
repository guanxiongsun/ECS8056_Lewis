#!/usr/bin/env python3
"""
btp_evidence_addenda.py: §11 addenda to the BtP claims ledger (revision round, 2 Oct 2026).

    .venv/bin/python docs/btp_paper/evidence/btp_evidence.py          # Table 1 (incl. the gpu row) first
    .venv/bin/python docs/btp_paper/evidence/btp_evidence_addenda.py

Recomputes from existing outputs only (no GPU), reusing `btp_evidence.py` (same folder):

  A1  GPU-only comparison: NF4 + 255-token readout (c1), Colab A100 log vs GH200 replay
  A2  physical units: bridge_orig q01/q99 of dy, one bin in action units, control rate, unit evidence
  A3  typical first-action magnitude in the reference run; word effects and the positive control as fractions
  A4  Holm across Table 1: 16 rows x 3 p-values; without tok/w255; with gpu; distinct analyses only
  A5  opposite both-correct = split share x right-way share, Wilson CIs; scenes per base frame
  A6  label-swap invariance of the sign-agreement contrast (numeric check of the formal statement)
  A7  wording x run grid: splits k/n, opposite both-correct k/n, sign-agreement contrast, signed effect
  A8  base-frame structure of the 340 scenes
  A9  frame-level inference for the signed effect (sign-flip by base frame; frame-bootstrap CI; Holm)
  A10 placebo for the direction-blind contrast: the grab/take paraphrase pair
  A11 Table 1 with one bin = /255 grid step vs /254 width: every cell compared
  A12 argmax decisions counted in grid steps from the zero-motion bin 128
  A13 multiplicity summary for the paper's highlighted claims

Writes `addenda.json` next to this file and prints the markdown tables used in claims_ledger.md §11.
Nothing in the repository is modified.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import btp_paper.evidence.btp_evidence as E  # noqa: E402  (also puts the repo on sys.path)
import reanalysis as R  # noqa: E402

TABLE1_16 = ["ref", "dx", "sign", "zero", "tok", "w254", "mir_naive", "argmax", "thr0", "thr2", "nf4", "mir",
             "w_prenominal", "w_object", "w_table_side", "sub_det"]
STATS3 = (("contrast", "contrast_p"), ("splits", "splits_p"), ("signed", "signed_p"))


def wilson(k, n):
    lo, hi = R.wilson_interval(int(k), int(n))
    return [lo, hi]


# --------------------------------------------------------------------------- #
# A1. GPU at fixed precision and readout
# --------------------------------------------------------------------------- #
def a1_gpu(data) -> dict:
    log, geo, runs = data["log"], data["geo"], data["runs"]
    a100 = E.logged_rows(log)
    gh = E.wording_rows(runs["replay_nf4"], "baseline")
    out = {}
    for label, rows in (("A100_nf4_c1", a100), ("GH200_nf4_c1", gh)):
        e = E.evaluate(rows, geo, value="c1")
        out[label] = {"by_layout": {k: {m: v[m] for m in ("n", "same_sign", "both_correct", "right_share")}
                                    | {f"{m}_ci": v.get(f"{m}_ci") for m in ("same_sign", "both_correct")}
                                    for k, v in e["by_layout"].items()},
                      **{k: e[k] for k in ("opp_n", "opp_both_correct", "opp_same_sign", "contrast_pts", "contrast_p",
                                           "contrast_pairs", "contrast_discordant", "splits", "splits_right_way",
                                           "splits_p", "signed_median_bins", "signed_p")}}
    a, g = out["A100_nf4_c1"], out["GH200_nf4_c1"]
    out["verdict_changes"] = {
        name: {"A100_p": a[pk], "GH200_p": g[pk], "same_side_of_.05": bool((a[pk] < .05) == (g[pk] < .05))}
        for name, pk in (("contrast", "contrast_p"), ("splits", "splits_p"), ("signed", "signed_p"))}
    out["verdict_changes"]["any_change"] = not all(v["same_side_of_.05"] for v in out["verdict_changes"].values())
    key = R.REPLAY_KEY
    rp = runs["replay_nf4"]
    out["agreement_all_2720"] = E._compare_tokens(log, rp, key)
    out["agreement_baseline_680"] = E._compare_tokens(log[log["condition"] == R.BASELINE],
                                                      rp[rp["condition"] == R.BASELINE], key)
    out["environment"] = {
        "logged_in_csv": {"A100": E._environment(log), "GH200": E._environment(rp)},
        "identical": ["torch 2.11.0+cu128", "transformers 4.40.1", "tokenizers 0.19.1", "timm 0.9.10",
                      "huggingface_hub 0.23.4", "accelerate 0.30.1", "bitsandbytes 0.50.2",
                      "NF4 double quantisation, bf16 compute dtype", "eager attention", "seed 42",
                      "greedy decoding, batch size 1, attention mask dropped, empty token appended",
                      "bridge_orig q01/q99 (to 5 d.p.) and action-space constants"],
        "different": {"GPU": "A100-SXM4-40GB (sm_80) vs GH200 120GB (sm_90)",
                      "CPU architecture": "x86-64 (Colab) vs aarch64 (Grace)",
                      "Python": "3.12.13 (Colab, NB01/03/05 output) vs 3.12.14 (Isambard, setup_env.log)",
                      "NVIDIA driver": "not logged on Colab; 565.57.01 on GH200 (smoke log)",
                      "date / weights download": "30 Aug (Colab) vs 1 Oct (Isambard); no model revision hash logged"},
        "unknown_on_colab": {"numpy": "2.5.2 on Isambard", "Pillow": "12.3.0 on Isambard",
                             "torchvision": "0.26.0+cu128 on Isambard", "cuBLAS": "12.8.4.1 on Isambard",
                             "cuDNN": "9.19.0.56 on Isambard"},
        "runner": "NB05 model.predict_action_dist vs scripts/run_gpu.py:predict (same decode; run_gpu.py also "
                  "sets torch.backends.cuda.matmul.allow_tf32 = False, which is PyTorch's default)",
    }
    return out


# --------------------------------------------------------------------------- #
# A2. Units
# --------------------------------------------------------------------------- #
def a2_units(data) -> dict:
    q01, q99 = float(E.Q01[1]), float(E.Q99[1])
    b = data["bridge"]
    gt = b["gt_dy"].astype(float).abs()
    return {
        "dy_q01": q01, "dy_q99": q99,
        "source": "vla.get_action_stats('bridge_orig') via model.describe_action_space: full precision in "
                  "outputs/runs/logs/sgvla-smoke_6987042.out (GH200); the same values to 5 d.p. in the printed "
                  "outputs of NB01 cell 14 and NB05 cell 10 (Colab A100)",
        "one_bin_255": float(E.STEP_255[1]), "one_bin_254": float(E.WIDTH_254[1]),
        "one_bin_mm_if_metres": {"255": 1000 * float(E.STEP_255[1]), "254": 1000 * float(E.WIDTH_254[1])},
        "control_rate_hz": 5,
        "one_bin_mm_per_s_at_5hz": 5 * 1000 * float(E.WIDTH_254[1]),
        "q99_speed_cm_per_s_at_5hz": 5 * 100 * q99,
        "demo_gt_dy_5step_abs": {"median": float(gt.median()), "p90": float(gt.quantile(.9)),
                                 "p99": float(gt.quantile(.99)), "n": int(len(gt)),
                                 "definition": "sum of world_vector[1] over the first 5 steps after the recorded "
                                               "no-op, OXE 'bridge' 0.1.0 release (data.extract_episode)"},
        "episode_steps": {"median": float(b["num_steps"].median()), "q25": float(b["num_steps"].quantile(.25)),
                          "q75": float(b["num_steps"].quantile(.75))},
    }


# --------------------------------------------------------------------------- #
# A3. Typical magnitude of the first lateral action
# --------------------------------------------------------------------------- #
def a3_magnitude(data) -> dict:
    runs, geo, log = data["runs"], data["geo"], data["log"]
    bf = E.wording_rows(runs["replay_bf16"], "baseline")
    mag = bf["cf1"].abs() / E.ONE_BIN
    q25, med, q75 = (float(x) for x in np.percentile(mag, [25, 50, 75]))
    effects = {
        "original_bf16_signed_median": E.evaluate(bf, geo, value="cf1", n_boot=0)["signed_median_bins"],
        "prenominal_bf16_signed_median": E.evaluate(E.wording_rows(runs["ladder_bf16"], "prenominal"), geo,
                                                    value="cf1", n_boot=0)["signed_median_bins"],
        "table_side_bf16_signed_median": E.evaluate(E.wording_rows(runs["ladder_bf16"], "table_side"), geo,
                                                    value="cf1", n_boot=0)["signed_median_bins"],
        "positive_control_bf16_paste_shift_median": E._current_check(runs["replay_bf16"], geo, value="cf1",
                                                                     floor=E.ONE_BIN)["paste_displacement"][
            "median_shift_bins"],
    }
    lg = E.logged_rows(log)["c1"].abs() / E.ONE_BIN
    return {"n": int(len(mag)), "median_bins": med, "iqr_bins": [q25, q75],
            "median_bins_255": float(np.median(bf["cf1"].abs() / E.STEP_255[1])),
            "share_decided": float((mag >= 1).mean()),
            "median_mm_if_metres": med * E.ONE_BIN * 1000,
            "logged4bit_c1_median_bins": float(lg.median()),
            "logged4bit_c1_iqr_bins": [float(lg.quantile(.25)), float(lg.quantile(.75))],
            "effects_bins": effects,
            "effects_as_fraction_of_median": {k: abs(v) / med for k, v in effects.items()}}


# --------------------------------------------------------------------------- #
# A4. Holm across Table 1
# --------------------------------------------------------------------------- #
def holm_family(df: pd.DataFrame, rows) -> dict:
    tests = {}
    for r in rows:
        rec = df.set_index("row").loc[r]
        for name, col in STATS3:
            tests[f"{r}|{name}"] = float(rec[col])
    dec = R.holm(tests)
    adj = E.holm_adjusted(tests)
    surv = sorted([k for k, v in dec.items() if v["reject"]], key=lambda k: tests[k])
    return {"n_tests": len(tests), "survivors": surv, "n_survivors": len(surv),
            "survivor_p": {k: tests[k] for k in surv}, "survivor_adjusted_p": {k: adj[k] for k in surv},
            "first_non_survivor": min((k for k in tests if k not in surv), key=lambda k: tests[k]),
            "first_non_survivor_p": min(v for k, v in tests.items() if k not in surv),
            "first_non_survivor_adjusted_p": adj[min((k for k in tests if k not in surv), key=lambda k: tests[k])]}


def a4_holm(df: pd.DataFrame) -> dict:
    gpu_cmp = df.set_index("row").loc["gpu"]
    out = {"16_rows": holm_family(df, TABLE1_16),
           "14_rows_without_tok_w254": holm_family(df, [r for r in TABLE1_16 if r not in ("tok", "w254")]),
           "17_rows_with_gpu": holm_family(df, TABLE1_16 + ["gpu"]),
           "12_distinct_rows": holm_family(df, [r for r in TABLE1_16 if r not in ("sign", "tok", "w254", "mir_naive")])}
    out["gpu_row_p"] = {"A100": {n: float(gpu_cmp[c]) for n, c in STATS3},
                        "GH200_comparator": {n: float(gpu_cmp[f"cmp_{c}"]) for n, c in STATS3}}
    ref = df.set_index("row")
    dup = {}
    for r, base in (("sign", "ref"), ("tok", "ref"), ("w254", "ref"), ("mir_naive", "mir")):
        dup[r] = {n: [float(ref.loc[r, c]), float(ref.loc[base, c])] for n, c in STATS3}
    out["near_duplicate_p_values"] = dup
    return out


# --------------------------------------------------------------------------- #
# A5. Decomposition of opposite both-correct
# --------------------------------------------------------------------------- #
def a5_decomposition(data, df: pd.DataFrame) -> dict:
    geo = data["geo"]
    table = {}
    idx = df.set_index("row")
    for r in TABLE1_16 + ["gpu", "gpu_cmp"]:
        if r == "gpu_cmp":
            rec = idx.loc["gpu"]
            n, s, k = rec["cmp_opp_n"], rec["cmp_splits"], rec["cmp_splits_right_way"]
            lo_b, hi_b = rec["cmp_opp_both_correct_lo"], rec["cmp_opp_both_correct_hi"]
        else:
            rec = idx.loc[r]
            n, s, k = rec["opp_n"], rec["splits"], rec["splits_right_way"]
            lo_b, hi_b = rec["opp_both_correct_lo"], rec["opp_both_correct_hi"]
        n, s, k = int(n), int(s), int(k)
        table[r] = {"n": n, "splits": s, "right_way": k, "x_split_share": s / n, "x_ci": wilson(s, n),
                    "y_right_way_share": k / s if s else float("nan"), "y_ci": wilson(k, s),
                    "both_correct": k / n, "both_correct_wilson": wilson(k, n),
                    "both_correct_cluster_boot": [float(lo_b), float(hi_b)],
                    "identity_holds": bool(np.isclose((s / n) * (k / s if s else 0), k / n))}
    opp = geo[geo["configuration"] == "opposite"]
    per = opp.groupby("base_scene_id").size()
    ref = E.evaluate(E.wording_rows(data["runs"]["replay_bf16"], "baseline"), geo, value="cf1", n_boot=0)
    out = R.scene_outcomes(E.wording_rows(data["runs"]["replay_bf16"], "baseline").assign(
        c1=lambda f: f["cf1"]), geo)
    res_opp = out[out["resolved"] & (out["configuration"] == "opposite")]
    per_res = res_opp.groupby("base_scene_id").size()
    splits = res_opp[~res_opp["same_sign"]]
    per_split = splits.groupby("base_scene_id").size()
    clustering = {
        "opposite_scenes": int(len(opp)), "frames_with_opposite": int(per.size),
        "opposite_per_frame": {int(k): int(v) for k, v in per.value_counts().sort_index().items()},
        "ref_resolved_opposite": int(len(res_opp)), "ref_frames": int(per_res.size),
        "ref_resolved_opposite_per_frame": {int(k): int(v) for k, v in per_res.value_counts().sort_index().items()},
        "ref_splits": int(len(splits)), "ref_split_frames": int(per_split.size),
        "ref_splits_per_frame": {int(k): int(v) for k, v in per_split.value_counts().sort_index().items()},
        "check_ref_opp_n": ref["opp_n"],
    }
    return {"rows": table, "clustering": clustering}


# --------------------------------------------------------------------------- #
# A6. Label-swap invariance
# --------------------------------------------------------------------------- #
def a6_label_swap(data) -> dict:
    geo = data["geo"]
    rows = E.wording_rows(data["runs"]["replay_bf16"], "baseline")
    swapped = rows.assign(role=rows["role"].map({"a": "b", "b": "a"}))
    rng = np.random.default_rng(20261002)
    half = set(rng.choice(rows["scene_id"].unique(), size=170, replace=False))
    sel = rows["scene_id"].isin(half)
    partial = rows.copy()
    partial.loc[sel, "role"] = partial.loc[sel, "role"].map({"a": "b", "b": "a"})
    pick = lambda e: {"contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],  # noqa: E731
                      "contrast_pairs": e["contrast_pairs"], "contrast_discordant": e["contrast_discordant"],
                      "same_sign": {k: v["same_sign"] for k, v in e["by_layout"].items()},
                      "right_share": {k: v["right_share"] for k, v in e["by_layout"].items()},
                      "splits": f"{e['splits_right_way']}/{e['splits']}", "splits_p": e["splits_p"],
                      "opp_both_correct": e["opp_both_correct"],
                      "signed_median_bins": e["signed_median_bins"], "signed_p": e["signed_p"]}
    o, sw, pa = (pick(E.evaluate(x, geo, value="cf1", n_boot=0)) for x in (rows, swapped, partial))
    checks = {"contrast_unchanged": bool(np.isclose(o["contrast_pts"], sw["contrast_pts"])
                                         and np.isclose(o["contrast_p"], sw["contrast_p"])),
              "same_sign_unchanged": all(np.isclose(o["same_sign"][k], sw["same_sign"][k]) for k in R.CONFIGS),
              "signed_median_negated": bool(np.isclose(o["signed_median_bins"], -sw["signed_median_bins"])),
              "signed_p_unchanged": bool(np.isclose(o["signed_p"], sw["signed_p"])),
              "splits_complemented": o["splits"].split("/")[1] == sw["splits"].split("/")[1] and
              int(o["splits"].split("/")[0]) + int(sw["splits"].split("/")[0]) == int(o["splits"].split("/")[1]),
              "half_swap_contrast_unchanged": bool(np.isclose(o["contrast_pts"], pa["contrast_pts"])
                                                   and np.isclose(o["contrast_p"], pa["contrast_p"]))}
    return {"original": o, "all_swapped": sw, "half_swapped_seeded": pa, "checks": checks}


# --------------------------------------------------------------------------- #
# A7. Wording x run grid
# --------------------------------------------------------------------------- #
RUNS4 = ("4-bit", "4-bit mirror", "bf16", "bf16 mirror")


def a7_grid(data) -> dict:
    g_cf = E.wording_grid(data, "cf1")
    g_c1 = E.wording_grid(data, "c1")
    return {"cf1": g_cf, "c1": g_c1}


def grid_markdown(grid: dict) -> str:
    cf, c1 = grid["cf1"], grid["c1"]

    def cell_splits(k):
        a, b = cf[k], c1[k]
        s = (f"{a['splits_right_way']}/{a['splits']} ({E.pfmt(a['splits_p'])}); "
             f"BC {a['splits_right_way']}/{a['opp_n']} = {100 * a['opp_both_correct']:.1f}%")
        if (a["splits_right_way"], a["splits"], a["opp_n"]) != (b["splits_right_way"], b["splits"], b["opp_n"]):
            s += (f" [c1: {b['splits_right_way']}/{b['splits']}; "
                  f"BC {b['splits_right_way']}/{b['opp_n']} = {100 * b['opp_both_correct']:.1f}%]")
        return s

    def cell_contrast(k):
        a, b = cf[k], c1[k]
        s = f"{a['contrast_pts']:+.1f} ({E.pfmt(a['contrast_p'])}; {a['contrast_pairs']}/{a['contrast_discordant']})"
        if not np.isclose(a["contrast_pts"], b["contrast_pts"]) or not np.isclose(a["contrast_p"], b["contrast_p"]):
            s += f" [c1: {b['contrast_pts']:+.1f} ({E.pfmt(b['contrast_p'])})]"
        return s

    head = "| Wording | " + " | ".join(RUNS4) + " |\n|---|---|---|---|---|"
    t1 = [head] + [f"| {w} | " + " | ".join(cell_splits(f"{r}|{w}") for r in RUNS4) + " |" for w in E.WORDINGS]
    t2 = [head] + [f"| {w} | " + " | ".join(cell_contrast(f"{r}|{w}") for r in RUNS4) + " |" for w in E.WORDINGS]
    return "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n"


# --------------------------------------------------------------------------- #
# A8. Base-frame structure
# --------------------------------------------------------------------------- #
def a8_frames(data) -> dict:
    geo = data["geo"]
    per = geo.groupby("base_scene_id")["configuration"].apply(list)
    rec = geo.groupby("base_scene_id")["recorded_configuration"].apply(list)
    n_opp = per.apply(lambda xs: sum(x == "opposite" for x in xs))
    n_ss = per.apply(lambda xs: sum(x.startswith("same_side") for x in xs))
    two_opp = per[n_opp == 2].index
    return {
        "n_scenes": int(len(geo)), "n_frames": int(per.size),
        "scenes_per_frame": {int(k): int(v) for k, v in per.apply(len).value_counts().sort_index().items()},
        "opposite_per_frame": {int(k): int(v) for k, v in n_opp.value_counts().sort_index().items()},
        "same_side_per_frame": {int(k): int(v) for k, v in n_ss.value_counts().sort_index().items()},
        "frames_one_opposite_one_same_side": int(((n_opp == 1) & (n_ss == 1)).sum()),
        "frames_with_both_ssl_and_ssr": int(per.apply(lambda xs: "same_side_left" in xs and "same_side_right" in xs).sum()),
        "frames_with_two_opposite": [{"base_scene_id": int(f), "hand_labels": per[f], "recorded": rec[f]}
                                     for f in two_opp],
        "recorded_per_frame_opposite": {int(k): int(v) for k, v in rec.apply(
            lambda xs: sum(x == "opposite" for x in xs)).value_counts().sort_index().items()},
    }


# --------------------------------------------------------------------------- #
# A9. Frame-level inference for the signed effect
# --------------------------------------------------------------------------- #
N_FLIP, N_BOOT_MED, SEED = 20000, 4000, 20261002
RUNMAP = {"4-bit": ("replay_nf4", "ladder", "original"), "4-bit mirror": ("replay_nf4", "ladder", "mirror"),
          "bf16": ("replay_bf16", "ladder_bf16", "original"), "bf16 mirror": ("replay_bf16", "ladder_bf16", "mirror")}


def oriented_by_scene(rows, geo, value="cf1", transform="original", unit=E.ONE_BIN):
    """Per-scene signed left-right difference in bins (+ = 'left' more image-left) and its base frame."""
    work = rows.assign(c1=rows[value].astype(float))
    out = R.scene_outcomes(work, geo, mirrored=(transform == "mirror"))
    return out["oriented"].to_numpy() / unit, out["base_scene_id"].to_numpy()


def frame_signflip(d, frames, rng, n_perm=N_FLIP) -> dict:
    """Two-sided sign-flip test of the Wilcoxon statistic T = sum sign(d) rank|d| (zeros dropped), with the
    signs of all scenes in a base frame flipped together. p = (1 + #{|T*| >= |T|}) / (1 + n_perm)."""
    keep = d != 0
    dd, ff = d[keep], frames[keep]
    s = np.sign(dd) * stats.rankdata(np.abs(dd))
    codes, uniq = pd.factorize(ff)
    per_frame = np.bincount(codes, weights=s, minlength=len(uniq))
    t = float(s.sum())
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(uniq)))
    tn = (flips * per_frame).sum(axis=1)   # element-wise: avoids spurious BLAS FP warnings on macOS
    p = (1 + int(np.sum(np.abs(tn) >= abs(t) - 1e-9))) / (1 + n_perm)
    return {"T": t, "p": float(p), "n_nonzero": int(keep.sum()), "n_frames": int(len(uniq))}


def frame_boot_median(d, frames, rng, n_boot=N_BOOT_MED) -> list:
    codes, uniq = pd.factorize(frames)
    groups = [d[codes == i] for i in range(len(uniq))]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return [float(x) for x in np.percentile(meds, [2.5, 97.5])]


def a9_frame_level(data) -> dict:
    runs, geo = data["runs"], data["geo"]
    rng = np.random.default_rng(SEED)
    cells = {}
    for run, (rtag, ltag, tr) in RUNMAP.items():
        for w in E.WORDINGS:
            rows = E.wording_rows(runs[rtag] if w == "baseline" else runs[ltag], w, tr)
            d, f = oriented_by_scene(rows, geo, transform=tr)
            cells[f"{run}|{w}"] = {"median_bins": float(np.median(d)),
                                   "wilcoxon_p_scene": E.A.wilcoxon_paired(d)["p_value"],
                                   **{f"flip_{k}": v for k, v in frame_signflip(d, f, rng).items()},
                                   "median_frame_boot_ci": frame_boot_median(d, f, rng)}
    holm_run = {run: E.holm_adjusted({w: cells[f"{run}|{w}"]["flip_p"] for w in E.WORDINGS}) for run in RUNMAP}
    holm_all = E.holm_adjusted({k: v["flip_p"] for k, v in cells.items()})
    counts = {}
    for w in E.WORDINGS:
        counts[w] = {"raw": sum(cells[f"{r}|{w}"]["flip_p"] < .05 for r in RUNMAP),
                     "holm_within_run": sum(holm_run[r][w] < .05 for r in RUNMAP),
                     "holm_all24": sum(holm_all[f"{r}|{w}"] < .05 for r in RUNMAP),
                     "scene_level_raw": sum(cells[f"{r}|{w}"]["wilcoxon_p_scene"] < .05 for r in RUNMAP)}
    return {"cells": cells, "holm_within_run": holm_run, "holm_all24": holm_all, "counts": counts,
            "settings": {"n_flip": N_FLIP, "n_boot_median": N_BOOT_MED, "seed": SEED, "value": "cf1",
                         "unit": "bins of 0.000325"}}


# --------------------------------------------------------------------------- #
# A10. Placebo: the grab/take paraphrase pair
# --------------------------------------------------------------------------- #
def a10_placebo(data) -> dict:
    runs, geo = data["runs"], data["geo"]
    out = {}
    for prec, ltag in (("bf16", "ladder_bf16"), ("4-bit", "ladder")):
        for tr in ("original", "mirror"):
            rows = R.wording_log(runs[ltag], "paraphrase", tr)
            entry = {"instructions_example": rows.groupby("role")["instruction"].first().to_dict()}
            for label, floor in (("one_bin_254", E.ONE_BIN), ("one_bin_255", E.TABLE1_BIN)):
                e = E.evaluate(rows, geo, value="cf1", images=tr, floor=floor, unit=floor, n_boot=0)
                entry[label] = {"contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                                "pairs": e["contrast_pairs"], "discordant": e["contrast_discordant"],
                                "same_sign": {k: v["same_sign"] for k, v in e["by_layout"].items()}}
            out[f"{prec} {tr}"] = entry
    floor = {}
    for prec, (rtag, ltag) in E.RUN_PAIRS.items():
        base = R.paired(R.wording_log(runs[rtag], R.BASELINE), R.BASELINE, "cf1")
        para = R.paired(runs[ltag][(runs[ltag]["condition"] == "ladder_paraphrase")
                                   & (runs[ltag]["image_transform"] == "original")], "ladder_paraphrase", "cf1")
        j = base.join(para, lsuffix="_lr", rsuffix="_pp").dropna()
        lr = (j["a_lr"] - j["b_lr"]).abs() / E.ONE_BIN
        pp = (j["a_pp"] - j["b_pp"]).abs() / E.ONE_BIN
        floor[prec] = {"n": int(len(j)), "left_right_median_bins": float(lr.median()),
                       "grab_take_median_bins": float(pp.median()), "share_lr_larger": float(np.mean(lr > pp)),
                       "wilcoxon_p": float(stats.wilcoxon(lr, pp).pvalue)}
    out["paraphrase_floor_cf1"] = floor
    return out


# --------------------------------------------------------------------------- #
# A11. Table 1 with the /255 reference vs the /254 reference
# --------------------------------------------------------------------------- #
CMP_COLS = ["opp_n", "opp_both_correct", "opp_same_sign", "right_ssl", "right_opp", "right_ssr", "contrast_pts",
            "contrast_p", "splits_right_way", "splits", "splits_p", "signed_median_bins", "signed_p",
            "ss_left_both_correct", "ss_right_both_correct"]
DISPLAY = {"opp_n": 0, "opp_both_correct": 3, "opp_same_sign": 3, "right_ssl": 2, "right_opp": 2, "right_ssr": 2,
           "contrast_pts": 1, "contrast_p": 3, "splits_right_way": 0, "splits": 0, "splits_p": 3,
           "signed_median_bins": 2, "signed_p": 3, "ss_left_both_correct": 3, "ss_right_both_correct": 3}


def a11_ref255(data, df255: pd.DataFrame) -> dict:
    df254 = E.build_table1(data, n_boot=0, ref_bin="254")
    a = df255[~df255["needs_gpu"].astype(bool)].set_index("row")
    b = df254[~df254["needs_gpu"].astype(bool)].set_index("row")
    pairs = [(r, r) for r in a.index if r in b.index] + [("ref", "w255"), ("w254", "ref")]
    exact, display = [], []
    for ra, rb in pairs:
        for c in CMP_COLS:
            x, y = float(a.loc[ra, c]), float(b.loc[rb, c])
            if not np.isclose(x, y, rtol=0, atol=1e-12):
                exact.append({"row_255": ra, "row_254": rb, "col": c, "ref255": x, "ref254": y})
                if round(x, DISPLAY[c]) != round(y, DISPLAY[c]):
                    display.append({"row_255": ra, "row_254": rb, "col": c, "ref255": x, "ref254": y})
    same_name = [d for d in exact if d["row_255"] == d["row_254"]]
    return {"n_rows_compared": int(len(a)), "cells_differing_exactly": exact,
            "cells_differing_at_display_precision": display,
            "same_named_rows_differing_at_display": [d for d in display if d["row_255"] == d["row_254"]],
            "n_same_named_exact_differences": len(same_name),
            "note": "ref(/255) is compared with ref(/254) and with w255(/254 table); w254(/255 table) with ref(/254)"}


# --------------------------------------------------------------------------- #
# A12. Argmax decisions counted from the zero-motion bin 128
# --------------------------------------------------------------------------- #
def a12_argmax_bin128(data) -> dict:
    geo, runs = data["geo"], data["runs"]
    bf = E.wording_rows(runs["replay_bf16"], "baseline")
    step = E.TABLE1_BIN
    b1 = bf["b1"].astype(int)
    rel = bf.assign(argmax128=(b1 - 128) * step)
    out = {"n": int(len(bf)), "share_bin128": float((b1 == 128).mean()), "k_bin128": int((b1 == 128).sum()),
           "share_bin127": float((b1 == 127).mean()), "share_bin129": float((b1 == 129).mean()),
           "bin128_dy_bins": float((E.NORM_ZERO_DY + step) / step), "variants": {}}
    for k in (0, 1, 2):
        phys = E.evaluate(bf, geo, value="a1", floor=k * step, unit=step, n_boot=0)
        grid = E.evaluate(rel, geo, value="argmax128", floor=(0.0 if k == 0 else (k - 0.5) * step), unit=step,
                          n_boot=0)
        for label, e in (("physical_value", phys), ("steps_from_bin128", grid)):
            c = E.compact(e)
            out["variants"][f"{label}|k={k}"] = {
                "opp_n": c["opp_n"], "opp_both_correct": c["opp_both_correct"], "opp_same_sign": c["opp_same_sign"],
                "right": [c["right_ssl"], c["right_opp"], c["right_ssr"]], "contrast_pts": c["contrast_pts"],
                "contrast_p": c["contrast_p"], "splits": f"{c['splits_right_way']}/{c['splits']}",
                "splits_p": c["splits_p"], "signed_median": c["signed_median_bins"], "signed_p": c["signed_p"],
                "signed_r": c["signed_rank_biserial"], "ties": c["share_exact_ties"]}
    return out


# --------------------------------------------------------------------------- #
# A13. Multiplicity summary
# --------------------------------------------------------------------------- #
# The paper's highlighted claims (main.tex, 2 Oct): (label, wording x run cell, statistic, Table 1 row)
HIGHLIGHTED = [
    ("prenominal, bf16: 20/26 splits the right way", "bf16|prenominal", "splits", "w_prenominal"),
    ("table side, bf16: 3/24 splits the right way", "bf16|table_side", "splits", "w_table_side"),
    ("mirrored, bf16 (original wording): 4/25 splits", "bf16 mirror|baseline", "splits", "mir"),
    ("original wording, bf16: signed -0.15", "bf16|baseline", "signed", "ref"),
    ("prenominal, bf16: signed +0.35", "bf16|prenominal", "signed", "w_prenominal"),
    ("table side, bf16: signed -0.48", "bf16|table_side", "signed", "w_table_side"),
    ("mirrored, bf16 (original wording): signed -0.48", "bf16 mirror|baseline", "signed", "mir"),
    ("original wording, bf16: contrast +10.7 (direction-blind)", "bf16|baseline", "contrast", "ref"),
    ("table side, bf16: contrast +9.6 (direction-blind)", "bf16|table_side", "contrast", "w_table_side"),
    ("table side, 4-bit: contrast +14.8 (direction-blind)", "4-bit|table_side", "contrast", None),
    ("argmax readout, bf16: signed 0.00 (r = -0.16)", None, "signed", "argmax"),
]


def a13_multiplicity(grid_cf: dict, a9: dict, df: pd.DataFrame) -> dict:
    runs = list(RUNMAP)
    stat_p = {"signed": "signed_p", "splits": "splits_p", "contrast": "contrast_p"}
    fam = {}
    # (a) within run over the six wordings: signed (scene, frame) and splits (scene)
    fam["a_signed_scene"] = {r: E.holm_adjusted({w: grid_cf[f"{r}|{w}"]["signed_p"] for w in E.WORDINGS}) for r in runs}
    fam["a_signed_frame"] = a9["holm_within_run"]
    fam["a_splits"] = {r: E.holm_adjusted({w: grid_cf[f"{r}|{w}"]["splits_p"] for w in E.WORDINGS}) for r in runs}
    # (b) the signed effect over all 24 cells
    fam["b_signed_scene"] = E.holm_adjusted({k: v["signed_p"] for k, v in grid_cf.items()})
    fam["b_signed_frame"] = a9["holm_all24"]
    # (c) the contrast within run over the five alternative wordings (and, for reference, all six)
    fam["c_contrast_5alt"] = {r: E.holm_adjusted({w: grid_cf[f"{r}|{w}"]["contrast_p"] for w in E.WORDINGS
                                                  if w != "baseline"}) for r in runs}
    fam["c_contrast_6"] = {r: E.holm_adjusted({w: grid_cf[f"{r}|{w}"]["contrast_p"] for w in E.WORDINGS}) for r in runs}
    # (d) the 48 Table 1 tests (descriptive)
    t1 = df.set_index("row")
    tests = {f"{r}|{n}": float(t1.loc[r, c]) for r in TABLE1_16 for n, c in STATS3}
    fam["d_table1_48"] = E.holm_adjusted(tests)
    summary = {
        "a_rejected": {r: {"signed_scene": [w for w, q in fam["a_signed_scene"][r].items() if q < .05],
                           "signed_frame": [w for w, q in fam["a_signed_frame"][r].items() if q < .05],
                           "splits": [w for w, q in fam["a_splits"][r].items() if q < .05]} for r in runs},
        "b_rejected": {"signed_scene": [k for k, q in fam["b_signed_scene"].items() if q < .05],
                       "signed_frame": [k for k, q in fam["b_signed_frame"].items() if q < .05]},
        "c_rejected": {r: {"five_alternatives": [w for w, q in fam["c_contrast_5alt"][r].items() if q < .05],
                           "six_wordings": [w for w, q in fam["c_contrast_6"][r].items() if q < .05]} for r in runs},
        "d_survivors": [k for k, q in fam["d_table1_48"].items() if q < .05],
    }
    claims = []
    for label, cell, stat, t1row in HIGHLIGHTED:
        run, w = (cell.split("|") if cell else (None, None))
        raw = (grid_cf[cell][stat_p[stat]] if cell else float(t1.loc[t1row, stat_p[stat]]))
        entry = {"claim": label, "statistic": stat, "raw_p_scene": raw}
        if cell and stat == "signed":
            entry["raw_p_frame"] = a9["cells"][cell]["flip_p"]
            entry["a_scene"] = fam["a_signed_scene"][run][w]
            entry["a_frame"] = fam["a_signed_frame"][run][w]
            entry["b_scene"] = fam["b_signed_scene"][cell]
            entry["b_frame"] = fam["b_signed_frame"][cell]
        if cell and stat == "splits":
            entry["a_scene"] = fam["a_splits"][run][w]
        if cell and stat == "contrast":
            entry["c_5alt"] = fam["c_contrast_5alt"][run].get(w, None)   # None: not in the family (baseline)
            entry["c_6"] = fam["c_contrast_6"][run][w]
        if t1row:
            entry["d_table1"] = fam["d_table1_48"][f"{t1row}|{stat}"]
        claims.append(entry)
    # The paper's within-run counts sentence, scene vs frame level
    counts = {w: {"signed_scene": sum(fam["a_signed_scene"][r][w] < .05 for r in runs),
                  "signed_frame": sum(fam["a_signed_frame"][r][w] < .05 for r in runs),
                  "splits": sum(fam["a_splits"][r][w] < .05 for r in runs),
                  "global_signed_scene": sum(fam["b_signed_scene"][f"{r}|{w}"] < .05 for r in runs),
                  "global_signed_frame": sum(fam["b_signed_frame"][f"{r}|{w}"] < .05 for r in runs)}
              for w in E.WORDINGS}
    return {"families": fam, "summary": summary, "claims": claims, "counts": counts}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(E.REPO, "google_drive", "v2"))
    ap.add_argument("--runs", default=os.path.join(E.REPO, "outputs", "runs"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    data = E.load_all(args.data, args.runs)
    df = E.build_table1(data, n_boot=args.n_boot)
    res = {"A1_gpu_only": a1_gpu(data), "A2_units": a2_units(data), "A3_magnitude": a3_magnitude(data),
           "A4_holm_table1": a4_holm(df), "A5_decomposition": a5_decomposition(data, df),
           "A6_label_swap": a6_label_swap(data)}
    grid = a7_grid(data)
    res["A7_grid"] = grid
    res["A7_markdown"] = grid_markdown(grid)
    res["A8_frames"] = a8_frames(data)
    res["A9_frame_level"] = a9_frame_level(data)
    res["A10_placebo"] = a10_placebo(data)
    res["A11_ref255"] = a11_ref255(data, df)
    res["A12_argmax_bin128"] = a12_argmax_bin128(data)
    res["A13_multiplicity"] = a13_multiplicity(grid["cf1"], res["A9_frame_level"], df)
    with open(os.path.join(HERE, "addenda.json"), "w") as f:
        json.dump(E.jclean(res), f, indent=1)
    show = {k: res[k] for k in ("A3_magnitude", "A8_frames", "A10_placebo", "A11_ref255", "A12_argmax_bin128")}
    print(json.dumps(E.jclean(show), indent=1))
    print(json.dumps(E.jclean({"A9_counts": res["A9_frame_level"]["counts"],
                               "A13_summary": res["A13_multiplicity"]["summary"],
                               "A13_counts": res["A13_multiplicity"]["counts"]}), indent=1))
    print(res["A7_markdown"])
    print(f"[done] -> {os.path.join(HERE, 'addenda.json')}")


if __name__ == "__main__":
    main()
