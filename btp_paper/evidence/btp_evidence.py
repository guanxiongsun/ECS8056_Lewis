#!/usr/bin/env python3
"""
btp_evidence.py: evidence audit for the BtP workshop paper ("Beneath the probe").

    .venv/bin/python docs/btp_paper/evidence/btp_evidence.py
    .venv/bin/python docs/btp_paper/evidence/btp_evidence.py --data google_drive/v2 --runs outputs/runs

Recomputes, from the logged predictions (`google_drive/v2/probe_predictions.csv`,
4-bit NF4 on a Colab A100) and the GPU runs (`outputs/runs/*.shard*.csv`,
GH200), every number the BtP paper states, and writes next to this file:

  evidence.json        every number, keyed by claim (claim1 ... claim10, table1)
  one_at_a_time.csv    the one-at-a-time sensitivity table (paper Table 1)
  one_at_a_time.md     the same table in markdown

Definitions are the ones in `reanalysis.py` / `analysis.py` (done.md conventions):
lateral channel dy = component 1, image-right negative; "decided" = |dy| >= one
bin = 0.000325 = (q99 - q01) / 254; argmax grid step (q99 - q01) / 255; mirrored
images swap targets as well as negating them. Functions from those modules are
called, not re-implemented; `evaluate` only composes them so that one choice
can be changed at a time. Nothing in the repository is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

import analysis as A  # noqa: E402
import reanalysis as R  # noqa: E402
from compose_scenes import evaluation_scenes, manipulation_rate  # noqa: E402

# --------------------------------------------------------------------------- #
# Action-space constants (bridge_orig), as printed by `describe_action_space`
# on the loaded model: outputs/runs/logs/sgvla-smoke_6987042.out.
# --------------------------------------------------------------------------- #
Q01 = np.array([-0.02872725307941437, -0.04170349963009357, -0.026093858778476715,
                -0.08092105075716972, -0.09288699507713317, -0.20718276381492615, 0.0])
Q99 = np.array([0.028309678435325586, 0.040855254605412394, 0.040161586627364146,
                0.08192047759890528, 0.07792850524187081, 0.20382574498653397, 1.0])
WIDTH_254 = (Q99 - Q01) / 254          # the pipeline's "one bin" (bin_widths divides by n_bins - 1)
STEP_255 = (Q99 - Q01) / 255           # spacing of the de-normalised argmax grid
ONE_BIN = R.ONE_BIN                    # 0.000325, dy
DX_BIN = float(WIDTH_254[0])           # 0.0002246, dx's own one-bin floor
NORM_ZERO_DY = float(0.5 * (Q99[1] + Q01[1]))  # normalised 0 (bin 127) de-normalised: -0.000424

WORDINGS = ("baseline", "prenominal", "object", "table_side", "absent_noun", "move")
RUN_PAIRS = {"nf4": ("replay_nf4", "ladder"), "bf16": ("replay_bf16", "ladder_bf16")}


def jclean(obj):
    """JSON-serialisable copy (numpy scalars, tuples, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): jclean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jclean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def pct(x, nd=1):
    return "nan" if x is None or x != x else f"{100 * x:.{nd}f}%"


def pfmt(p):
    if p is None or p != p:
        return "n/a"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def cluster_mean_ci(frame: pd.DataFrame, col: str, *, cluster: str = "base_scene_id",
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """`R.cluster_bootstrap(frame, mean of col)` with identical draws, computed from
    per-cluster sums instead of concatenating frames (about 500x faster). The run
    checks equality against `R.cluster_bootstrap` once (`check_bootstrap`)."""
    if frame.empty:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    grouped = frame.groupby(cluster)[col]
    sums = grouped.apply(lambda s: s.astype(float).sum())
    counts = grouped.size()
    keys = np.array(list(sums.index))
    pos = {k: i for i, k in enumerate(keys)}
    s, c = sums.to_numpy(float), counts.to_numpy(float)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        idx = np.fromiter((pos[k] for k in sample), int, len(sample))
        draws[b] = s[idx].sum() / c[idx].sum()
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": float(frame[col].astype(float).mean()), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot}


def check_bootstrap(log, geo) -> dict:
    """The fast bootstrap must reproduce R.cluster_bootstrap exactly (one case)."""
    out = R.scene_outcomes(log, geo)
    rows = out[out["resolved"] & (out["configuration"] == "opposite")]
    slow = R.cluster_bootstrap(rows, lambda f: f["both_correct"].astype(float).mean())
    fast = cluster_mean_ci(rows, "both_correct")
    same = bool(np.isclose(slow["lo"], fast["lo"]) and np.isclose(slow["hi"], fast["hi"]))
    if not same:
        raise RuntimeError(f"fast bootstrap disagrees with R.cluster_bootstrap: {slow} vs {fast}")
    return {"checked": "opposite both_correct, logged 4-bit", "slow": slow, "fast": fast, "identical": same}


def binom_p(k: int, n: int) -> float:
    return float(stats.binomtest(int(k), int(n)).pvalue) if n else float("nan")


def wilson(k: int, n: int):
    return R.wilson_interval(int(k), int(n))


def holm_adjusted(pvalues: dict) -> dict:
    """Holm step-down adjusted p values (monotone), alongside R.holm's decisions."""
    names = [k for k, p in pvalues.items() if p is not None and np.isfinite(p)]
    order = sorted(names, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_all(data_dir: str, runs_dir: str) -> dict:
    log = R.load_probe(data_dir)
    constructed = R.load_constructed(data_dir)
    geo = R.scene_geometry(log, constructed)
    runs = {}
    for tag in ("replay_nf4", "replay_bf16", "ladder", "ladder_bf16", "natural", "natural_bf16"):
        try:
            runs[tag] = R.load_run(os.path.join(runs_dir, f"{tag}.csv"), readout="255")  # keeps c* and cf* apart
        except FileNotFoundError:
            print(f"[warn] run {tag} not found in {runs_dir}")
    bridge = R.load_bridge_manifest(data_dir)
    return {"log": log, "constructed": constructed, "geo": geo, "runs": runs, "bridge": bridge,
            "data_dir": data_dir, "runs_dir": runs_dir}


def wording_rows(run: pd.DataFrame, wording: str, transform: str = "original") -> pd.DataFrame:
    """One wording's rows relabelled as the baseline pair (as run_gpu_analysis.wording_table)."""
    if wording == "baseline":
        condition = R.BASELINE if transform == "original" else R.MIRROR
        rows = run[(run["condition"] == condition) & (run["image_transform"] == transform)].copy()
        rows["condition"] = R.BASELINE
        return rows
    return R.wording_log(run, wording, transform)


def logged_rows(log: pd.DataFrame, transform: str = "original") -> pd.DataFrame:
    condition = R.BASELINE if transform == "original" else R.MIRROR
    rows = log[log["condition"] == condition].copy()
    rows["condition"] = R.BASELINE
    return rows


# --------------------------------------------------------------------------- #
# One evaluation, with one choice changeable at a time
# --------------------------------------------------------------------------- #
def evaluate(rows: pd.DataFrame, geo: pd.DataFrame, *, value: str = "cf1", floor: float = ONE_BIN,
             unit: float = ONE_BIN, sign: int = 1, zero: float = 0.0, images: str = "original",
             scoring: str = "correct", scenes=None, n_boot: int = 2000) -> dict:
    """Headline + decisive statistics for one left/right pair set.

    rows     prediction rows of one wording, condition relabelled to `baseline`
    value    column read as the lateral action (c1, cf1, a1, c0, cf0 ...)
    floor    "decided" threshold (|value| >= floor; 0 means nonzero)
    unit     one bin of that channel, for the signed median in bins
    sign     -1 flips the sign convention (image-right = positive)
    zero     subtracted before thresholding (e.g. the normalised-zero bin)
    images   'original' or 'mirror' (the images the model saw)
    scoring  for mirrored images: 'correct' (swap + negate targets; R.scene_outcomes
             mirrored=True), 'naive' (negate each role's own target, as NB06 did),
             'none' (original targets)
    scenes   optional subset of scene ids

    Layout labels are always those of the image the model saw. Composes
    R.scene_outcomes (scoring), A.same_side_test (paired McNemar contrast) and
    A.wilcoxon_paired (signed median); `evaluate` with defaults reproduces
    R.wording_tests (validated in `validate_evaluate`).
    """
    work = rows.copy()
    if scenes is not None:
        work = work[work["scene_id"].isin(set(scenes))]
    work["c1"] = sign * (work[value].astype(float) - zero)
    g = geo if scenes is None else geo[geo.index.isin(set(scenes))]
    orient = 1.0
    if images == "mirror" and scoring == "naive":
        g = g.assign(target_sign_a_image=-g["target_sign_a_image"],
                     target_sign_b_image=-g["target_sign_b_image"])
        orient = -1.0  # the naive scorer expects "left" to go to the image-right twin
    mirrored = images == "mirror" and scoring == "correct"
    out = R.scene_outcomes(work, g, floor=floor, mirrored=mirrored)
    if images == "mirror":
        out["configuration"] = out["configuration"].map(R._MIRROR_CONFIG)
    res = out[out["resolved"]]
    opp = res[res["configuration"] == "opposite"]
    splits = opp[~opp["same_sign"]]
    k = int(splits["both_correct"].sum())
    contrast = A.same_side_test(work, min_magnitude=floor)
    paired = contrast["contrast_paired"]
    oriented = orient * out["oriented"].to_numpy() / unit
    w = A.wilcoxon_paired(oriented)
    by_layout = {}
    for name in R.CONFIGS:
        gl = res[res["configuration"] == name]
        by_layout[name] = {"n": int(len(gl)), "same_sign": float(gl["same_sign"].mean()) if len(gl) else float("nan"),
                           "both_correct": float(gl["both_correct"].mean()) if len(gl) else float("nan"),
                           "right_share": float(gl["right_share"].mean()) if len(gl) else float("nan")}
        if n_boot and len(gl):
            for m in ("same_sign", "both_correct", "right_share"):
                by_layout[name][f"{m}_ci"] = cluster_mean_ci(gl, m, n_boot=n_boot)
    return {
        "n_scenes": int(len(out)), "n_resolved": int(len(res)), "by_layout": by_layout,
        "opp_n": int(len(opp)),
        "opp_both_correct": float(opp["both_correct"].mean()) if len(opp) else float("nan"),
        "opp_same_sign": float(opp["same_sign"].mean()) if len(opp) else float("nan"),
        "contrast_pts": 100 * paired["difference"], "contrast_p": paired["p_value"],
        "contrast_pairs": paired["n_pairs"], "contrast_discordant": paired["n_discordant"],
        "contrast_unpaired_pts": 100 * contrast["contrast"]["difference"],
        "contrast_unpaired_p": contrast["contrast"]["p_value"],
        "splits": int(len(splits)), "splits_right_way": k,
        "splits_share": k / len(splits) if len(splits) else float("nan"),
        "splits_p": binom_p(k, len(splits)),
        "signed_median_bins": float(np.median(oriented)), "signed_p": w["p_value"],
        "signed_rank_biserial": w["rank_biserial"],
        "share_left_more_leftward": float(np.mean(oriented > 0)),
        "share_exact_ties": float(np.mean(oriented == 0)),
    }


def validate_evaluate(data) -> dict:
    """evaluate(value='c1') must reproduce R.wording_tests on every GPU run cell."""
    runs, geo, checked, bad = data["runs"], data["geo"], 0, []
    for precision, (rtag, ltag) in RUN_PAIRS.items():
        if rtag not in runs or ltag not in runs:
            continue
        for transform in ("original", "mirror"):
            for wording in WORDINGS:
                rows = wording_rows(runs[rtag] if wording == "baseline" else runs[ltag], wording, transform)
                ref = R.wording_tests(rows, geo, mirrored=(transform == "mirror"))
                mine = evaluate(rows, geo, value="c1", images=transform, n_boot=0)
                ok = (ref["splits_right_way"] == mine["splits_right_way"]
                      and ref["opposite_splits"] == mine["splits"]
                      and np.isclose(100 * ref["contrast_paired"]["difference"], mine["contrast_pts"])
                      and np.isclose(ref["contrast_paired"]["p_value"], mine["contrast_p"])
                      and np.isclose(ref["oriented_median_bins"], mine["signed_median_bins"])
                      and np.isclose(ref["wilcoxon_p"], mine["signed_p"]))
                checked += 1
                if not ok:
                    bad.append(f"{precision}/{transform}/{wording}")
    if bad:
        raise RuntimeError(f"evaluate() disagrees with R.wording_tests on {bad}")
    return {"cells_checked": checked, "all_identical": True}


def compact(e: dict) -> dict:
    """The outcome statistics of Table 1 from an `evaluate` result."""
    bl = e["by_layout"]
    out = {
        "opp_n": e["opp_n"],
        "opp_both_correct": e["opp_both_correct"],
        "opp_same_sign": e["opp_same_sign"],
        "right_ssl": bl["same_side_left"]["right_share"],
        "right_opp": bl["opposite"]["right_share"],
        "right_ssr": bl["same_side_right"]["right_share"],
        "contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
        "contrast_pairs": e["contrast_pairs"], "contrast_discordant": e["contrast_discordant"],
        "splits_right_way": e["splits_right_way"], "splits": e["splits"],
        "splits_share": e["splits_share"], "splits_p": e["splits_p"],
        "signed_median_bins": e["signed_median_bins"], "signed_p": e["signed_p"],
        "signed_rank_biserial": e["signed_rank_biserial"], "share_exact_ties": e["share_exact_ties"],
        "ss_left_both_correct": bl["same_side_left"]["both_correct"],
        "ss_right_both_correct": bl["same_side_right"]["both_correct"],
    }
    for key, m in (("opp_both_correct", "both_correct"), ("opp_same_sign", "same_sign")):
        ci = bl["opposite"].get(f"{m}_ci")
        out[f"{key}_lo"] = ci["lo"] if ci else float("nan")
        out[f"{key}_hi"] = ci["hi"] if ci else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Claim 1: action frame (dx vs dy) and axis identification
# --------------------------------------------------------------------------- #
def _nb06_check(log, data_dir, value_col):
    """The original NB06 §10 call: compose_scenes.manipulation_rate on the frozen set."""
    scenes = evaluation_scenes(os.path.join(data_dir, "constructed"))
    neutral = log[(log["scene_source"] == "constructed") & (log["condition"] == R.NEUTRAL)].copy()
    neutral["construct_id"] = neutral["scene_id"].astype(str)
    r = manipulation_rate(neutral, scenes, value_col=value_col)
    out = {"n": r["n"], "toward_pasted": r["toward_pasted"], "rate": r["rate"],
           "ci": wilson(r["toward_pasted"], r["n"]), "binom_p_vs_50": binom_p(r["toward_pasted"], r["n"]),
           "by_configuration": {}}
    for name, e in r["by_configuration"].items():
        out["by_configuration"][name] = {"n": e["n"], "toward_pasted": e["toward_pasted"], "rate": e["rate"],
                                         "ci": wilson(e["toward_pasted"], e["n"])}
    return out


def _current_check(log, geo, *, value, floor):
    """reanalysis.manipulation_check and paste_displacement with `value` read as the channel."""
    work = log.assign(c1=log[value].astype(float))
    mc = R.manipulation_check(work, geo, floor=floor)["neutral"]
    mc["toward_paste_k"] = int(round(mc["toward_paste"] * mc["n"]))
    mc["binom_p_vs_50"] = binom_p(mc["toward_paste_k"], mc["n"])
    disp = R.paste_displacement(work, geo)
    # paste_displacement reports bins of ONE_BIN (dy); rescale to the channel's own bin
    scale = ONE_BIN / floor
    disp_out = {"n_frames": disp["n_frames"], "median_shift_bins": disp["median_shift_bins"] * scale,
                "mean_shift_bins": disp["mean_shift_bins"] * scale, "wilcoxon_p": disp["wilcoxon_p"],
                "share_shifted_toward_paste": disp["share_shifted_toward_paste"],
                "mean_ci": [disp["ci_mean_shift"]["lo"] * scale, disp["ci_mean_shift"]["hi"] * scale]}
    return {"manipulation_check": mc, "paste_displacement": disp_out}


def _axis_identification(data):
    """NB03 §15–18 recomputed from object_tracking.csv and the Bridge manifest."""
    base = os.path.join(data["data_dir"], "bridge")
    geo = pd.read_csv(os.path.join(base, "object_tracking_set.csv"))
    review = pd.read_csv(os.path.join(base, "object_tracking_review.csv"))
    geo = geo[geo["scene_id"].isin(set(review.loc[review["approved"] == "yes", "scene_id"]))].copy()
    preds = pd.read_csv(os.path.join(data["data_dir"], "object_tracking.csv"))
    preds = preds[preds["c0"].notna()]
    widths = list(WIDTH_254[:3])
    out = {"n_frames": int(len(geo)), "model": {}, "demonstrations": {}}
    for axis, name in A.TRANSLATION_AXIS_NAMES.items():
        by_sign = A.object_tracking_by_sign(preds, geo, axis_index=axis, min_magnitude=float(widths[axis]))
        entry = {}
        for sign, v in by_sign.items():
            r = v["report"]
            entry[f"sign{sign:+d}"] = {
                "flip_rate": r["flip_rate"], "toward_original": r["toward_original"],
                "toward_mirror": r["toward_mirror"], "n_flip_pairs": int(sum(
                    s["n_flip_decided"] for s in r["by_object_side"].values())),
                "negative_rate_original": r["negative_rate_original"],
                "negative_rate_mirror": r["negative_rate_mirror"],
                "by_side": {k: {"n_decided_original": s["n_decided_original"], "toward_original": s["toward_original"],
                                "n_decided_mirror": s["n_decided_mirror"], "toward_mirror": s["toward_mirror"],
                                "n_flip": s["n_flip_decided"], "flip_rate": s["flip_rate"]}
                            for k, s in r["by_object_side"].items()}}
        out["model"][name] = entry
    geom_gt = A.attach_tracking_ground_truth(geo.to_dict("records"), data["bridge"])
    gt_preds = A.tracking_gt_predictions(geom_gt)
    for axis, name in A.TRANSLATION_AXIS_NAMES.items():
        by_sign = A.object_tracking_by_sign(gt_preds, geom_gt, axis_index=axis, require_mirror=False,
                                            min_magnitude=0.0)
        out["demonstrations"][f"gt_{name}"] = {
            f"sign{sign:+d}": {"toward_original": v["report"]["toward_original"],
                               "by_side": {k: {"n": s["n_decided_original"], "toward": s["toward_original"]}
                                           for k, s in v["report"]["by_object_side"].items()}}
            for sign, v in by_sign.items()}
    return out


def claim1_action_frame(data) -> dict:
    log, geo = data["log"], data["geo"]
    out = {"nb06_original_definition": {}, "current_definition": {}}
    for label, col in (("dx_c0", "c0"), ("dy_c1", "c1"), ("dx_argmax_a0", "a0"), ("dy_argmax_a1", "a1")):
        out["nb06_original_definition"][label] = _nb06_check(log, data["data_dir"], col)
    out["current_definition"]["dy_c1_logged4bit"] = _current_check(log, geo, value="c1", floor=ONE_BIN)
    out["current_definition"]["dx_c0_logged4bit_own_bin"] = _current_check(log, geo, value="c0", floor=DX_BIN)
    out["current_definition"]["dx_c0_logged4bit_dy_bin"] = _current_check(log, geo, value="c0", floor=ONE_BIN)
    if "replay_bf16" in data["runs"]:
        bf = data["runs"]["replay_bf16"]
        out["current_definition"]["dy_cf1_bf16"] = _current_check(bf, geo, value="cf1", floor=ONE_BIN)
        out["current_definition"]["dx_cf0_bf16_own_bin"] = _current_check(bf, geo, value="cf0", floor=DX_BIN)
    out["axis_identification"] = _axis_identification(data)
    out["notes"] = {
        "nb06_informative": "scenes whose recorded instances straddle the recorded gripper x (image-centre "
                            "fallback included), any nonzero action, grouped by the hand-resolved label",
        "current": "hand-resolved opposite scenes; paste side from the order of the two instances; "
                   ">= one bin of the channel; Wilson CI; Fisher test of P(right|paste right) vs P(right|paste left)",
        "dx_bin": DX_BIN, "dy_bin": ONE_BIN,
    }
    return out


# --------------------------------------------------------------------------- #
# Claim 2: readout (argmax vs expected value; bin width /254 vs /255)
# --------------------------------------------------------------------------- #
def claim2_readout(data) -> dict:
    log, geo, runs = data["log"], data["geo"], data["runs"]
    out = {"resolution": R.resolution_corrected(log)}
    step = out["resolution"]["argmax_step"]
    wide = R.paired(log, R.BASELINE, "a1").dropna(subset=["a", "b"])
    steps = np.rint(np.abs(wide["a"] - wide["b"]) / step)
    out["contrast_steps"] = {"n_pairs": int(len(wide)), "ties": int((steps == 0).sum()),
                             "one_step": int((steps == 1).sum()), "two_plus": int((steps >= 2).sum())}
    # The /254 width vs the /255 grid step: which classifications change?
    a1, c1 = log["a1"].abs(), log["c1"].abs()
    out["width_254_vs_255"] = {
        "width_254": float(WIDTH_254[1]), "step_255": float(STEP_255[1]), "data_step": step,
        "one_step_contrasts_below_width": int((steps == 1).sum()),
        "share_one_step_contrasts": float((steps == 1).mean()),
        "argmax_predictions_between_step_and_width": int(((a1 >= STEP_255[1]) & (a1 < ONE_BIN)).sum()),
        "expected_predictions_between_step_and_width": int(((c1 >= STEP_255[1]) & (c1 < ONE_BIN)).sum()),
        "n_predictions": int(len(log)),
    }
    if "replay_bf16" in runs:
        cf = runs["replay_bf16"]["cf1"].abs()
        out["width_254_vs_255"]["bf16_cf1_between_step_and_width"] = int(((cf >= STEP_255[1]) & (cf < ONE_BIN)).sum())
    # Where the argmax grid sits relative to zero (normalised 0 is not physical 0)
    b1 = log["b1"].astype(int)
    grid = {int(k): {"dy": float(NORM_ZERO_DY + (k - 127) * STEP_255[1]),
                     "bins": float((NORM_ZERO_DY + (k - 127) * STEP_255[1]) / ONE_BIN)} for k in range(125, 131)}
    dec = R.decided(log["a1"])
    right = R.image_right(log["a1"])
    out["argmax_grid"] = {
        "normalised_zero_dy": NORM_ZERO_DY, "normalised_zero_bins": NORM_ZERO_DY / ONE_BIN, "grid": grid,
        "share_on_bin127_all": float((b1 == 127).mean()),
        "share_on_bin127_baseline": float((b1[log["condition"] == R.BASELINE] == 127).mean()),
        "share_of_decided_on_bin127": float((b1[dec] == 127).mean()),
        "image_right_share_decided_argmax": float(right[dec].mean()),
        "image_right_share_decided_argmax_without_bin127": float(right[dec & (b1 != 127)].mean()),
        "image_right_share_decided_expected": float(R.image_right(log["c1"])[R.decided(log["c1"])].mean()),
    }
    out["sign_disagreement_argmax_vs_expected"] = R.distribution_diagnostics(log, geo)[
        "sign_disagreement_argmax_vs_expected"]
    # Headline under each readout
    heads = {}
    heads["logged4bit_expected_c1"] = evaluate(logged_rows(log), geo, value="c1")
    heads["logged4bit_argmax_a1"] = evaluate(logged_rows(log), geo, value="a1")
    if "replay_bf16" in runs:
        bf = wording_rows(runs["replay_bf16"], "baseline")
        heads["bf16_expected_cf1"] = evaluate(bf, geo, value="cf1")
        heads["bf16_argmax_a1"] = evaluate(bf, geo, value="a1")
    out["headline_by_readout"] = heads
    return out


# --------------------------------------------------------------------------- #
# Claim 3: token map (31744 folded into bin 254 or left out)
# --------------------------------------------------------------------------- #
def claim3_token_map(data) -> dict:
    runs, log, geo = data["runs"], data["log"], data["geo"]
    per_run, pooled = {}, []
    for tag, run in runs.items():
        if "cf1" not in run:
            continue
        d1 = (run["cf1"] - run["c1"]).abs()
        both = R.decided(run["c1"]) & R.decided(run["cf1"])
        flips = (np.sign(run["c1"]) != np.sign(run["cf1"])) & both
        dec_change = R.decided(run["c1"]) != R.decided(run["cf1"])
        d6 = run["cf6"] - run["c6"]
        per_run[tag] = {
            "n": int(len(run)),
            "dy_changed_ge_1bin": float((d1 >= ONE_BIN).mean()), "dy_changed_ge_1bin_k": int((d1 >= ONE_BIN).sum()),
            "dy_max_change_bins": float(d1.max() / ONE_BIN),
            "dy_decided_sign_flips": float(flips[both].mean()) if both.any() else float("nan"),
            "dy_decided_status_changes": float(dec_change.mean()),
            "dy_min_mass_m1": float(run["m1"].min()), "dy_median_mass_m1": float(run["m1"].median()),
            "gripper_mean_c6_omitted": float(run["c6"].mean()), "gripper_mean_cf6_folded": float(run["cf6"].mean()),
            "gripper_median_c6": float(run["c6"].median()), "gripper_min_c6": float(run["c6"].min()),
            "gripper_share_changed_gt_0.01": float((d6.abs() > 0.01).mean()),
            "gripper_share_changed_gt_0.1": float((d6.abs() > 0.1).mean()),
            "gripper_share_changed_gt_0.5": float((d6.abs() > 0.5).mean()),
            "gripper_share_crossing_0.5": float(((run["c6"] < 0.5) != (run["cf6"] < 0.5)).mean()),
            "gripper_share_c6_below_0.5": float((run["c6"] < 0.5).mean()),
            "gripper_share_cf6_below_0.5": float((run["cf6"] < 0.5).mean()),
            "gripper_share_argmax_open": float((run["a6"] > 0.5).mean()),
        }
        pooled.append(run[["c1", "cf1", "c6", "cf6", "a6"]])
    allr = pd.concat(pooled, ignore_index=True)
    d1, d6 = (allr["cf1"] - allr["c1"]).abs(), (allr["cf6"] - allr["c6"]).abs()
    out = {"per_run": per_run, "pooled": {
        "n": int(len(allr)), "dy_changed_ge_1bin": float((d1 >= ONE_BIN).mean()),
        "gripper_mean_c6_omitted": float(allr["c6"].mean()), "gripper_mean_cf6_folded": float(allr["cf6"].mean()),
        "gripper_share_changed_gt_0.1": float((d6 > 0.1).mean()),
        "gripper_share_crossing_0.5": float(((allr["c6"] < 0.5) != (allr["cf6"] < 0.5)).mean()),
    }}
    # The logged (Colab) predictions have no cf6, but the argmax a6 shows the executed state.
    out["logged4bit"] = {
        "n": int(len(log)), "gripper_mean_c6_omitted": float(log["c6"].mean()),
        "gripper_share_argmax_open": float((log["a6"] > 0.5).mean()),
        "gripper_share_c6_below_0.5": float((log["c6"] < 0.5).mean()),
        "gripper_share_argmax_open_but_c6_below_0.5": float(((log["a6"] > 0.5) & (log["c6"] < 0.5)).mean()),
        "action_mass_min_median": float(log["action_mass_min"].median()),
        "action_mass_min_min": float(log["action_mass_min"].min()),
    }
    # Does the fold change any headline statistic? (bf16 replay, original wording)
    if "replay_bf16" in runs:
        rows = wording_rows(runs["replay_bf16"], "baseline")
        out["headline_bf16"] = {"folded_cf1": compact(evaluate(rows, geo, value="cf1", n_boot=0)),
                                "omitted_c1": compact(evaluate(rows, geo, value="c1", n_boot=0))}
    return out


# --------------------------------------------------------------------------- #
# Claim 4: numerical precision (4-bit NF4 vs bf16) and cross-GPU replay
# --------------------------------------------------------------------------- #
def _compare_tokens(ref: pd.DataFrame, new: pd.DataFrame, key) -> dict:
    """Lateral argmax / expected-value agreement between two runs on identical stimuli."""
    cols = list(key) + ["a1", "c1"] + (["cf1"] if "cf1" in ref and "cf1" in new else [])
    m = ref[cols].merge(new[cols], on=list(key), suffixes=("_ref", "_new"))
    step = float(STEP_255[1])
    d_steps = np.rint(np.abs(m["a1_ref"] - m["a1_new"]) / step)
    out = {"n_matched": int(len(m)),
           "a1_identical": float((d_steps == 0).mean()), "a1_changed": float((d_steps > 0).mean()),
           "a1_changed_k": int((d_steps > 0).sum()),
           "a1_one_step": float((d_steps == 1).mean()), "a1_two_plus": float((d_steps >= 2).mean())}
    for col in ("c1", "cf1"):
        if f"{col}_ref" not in m:
            continue
        x, y = m[f"{col}_ref"], m[f"{col}_new"]
        both = R.decided(x) & R.decided(y)
        out[f"{col}_pearson"] = float(np.corrcoef(x, y)[0, 1])
        out[f"{col}_spearman"] = float(stats.spearmanr(x, y)[0])
        out[f"{col}_sign_agreement_decided"] = float((np.sign(x) == np.sign(y))[both].mean())
        out[f"{col}_n_both_decided"] = int(both.sum())
        out[f"{col}_median_abs_diff_bins"] = float(((x - y).abs() / ONE_BIN).median())
    return out


def _environment(run: pd.DataFrame) -> dict:
    cols = ["gpu_name", "gpu_capability", "dtype", "torch", "transformers", "bitsandbytes", "seed", "do_sample"]
    cols += [c for c in ("precision",) if c in run]
    return {c: sorted(map(str, run[c].dropna().unique())) for c in cols if c in run}


def claim4_precision(data) -> dict:
    log, geo, runs = data["log"], data["geo"], data["runs"]
    key = R.REPLAY_KEY
    out = {"environment": {"logged": _environment(log)}}
    for tag in ("replay_nf4", "replay_bf16"):
        if tag in runs:
            out["environment"][tag] = _environment(runs[tag])
    comps = {}
    if "replay_nf4" in runs:
        comps["logged4bit_A100_vs_nf4_GH200"] = _compare_tokens(log, runs["replay_nf4"], key)
        comps["logged4bit_A100_vs_nf4_GH200_compare_replay"] = R.compare_replay(log, runs["replay_nf4"])
    if "replay_bf16" in runs:
        comps["logged4bit_A100_vs_bf16_GH200"] = _compare_tokens(log, runs["replay_bf16"], key)
        comps["logged4bit_A100_vs_bf16_GH200_compare_replay"] = R.compare_replay(log, runs["replay_bf16"])
    if "replay_nf4" in runs and "replay_bf16" in runs:
        comps["nf4_GH200_vs_bf16_GH200_replay"] = _compare_tokens(runs["replay_nf4"], runs["replay_bf16"], key)
    if "ladder" in runs and "ladder_bf16" in runs:
        comps["nf4_GH200_vs_bf16_GH200_ladder"] = _compare_tokens(
            runs["ladder"], runs["ladder_bf16"], ["scene_id", "condition", "role", "image_transform"])
    if "natural" in runs and "natural_bf16" in runs:
        comps["nf4_GH200_vs_bf16_GH200_natural"] = _compare_tokens(
            runs["natural"], runs["natural_bf16"], ["scene_id", "role", "image_transform"])
    out["comparisons"] = comps
    # How much of each cross-run change in dy comes with a change in the greedy dx token?
    amp = {}
    for label, ref, new in (("logged4bit_A100_vs_nf4_GH200", log, runs.get("replay_nf4")),
                            ("logged4bit_A100_vs_bf16_GH200", log, runs.get("replay_bf16")),
                            ("nf4_GH200_vs_bf16_GH200", runs.get("replay_nf4"), runs.get("replay_bf16"))):
        if ref is None or new is None:
            continue
        m = ref[key + ["b0", "b1", "c1"]].merge(new[key + ["b0", "b1", "c1"]], on=key, suffixes=("_o", "_n"))
        dx = (m["b0_o"] != m["b0_n"]).to_numpy()
        dy = (m["b1_o"] - m["b1_n"]).abs().to_numpy()
        dc = ((m["c1_o"] - m["c1_n"]).abs() / ONE_BIN).to_numpy()
        amp[label] = {"n": int(len(m)), "dx_token_changed": float(dx.mean()), "dy_token_changed": float((dy > 0).mean()),
                      "p_dy_changed_given_dx_same": float((dy[~dx] > 0).mean()),
                      "p_dy_changed_given_dx_changed": float((dy[dx] > 0).mean()),
                      "share_of_dy_2plus_step_changes_with_dx_changed": float(dx[dy >= 2].mean()),
                      "median_abs_dc1_bins_dx_same": float(np.median(dc[~dx])),
                      "median_abs_dc1_bins_dx_changed": float(np.median(dc[dx]))}
    out["dx_prefix_amplification"] = amp
    # Headline under each precision (original wording, original images)
    heads = {"logged4bit_c1": compact(evaluate(logged_rows(log), geo, value="c1"))}
    for prec, (rtag, _) in RUN_PAIRS.items():
        if rtag in runs:
            rows = wording_rows(runs[rtag], "baseline")
            heads[f"{prec}_GH200_cf1"] = compact(evaluate(rows, geo, value="cf1"))
            heads[f"{prec}_GH200_c1"] = compact(evaluate(rows, geo, value="c1"))
            for name in R.CONFIGS:
                heads[f"{prec}_GH200_cf1"][f"same_sign_{name}"] = evaluate(rows, geo, value="cf1", n_boot=0)[
                    "by_layout"][name]["same_sign"]
    for name in R.CONFIGS:
        heads["logged4bit_c1"][f"same_sign_{name}"] = evaluate(logged_rows(log), geo, value="c1", n_boot=0)[
            "by_layout"][name]["same_sign"]
    out["headline"] = heads
    return out


# --------------------------------------------------------------------------- #
# Claim 5: autoregressive prefix (dy is read after the greedy dx token)
# --------------------------------------------------------------------------- #
def _prefix(rows: pd.DataFrame, value: str) -> dict:
    b0 = R.paired(rows, R.BASELINE, "b0").dropna(subset=["a", "b"])
    v = R.paired(rows, R.BASELINE, value).loc[b0.index]
    same = (b0["a"] == b0["b"]).to_numpy()
    dc = ((v["a"] - v["b"]).abs() / ONE_BIN).to_numpy()
    a, b = v["a"].to_numpy(), v["b"].to_numpy()
    dec = R.decided(a) & R.decided(b)
    flip = (a * b < 0) & dec
    mw = stats.mannwhitneyu(dc[~same], dc[same], alternative="two-sided")
    d_tok = np.abs(b0["a"] - b0["b"]).to_numpy()
    return {"n_pairs": int(len(b0)), "dx_token_differs": float((~same).mean()), "dx_token_differs_k": int((~same).sum()),
            "median_abs_dy_change_same_dx": float(np.median(dc[same])),
            "median_abs_dy_change_diff_dx": float(np.median(dc[~same])),
            "mean_abs_dy_change_same_dx": float(np.mean(dc[same])),
            "mean_abs_dy_change_diff_dx": float(np.mean(dc[~same])),
            "share_of_total_abs_dy_change_in_diff_dx_pairs": float(dc[~same].sum() / dc.sum()),
            "mannwhitney_p": float(mw.pvalue),
            "dy_opposite_sign_decided_same_dx": float(flip[same & dec].mean()) if (same & dec).any() else float("nan"),
            "dy_opposite_sign_decided_diff_dx": float(flip[~same & dec].mean()) if (~same & dec).any() else float("nan"),
            "median_dx_token_distance_when_different": float(np.median(d_tok[~same])) if (~same).any() else float("nan")}


def claim5_prefix(data) -> dict:
    log, runs = data["log"], data["runs"]
    out = {"logged4bit_c1": _prefix(logged_rows(log), "c1"),
           "reanalysis_distribution_dx_prefix": R.distribution_diagnostics(log, data["geo"])["dx_prefix"]}
    for prec, (rtag, _) in RUN_PAIRS.items():
        if rtag in runs:
            out[f"{prec}_GH200_cf1"] = _prefix(wording_rows(runs[rtag], "baseline"), "cf1")
    out["how_dy_is_read"] = (
        "scripts/run_gpu.py:predict and model.predict_action_dist call vla.generate(do_sample=False, "
        "output_scores=True) for 7 tokens; scores[1] (dy) is computed after the greedy dx token "
        "(sequences[-7]) has been appended, so c1/cf1 = E[dy | image, prompt, greedy dx token]. "
        "No run marginalises over dx or teacher-forces a common dx token.")
    return out


# --------------------------------------------------------------------------- #
# Claim 6: scoring of mirrored images
# --------------------------------------------------------------------------- #
def _mirror_table(rows, geo, value, floor=ONE_BIN):
    out = {}
    for scoring in ("correct", "naive", "none"):
        e = evaluate(rows, geo, value=value, images="mirror", scoring=scoring, floor=floor)
        out[scoring] = {"by_layout_seen": {k: {m: v[m] for m in ("n", "same_sign", "both_correct", "right_share")}
                                           for k, v in e["by_layout"].items()},
                        "opp_both_correct_ci": e["by_layout"]["opposite"].get("both_correct_ci"),
                        "splits": e["splits"], "splits_right_way": e["splits_right_way"],
                        "splits_share": e["splits_share"], "splits_p": e["splits_p"],
                        "signed_median_bins": e["signed_median_bins"], "signed_p": e["signed_p"]}
    return out


def claim6_mirror(data) -> dict:
    log, geo, runs = data["log"], data["geo"], data["runs"]
    out = {"logged4bit_c1_one_bin": _mirror_table(logged_rows(log, "mirror"), geo, "c1"),
           "logged4bit_c1_nonzero": _mirror_table(logged_rows(log, "mirror"), geo, "c1", floor=0.0),
           "original_images_logged4bit_c1": compact(evaluate(logged_rows(log), geo, value="c1"))}
    if "replay_bf16" in runs:
        out["bf16_cf1_one_bin"] = _mirror_table(wording_rows(runs["replay_bf16"], "baseline", "mirror"), geo, "cf1")
    out["definitions"] = {
        "correct": "targets swapped and negated (t_a, t_b = -t_b, -t_a): after the flip 'left' names the other twin",
        "naive": "each role's own target negated (t_a, t_b = -t_a, -t_b), as Notebook 06 did",
        "none": "original targets kept",
        "layouts": "reported by the layout the model saw (a flipped both-left scene is both-right)"}
    return out


# --------------------------------------------------------------------------- #
# Claim 7: screening and the gripper fallback
# --------------------------------------------------------------------------- #
def claim7_screening(data) -> dict:
    log, geo, runs, man = data["log"], data["geo"], data["runs"], data["constructed"]
    sc = R.screening_by_configuration(data["data_dir"])
    opp = sc["opposite"]
    same_k = sc["same_side_left"]["approved"] + sc["same_side_right"]["approved"]
    same_n = sc["same_side_left"]["screened"] + sc["same_side_right"]["screened"]
    table = [[opp["approved"], opp["screened"] - opp["approved"]], [same_k, same_n - same_k]]
    sc["opposite_vs_same_side"]["fisher_p"] = float(stats.fisher_exact(table)[1])
    sc["opposite_vs_same_side"]["same_side_k_n"] = [same_k, same_n]
    three = [[sc[c]["approved"], sc[c]["screened"] - sc[c]["approved"]] for c in R.CONFIGS]
    sc["three_layouts_chi2_p"] = float(stats.chi2_contingency(three, correction=False)[1])
    reasons = {}
    for c in R.CONFIGS:
        for k, v in sc[c]["reasons"].items():
            reasons[k] = reasons.get(k, 0) + int(v)
    sc["rejection_reasons_total"] = reasons
    sc["rejected_total"] = int(sum(reasons.values()))
    review = pd.read_csv(os.path.join(data["data_dir"], "constructed", "constructed_review.csv"))
    sc["review_rows"] = int(len(review))
    sc["review_undecided"] = int(review["decision"].isna().sum())
    frozen = man[man["frozen"]]
    out = {"screening": sc}
    out["gripper_fallback"] = {
        "frozen_n": int(len(frozen)), "frozen_fallback": int((frozen["gripper_source"] == "image_center").sum()),
        "analysed_n": int(len(geo)), "analysed_fallback": int((~geo["detected_gripper"]).sum()),
        "analysed_by_layout": {c: {"n": int((geo["configuration"] == c).sum()),
                                   "fallback": int(((geo["configuration"] == c) & ~geo["detected_gripper"]).sum())}
                               for c in R.CONFIGS},
        "frozen_excluded_unclear": int(len(frozen) - len(geo)),
    }
    gl = geo["configuration"]
    tab = pd.crosstab(gl, geo["detected_gripper"])
    out["gripper_fallback"]["fallback_vs_layout_chi2_p"] = float(stats.chi2_contingency(tab, correction=False)[1])
    det = geo.index[geo["detected_gripper"]]
    notrel = geo.index[~geo["relabelled"]]
    out["headline_subsets"] = {
        "logged4bit_all": compact(evaluate(logged_rows(log), geo, value="c1")),
        "logged4bit_detected_gripper": compact(evaluate(logged_rows(log), geo, value="c1", scenes=det)),
        "logged4bit_not_relabelled": compact(evaluate(logged_rows(log), geo, value="c1", scenes=notrel)),
    }
    if "replay_bf16" in runs:
        rows = wording_rows(runs["replay_bf16"], "baseline")
        out["headline_subsets"]["bf16_detected_gripper"] = compact(evaluate(rows, geo, value="cf1", scenes=det))
        out["headline_subsets"]["bf16_not_relabelled"] = compact(evaluate(rows, geo, value="cf1", scenes=notrel))
    return out


# --------------------------------------------------------------------------- #
# Claims 8 and 9: the decisive statistic, and the wording effect across runs
# --------------------------------------------------------------------------- #
RUN_LABELS = (("nf4", "original", "4-bit"), ("nf4", "mirror", "4-bit mirror"),
              ("bf16", "original", "bf16"), ("bf16", "mirror", "bf16 mirror"))


def wording_grid(data, value: str) -> dict:
    """compact(evaluate(...)) for every precision x transform x wording, reading `value`."""
    runs, geo, grid = data["runs"], data["geo"], {}
    for prec, transform, label in RUN_LABELS:
        rtag, ltag = RUN_PAIRS[prec]
        if rtag not in runs or ltag not in runs:
            continue
        for wording in WORDINGS:
            rows = wording_rows(runs[rtag] if wording == "baseline" else runs[ltag], wording, transform)
            grid[f"{label}|{wording}"] = compact(evaluate(rows, geo, value=value, images=transform, n_boot=0))
    return grid


def claim8_decisive(grids: dict) -> dict:
    out = {}
    for value, grid in grids.items():
        rows = {}
        for key, e in grid.items():
            rows[key] = {"contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                         "contrast_pairs": e["contrast_pairs"], "contrast_discordant": e["contrast_discordant"],
                         "splits": f"{e['splits_right_way']}/{e['splits']}", "splits_share": e["splits_share"],
                         "splits_p": e["splits_p"], "signed_median_bins": e["signed_median_bins"],
                         "signed_p": e["signed_p"],
                         "contrast_sig_but_splits_backwards": bool(e["contrast_p"] < 0.05 and e["contrast_pts"] > 0
                                                                   and e["splits_share"] < 0.5)}
        out[value] = rows
    return out


def claim9_wording(grids: dict) -> dict:
    out = {}
    for value, grid in grids.items():
        labels = [lab for _, _, lab in RUN_LABELS]
        table, holm_run_splits, holm_run_signed, holm_report = {}, {}, {}, {}
        for lab in labels:
            fam_s = {w: grid[f"{lab}|{w}"]["splits_p"] for w in WORDINGS if f"{lab}|{w}" in grid}
            fam_m = {w: grid[f"{lab}|{w}"]["signed_p"] for w in WORDINGS if f"{lab}|{w}" in grid}
            fam_c = {w: grid[f"{lab}|{w}"]["contrast_p"] for w in WORDINGS if f"{lab}|{w}" in grid and w != "baseline"}
            holm_run_splits[lab] = holm_adjusted(fam_s)
            holm_run_signed[lab] = holm_adjusted(fam_m)
            holm_report[lab] = R.holm(fam_c)   # as run_gpu_analysis: 5 alternative wordings, paired contrast
        glob_m = holm_adjusted({k: e["signed_p"] for k, e in grid.items()})
        glob_s = holm_adjusted({k: e["splits_p"] for k, e in grid.items()})
        for w in WORDINGS:
            cells = {lab: grid.get(f"{lab}|{w}") for lab in labels}
            if any(c is None for c in cells.values()):
                continue
            shares = [cells[l]["splits_share"] for l in labels]
            medians = [cells[l]["signed_median_bins"] for l in labels]
            table[w] = {
                "splits": {l: f"{cells[l]['splits_right_way']}/{cells[l]['splits']}" for l in labels},
                "splits_share": dict(zip(labels, shares)),
                "splits_p": {l: cells[l]["splits_p"] for l in labels},
                "splits_p_holm_run": {l: holm_run_splits[l][w] for l in labels},
                "splits_p_holm_global24": {l: glob_s[f"{l}|{w}"] for l in labels},
                "signed_median_bins": dict(zip(labels, medians)),
                "signed_p": {l: cells[l]["signed_p"] for l in labels},
                "signed_p_holm_run": {l: holm_run_signed[l][w] for l in labels},
                "signed_p_holm_global24": {l: glob_m[f"{l}|{w}"] for l in labels},
                "splits_direction": ["+" if s > 0.5 else ("-" if s < 0.5 else "0") for s in shares],
                "signed_direction": ["+" if m > 0 else ("-" if m < 0 else "0") for m in medians],
                "n_runs_splits_sig_raw": int(sum(cells[l]["splits_p"] < 0.05 for l in labels)),
                "n_runs_splits_sig_holm_run": int(sum(holm_run_splits[l][w] < 0.05 for l in labels)),
                "n_runs_signed_sig_raw": int(sum(cells[l]["signed_p"] < 0.05 for l in labels)),
                "n_runs_signed_sig_holm_run": int(sum(holm_run_signed[l][w] < 0.05 for l in labels)),
                "n_runs_signed_sig_holm_global24": int(sum(glob_m[f"{l}|{w}"] < 0.05 for l in labels)),
                "both_correct_opp": {l: cells[l]["opp_both_correct"] for l in labels},
                "contrast_pts": {l: cells[l]["contrast_pts"] for l in labels},
                "contrast_p": {l: cells[l]["contrast_p"] for l in labels},
            }
        out[value] = {"by_wording": table, "holm_report_paired_contrast": holm_report}
    return out


# --------------------------------------------------------------------------- #
# Claim 10: the worked example's finding
# --------------------------------------------------------------------------- #
def claim10_example(data) -> dict:
    log, geo, runs, bridge = data["log"], data["geo"], data["runs"], data["bridge"]
    out = {"headline_logged4bit": evaluate(logged_rows(log), geo, value="c1")}
    if "replay_bf16" in runs:
        out["headline_bf16_cf1"] = evaluate(wording_rows(runs["replay_bf16"], "baseline"), geo, value="cf1")
    out["paste_displacement"] = R.paste_displacement(log, geo)
    out["manipulation_check_dy"] = R.manipulation_check(log, geo)
    frame = R.design_matrix(log, geo, bridge)
    drv = {}
    # reanalysis._r2's matmul raises spurious floating-point warnings with macOS Accelerate; the
    # values are checked against outputs/reanalysis/results.json below.
    import warnings
    import statsmodels.formula.api as smf
    alt = frame.assign(layout=frame["layout_img"])
    fits = (("all", frame), ("detected_gripper", frame[frame["detected_gripper"]]),
            ("image_centred_detected", alt[alt["detected_gripper"]]))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, f in fits:
            d = R.drivers(f)
            drv[label] = {"n": d["n"], "shapley_r2": d["shapley_r2"],
                          "ols": {k: d["ols"][k] for k in ("layout", "word", "grounded", "demo")}}
            # R.drivers does not expose convergence; refit its mixed model to record it.
            mixed = smf.mixedlm("y ~ layout + word + grounded + source + demo + mirror", f,
                                groups=f["base_scene_id"]).fit(reml=True)
            drv[label]["mixed_model_converged"] = bool(mixed.converged)
    published = os.path.join(REPO, "outputs", "reanalysis", "results.json")
    if os.path.exists(published):
        pub = json.load(open(published))
        same = all(np.isclose(pub[f"drivers_{k}"]["shapley_r2"][t], drv[k]["shapley_r2"][t], atol=1e-9)
                   for k in ("all", "detected_gripper", "image_centred_detected") for t in drv[k]["shapley_r2"])
        drv["matches_published_results_json"] = bool(same)
    drv["note"] = ("Shapley R2 is the LMG decomposition of OLS R2 (reanalysis._r2 uses least squares); "
                   "the mixed model (random intercept per base frame) supplies coefficients only.")
    out["variance_shares"] = drv
    out["within_scene_word_effect"] = R.within_scene_word_effect(frame)
    out["signed_contrast_baseline"] = R.signed_contrast(log, R.BASELINE)
    pf = {}
    for prec, (rtag, ltag) in RUN_PAIRS.items():
        if rtag in runs and ltag in runs:
            pf[prec] = R.paraphrase_floor(runs[rtag], runs[ltag])
    out["paraphrase_floor"] = pf
    la = R.language_audit(bridge)
    out["language_audit"] = {k: v for k, v in la.items() if k not in ("source_examples",)}
    out["role_association"] = R.role_association(bridge)
    return out


# --------------------------------------------------------------------------- #
# The one-at-a-time sensitivity table (Table 1)
# --------------------------------------------------------------------------- #
TABLE1_BIN = float(STEP_255[1])   # Table 1 reference "one bin" since the revision round: the grid step /255
TABLE1_DX_BIN = float(STEP_255[0])


def table1_rows(data, ref_bin: str = "255") -> list:
    """Each row changes exactly one choice from the reference analysis.

    ref_bin='255' (default): one bin = the de-normalised grid step (q99 - q01)/255, and row `w254` takes the
    pipeline's /254 width instead. ref_bin='254' rebuilds the earlier table (reference 0.000325, row `w255`),
    which `btp_evidence_addenda.py` uses to check that no row changes. Rows that do not set `floor`/`unit`
    get the reference bin in `build_table1`.
    """
    if ref_bin == "255":
        tb, dxb = TABLE1_BIN, TABLE1_DX_BIN
        alt = ("w254", "Bin width", "one bin taken as (q99-q01)/254 = 0.000325 instead of the grid step /255 "
               "= 0.0003238", dict(floor=ONE_BIN, unit=ONE_BIN))
        dx_text = "dx (component 0) instead of dy; dx's own one-bin floor (q99-q01)/255 = 0.000224"
    else:
        tb, dxb = ONE_BIN, DX_BIN
        alt = ("w255", "Bin width", "(q99-q01)/255 = 0.0003238 instead of /254 = 0.000325",
               dict(floor=float(STEP_255[1]), unit=float(STEP_255[1])))
        dx_text = "dx (component 0) instead of dy; dx's own one-bin floor 0.000225"
    runs, log, geo = data["runs"], data["log"], data["geo"]
    bf = wording_rows(runs["replay_bf16"], "baseline")
    bf_m = wording_rows(runs["replay_bf16"], "baseline", "mirror")
    nf = wording_rows(runs["replay_nf4"], "baseline")
    det = list(geo.index[geo["detected_gripper"]])
    notrel = list(geo.index[~geo["relabelled"]])
    ref_data = "replay_bf16 (GH200), cf1"
    rows = [
        ("ref", "Reference analysis", "-", ref_data, dict(rows=bf)),
        ("dx", "Action channel", dx_text, "replay_bf16, cf0", dict(rows=bf, value="cf0", floor=dxb, unit=dxb)),
        ("sign", "Sign convention", "image-right = positive dy (flipped)", ref_data, dict(rows=bf, sign=-1)),
        ("zero", "Zero point", "dy measured from normalised zero (bin 127 = -0.000424) instead of physical zero",
         ref_data, dict(rows=bf, zero=NORM_ZERO_DY)),
        ("argmax", "Readout", "argmax token instead of expected value", "replay_bf16, a1", dict(rows=bf, value="a1")),
        ("thr0", "Decision threshold", "any nonzero action counts as decided (0 bins)", ref_data,
         dict(rows=bf, floor=0.0)),
        ("thr2", "Decision threshold", f"2 bins ({2 * tb:.6f})", ref_data, dict(rows=bf, floor=2 * tb)),
        (alt[0], alt[1], alt[2], ref_data, dict(rows=bf, **alt[3])),
        ("tok", "Token map", "token 31744 left out (renormalise over 31745-31999)", "replay_bf16, c1",
         dict(rows=bf, value="c1")),
        ("nf4", "Precision", "4-bit NF4 instead of bf16 (same GH200)", "replay_nf4 (GH200), cf1",
         dict(rows=nf, value="cf1")),
        ("nf4log", "Precision + GPU", "4-bit NF4 logged on Colab A100 (token fold unavailable: c1)",
         "probe_predictions.csv, c1", dict(rows=logged_rows(log), value="c1")),
        ("gpu", "GPU", "Colab A100 instead of GH200, at fixed NF4 precision and the 255-token readout (c1); "
         "comparator is GH200 NF4 c1, not the reference",
         "probe_predictions.csv (A100) vs replay_nf4 (GH200), both c1",
         dict(rows=logged_rows(log), value="c1",
              _cmp=dict(rows=nf, value="c1", label="GH200 NF4 c1", main_label="A100", cmp_label="GH200"))),
        ("mir", "Images (replication)", "mirrored images, targets swapped and negated (correct scoring)",
         "replay_bf16 mirror, cf1", dict(rows=bf_m, images="mirror", scoring="correct")),
        ("mir_naive", "Mirror scoring", "mirrored images, each role's target negated without swapping (NB06)",
         "replay_bf16 mirror, cf1", dict(rows=bf_m, images="mirror", scoring="naive")),
        ("mir_none", "Mirror scoring", "mirrored images, original targets kept", "replay_bf16 mirror, cf1",
         dict(rows=bf_m, images="mirror", scoring="none")),
    ]
    for w, text in (("prenominal", '"pick up the left {noun}"'), ("object", '"pick up the object on the left"'),
                    ("table_side", '"pick up the {noun} on the left side of the table"'),
                    ("absent_noun", '"pick up the {other noun} on the left" (extra)'),
                    ("move", '"move left" / "move right" (extra; no referent)')):
        rows.append((f"w_{w}", "Wording", text, "ladder_bf16, cf1",
                     dict(rows=wording_rows(runs["ladder_bf16"], w))))
    rows += [
        ("sub_det", "Scene subset", "detected-gripper scenes only (199 of 340)", ref_data,
         dict(rows=bf, scenes=det)),
        ("sub_notrel", "Scene subset", "scenes not relabelled by hand (287 of 340)", ref_data,
         dict(rows=bf, scenes=notrel)),
    ]
    return rows


GPU_ROWS = [
    ("gpu_prefix", "Autoregressive prefix", "dy read with a common (teacher-forced) dx token, or marginalised over dx",
     "needs a prefix-control run (scripts/run_gpu.py has no such mode)"),
    ("gpu_screen", "Stimulus screening", "include the 824 rejected (and 1,604 unscreened) composites",
     "needs a GPU run on frames already in google_drive/v2/constructed/frames"),
    ("gpu_dist", "Readout", "median or mode of the full 256-way dy distribution",
     "needs full-distribution logging (CSVs keep argmax, expected value, top prob, entropy, mass only)"),
]


CMP_FIELDS = ["opp_n", "opp_both_correct", "opp_both_correct_lo", "opp_both_correct_hi", "opp_same_sign",
              "opp_same_sign_lo", "opp_same_sign_hi", "right_ssl", "right_opp", "right_ssr", "contrast_pts",
              "contrast_p", "splits_right_way", "splits", "splits_p", "signed_median_bins", "signed_p"]
COMPARATOR = {"ref": "", "mir_naive": "mir", "mir_none": "mir"}


def build_table1(data, n_boot=2000, ref_bin: str = "255") -> pd.DataFrame:
    """One record per Table 1 row. A row with `_cmp` (only `gpu`) is a two-run comparison that cannot be
    expressed against the bf16 reference; its comparator's statistics go in the `cmp_*` columns."""
    tb = TABLE1_BIN if ref_bin == "255" else ONE_BIN
    recs = []
    for rid, choice, change, src, kw in table1_rows(data, ref_bin=ref_bin):
        kw = dict(kw)
        rows = kw.pop("rows")
        cmp_kw = kw.pop("_cmp", None)
        kw.setdefault("floor", tb)
        kw.setdefault("unit", tb)
        e = evaluate(rows, data["geo"], n_boot=n_boot, **kw)
        c = compact(e)
        c.update({"row": rid, "choice": choice, "change": change, "data": src, "needs_gpu": False,
                  "comparator": COMPARATOR.get(rid, "ref")})
        if cmp_kw is not None:
            cmp_kw = dict(cmp_kw)
            cmp_rows = cmp_kw.pop("rows")
            c["comparator"] = cmp_kw.pop("label")
            c["main_label"], c["cmp_label"] = cmp_kw.pop("main_label"), cmp_kw.pop("cmp_label")
            cmp_kw.setdefault("floor", tb)
            cmp_kw.setdefault("unit", tb)
            ce = compact(evaluate(cmp_rows, data["geo"], n_boot=n_boot, **cmp_kw))
            c.update({f"cmp_{k}": ce[k] for k in CMP_FIELDS})
        recs.append(c)
    for rid, choice, change, why in GPU_ROWS:
        recs.append({"row": rid, "choice": choice, "change": change, "data": why, "needs_gpu": True})
    cols = ["row", "choice", "change", "data", "needs_gpu", "comparator", "opp_n", "opp_both_correct", "opp_both_correct_lo",
            "opp_both_correct_hi", "opp_same_sign", "opp_same_sign_lo", "opp_same_sign_hi", "right_ssl", "right_opp",
            "right_ssr", "contrast_pts", "contrast_p", "contrast_pairs", "contrast_discordant", "splits_right_way",
            "splits", "splits_share", "splits_p", "signed_median_bins", "signed_p", "signed_rank_biserial",
            "share_exact_ties", "ss_left_both_correct", "ss_right_both_correct", "main_label", "cmp_label"]
    cols += [f"cmp_{k}" for k in CMP_FIELDS]
    return pd.DataFrame(recs).reindex(columns=cols)


def table1_markdown(df: pd.DataFrame) -> str:
    def p0(x):
        return "" if x != x else f"{100 * x:.0f}"

    def p1(x):
        return "" if x != x else f"{100 * x:.1f}"

    head = ("| Row | Choice changed (one at a time) | Opp. n | Opp. both correct [95% CI] | Opp. same sign [95% CI] | "
            "Image-right SSL / opp / SSR | Same-side − opp. sign-agreement, pts (McNemar p) | "
            "Opp. splits the right way (binomial p) | Signed left−right median, bins (Wilcoxon p) |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if r["needs_gpu"]:
            lines.append(f"| {r['row']} | **{r['choice']}:** {r['change']} | NEEDS GPU | NEEDS GPU | NEEDS GPU | "
                         f"NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU |")
            continue
        label = "**Reference:** dy, expected value (31744 folded), bf16 GH200, original wording, original images, " \
                "all 340 scenes, ≥1 bin (grid step 0.0003238)" if r["row"] == "ref" else f"**{r['choice']}:** {r['change']}"
        if isinstance(r.get("cmp_label"), str) and r.get("cmp_label"):
            m, k = r["main_label"], r["cmp_label"]
            lines.append(
                f"| {r['row']} | {label} | {m} {int(r['opp_n'])}; {k} {int(r['cmp_opp_n'])} | "
                f"{m} {p1(r['opp_both_correct'])} [{p1(r['opp_both_correct_lo'])}, {p1(r['opp_both_correct_hi'])}]; "
                f"{k} {p1(r['cmp_opp_both_correct'])} [{p1(r['cmp_opp_both_correct_lo'])}, "
                f"{p1(r['cmp_opp_both_correct_hi'])}] | "
                f"{m} {p1(r['opp_same_sign'])} [{p1(r['opp_same_sign_lo'])}, {p1(r['opp_same_sign_hi'])}]; "
                f"{k} {p1(r['cmp_opp_same_sign'])} [{p1(r['cmp_opp_same_sign_lo'])}, {p1(r['cmp_opp_same_sign_hi'])}] | "
                f"{m} {p0(r['right_ssl'])} / {p0(r['right_opp'])} / {p0(r['right_ssr'])}; "
                f"{k} {p0(r['cmp_right_ssl'])} / {p0(r['cmp_right_opp'])} / {p0(r['cmp_right_ssr'])} | "
                f"{m} {r['contrast_pts']:+.1f} ({pfmt(r['contrast_p'])}); "
                f"{k} {r['cmp_contrast_pts']:+.1f} ({pfmt(r['cmp_contrast_p'])}) | "
                f"{m} {int(r['splits_right_way'])}/{int(r['splits'])} = {p0(r['splits_share'])}% "
                f"({pfmt(r['splits_p'])}); {k} {int(r['cmp_splits_right_way'])}/{int(r['cmp_splits'])} = "
                f"{p0(r['cmp_splits_right_way'] / r['cmp_splits'])}% ({pfmt(r['cmp_splits_p'])}) | "
                f"{m} {r['signed_median_bins']:+.2f} ({pfmt(r['signed_p'])}); "
                f"{k} {r['cmp_signed_median_bins']:+.2f} ({pfmt(r['cmp_signed_p'])}) |")
            continue
        lines.append(
            f"| {r['row']} | {label} | {int(r['opp_n'])} | {p1(r['opp_both_correct'])} "
            f"[{p1(r['opp_both_correct_lo'])}, {p1(r['opp_both_correct_hi'])}] | {p1(r['opp_same_sign'])} "
            f"[{p1(r['opp_same_sign_lo'])}, {p1(r['opp_same_sign_hi'])}] | {p0(r['right_ssl'])} / "
            f"{p0(r['right_opp'])} / {p0(r['right_ssr'])} | {r['contrast_pts']:+.1f} ({pfmt(r['contrast_p'])}) | "
            f"{int(r['splits_right_way'])}/{int(r['splits'])} = {p0(r['splits_share'])}% ({pfmt(r['splits_p'])}) | "
            f"{r['signed_median_bins']:+.2f} ({pfmt(r['signed_p'])})"
            + (f"; r = {r['signed_rank_biserial']:+.2f}, {100 * r['share_exact_ties']:.0f}% ties"
               if abs(r["signed_median_bins"]) < 0.005 else "") + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Inventory for planning further GPU experiments
# --------------------------------------------------------------------------- #
def inventory(data) -> dict:
    import glob
    out = {}
    readout = {}
    for tag, run in data["runs"].items():
        cols = list(run.columns)
        per_dim = sorted({c.rstrip("0123456789") for c in cols if c[-1:].isdigit() and c[0] in "abcpfhm"
                          and c.rstrip("0123456789") in ("a", "b", "c", "cf", "p", "h", "m")})
        wide = [c for c in cols if any(t in c.lower() for t in ("dist", "prob", "logit", "tok_"))]
        readout[tag] = {"n_rows": int(len(run)), "n_columns": len(cols), "per_dimension_fields": per_dim,
                        "distribution_like_columns": wide}
    out["gpu_csv_readout"] = readout
    # Cross-device determinism: identical stimuli that landed on different shards (GPUs)
    det = {}
    for tag in data["runs"]:
        files = sorted(glob.glob(os.path.join(data["runs_dir"], f"{tag}.shard*.csv")))
        run = pd.concat([pd.read_csv(f).assign(_shard=i) for i, f in enumerate(files)], ignore_index=True)
        k = ["image_scene_id", "image_transform", "instruction"]
        dup = run[run.duplicated(k, keep=False)]
        groups = list(dup.groupby(k))
        det[tag] = {"duplicate_groups": len(groups),
                    "groups_across_shards": int(sum(g["_shard"].nunique() > 1 for _, g in groups)),
                    "groups_with_any_difference": int(sum(
                        (g[[f"a{i}" for i in range(7)] + [f"c{i}" for i in range(7)]].nunique() > 1).any()
                        for _, g in groups))}
    out["cross_gpu_duplicates"] = det
    # Frames on disk for every screened (and unscreened) composite
    cdir = os.path.join(data["data_dir"], "constructed")
    review = pd.read_csv(os.path.join(cdir, "constructed_review.csv"))
    frames = {f[:-4] for f in os.listdir(os.path.join(cdir, "frames")) if f.endswith(".png")}
    review["decision"] = review["decision"].fillna("unscreened")
    out["constructed_frames"] = {
        d: {"n": int((review["decision"] == d).sum()),
            "with_frame": int(review.loc[review["decision"] == d, "construct_id"].isin(frames).sum())}
        for d in ("approved", "rejected", "unscreened")}
    out["constructed_frames"]["frozen_with_frame"] = int(data["constructed"].loc[
        data["constructed"]["frozen"], "construct_id"].isin(frames).sum())
    man = data["constructed"]
    out["constructed_frames"]["manifest_has"] = [c for c in ("instr_a", "instr_b", "configuration",
                                                             "target_sign_a_image", "target_sign_b_image",
                                                             "x_source", "x_pasted", "x_gripper", "gripper_source",
                                                             "human_configuration") if c in man]
    out["constructed_frames"]["human_label_rows"] = int(man["human_configuration"].notna().sum())
    # Ladder stimuli that coincide with another wording
    lad = data["runs"].get("ladder_bf16")
    if lad is not None:
        geo = data["geo"]
        ab = lad[(lad["condition"] == "ladder_absent_noun") & (lad["role"] == "a")
                 & (lad["image_transform"] == "original")]
        out["ladder_overlaps"] = {
            "scenes_whose_noun_is_object": int((geo["noun"] == "object").sum()),
            "absent_noun_drew_object": int((ab["ladder_other_noun"] == "object").sum()),
            "note": "for these scenes the object wording equals the original wording, or the absent-noun "
                    "prompt equals the object wording (the noun 'object' is in the probe's noun list)"}
    out["run_gpu_py"] = {
        "subcommands": ["smoke", "replay", "ladder", "natural"],
        "options": ["--precision {nf4,bf16}", "--out", "--shard/--nshards", "--n (smoke gate size)",
                    "--limit", "--conditions (replay)", "--mirror (ladder, natural)"],
        "batching": "none: predict() (scripts/run_gpu.py:73-110) tokenises one prompt, pops attention_mask "
                    "(l.81), appends EMPTY_TOKEN_ID, and calls vla.generate on a batch of 1 (l.88-89); "
                    "items are sharded across GPUs by index modulo nshards (l.134), each GPU running batch 1",
        "decoding": "greedy, 7 tokens, output_scores; dy scores are conditioned on the greedy dx token",
        "logged_per_dimension": "a (argmax action), b (argmax bin), c (EV, 31744 omitted), cf (EV, 31744 folded), "
                                "p (top prob, 255-token slice), h (entropy, 255-token slice), m (mass on 256 "
                                "action tokens); the 7x256 probabilities exist in memory (l.103) but are not saved",
    }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "google_drive", "v2"))
    ap.add_argument("--runs", default=os.path.join(REPO, "outputs", "runs"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    data = load_all(args.data, args.runs)
    res = {"meta": {"data": os.path.relpath(args.data, REPO), "runs": os.path.relpath(args.runs, REPO),
                    "n_boot": args.n_boot, "one_bin": ONE_BIN, "dx_bin": DX_BIN, "step_255": float(STEP_255[1]),
                    "normalised_zero_dy": NORM_ZERO_DY}}
    res["checks"] = {"bootstrap": check_bootstrap(data["log"], data["geo"]), "evaluate": validate_evaluate(data)}
    print(f"[checks] {res['checks']['evaluate']}; bootstrap identical: {res['checks']['bootstrap']['identical']}")
    res["claim1_action_frame"] = claim1_action_frame(data)
    res["claim2_readout"] = claim2_readout(data)
    res["claim3_token_map"] = claim3_token_map(data)
    res["claim4_precision"] = claim4_precision(data)
    res["claim5_prefix"] = claim5_prefix(data)
    res["claim6_mirror"] = claim6_mirror(data)
    res["claim7_screening"] = claim7_screening(data)
    grids = {"c1_as_reported": wording_grid(data, "c1"), "cf1_convention": wording_grid(data, "cf1")}
    res["wording_grid"] = grids
    res["claim8_decisive"] = claim8_decisive(grids)
    res["claim9_wording"] = claim9_wording(grids)
    res["claim10_example"] = claim10_example(data)
    res["inventory"] = inventory(data)
    df = build_table1(data, n_boot=args.n_boot)
    res["table1"] = df.to_dict(orient="records")
    df.to_csv(os.path.join(args.out, "one_at_a_time.csv"), index=False, float_format="%.6g")
    with open(os.path.join(args.out, "one_at_a_time.md"), "w") as f:
        f.write("# One-at-a-time sensitivity table (BtP Table 1)\n\n"
                "Generated by `btp_evidence.py`. Each row changes one choice from the reference. "
                "Opp. = opposite scenes; SSL/SSR = both twins left/right of the gripper (layout the model saw). "
                "Rates over scenes where both actions are decided; CIs are base-frame cluster bootstraps "
                f"({args.n_boot} draws). Contrast: paired within base frame, exact McNemar. Splits: opposite "
                "scenes where the two instructions move apart; right way = each toward its own twin (binomial vs "
                "50%). Signed median over all 340 pairs, + = 'left' more image-left (Wilcoxon; r = rank-biserial, shown "
                "when the median is a tie).\n\nNotes: on the dx row 'image-right' means negative dx (the dy sign "
                "convention applied to dx, as NB06 did) and the signed median is in dx bins. The zero-point row is "
                "an addition to the plan's list: OpenVLA's normalised zero (bin 127) de-normalises to -0.000424 "
                "(-1.3 bins) because q01/q99 of dy are asymmetric; physical zero is the correct reference. One bin is the "
                "grid step (q99 - q01)/255 = 0.0003238 (row w254 uses the /254 width 0.000325; the table is unchanged). "
                "Row gpu "
                "is a two-run comparison at fixed NF4 precision and fixed 255-token readout (c1): A100 values first, "
                "GH200 values after the semicolon (it shares its A100 values with row nf4log). Rows "
                "marked NEEDS GPU cannot be computed from the logged outputs.\n\n")
        f.write(table1_markdown(df))
    with open(os.path.join(args.out, "evidence.json"), "w") as f:
        json.dump(jclean(res), f, indent=1)
    print(table1_markdown(df))
    print(f"[done] {time.time() - t0:.0f}s -> {args.out}/evidence.json, one_at_a_time.csv, one_at_a_time.md")


if __name__ == "__main__":
    main()
