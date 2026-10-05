#!/usr/bin/env python3
"""Ledger §12: results of the 3 Oct A100 runs (BtP plan B4.1 and X2).

    .venv/bin/python docs/btp_paper/evidence/btp_runs_a100.py

Inputs: outputs/btp/a100/*.shard*.csv (pulled from termitech) and the GH200 runs in outputs/runs.
  B4.1  bf16 on an A100: (a) GPU-only effect at bf16 against the GH200 replay and ladder, same
        folded readout; (b) every wording's direction on the A100; (c) O4, the placebo-corrected
        contrast, for all four wordings (placebo pairs: grab/take naming the same twin).
  X2    natural frames, raw vs lower-cased instructions, bf16 and NF4, same A100s.
Writes docs/btp_paper/evidence/a100.json and prints a markdown summary for the ledger.
"""
from __future__ import annotations

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
A100 = os.path.join(REPO, "outputs", "btp", "a100")
STEP = float(E.STEP_255[1])
WORDINGS = ("baseline", "prenominal", "object", "table_side")
PLACEBO = {"baseline": "paraphrase", "prenominal": "placebo_prenominal", "object": "placebo_object",
           "table_side": "placebo_table_side"}
SEED, N_BOOT, N_PERM = 20261003, 4000, 10000


def load(tag, root=A100):
    return R.load_run(os.path.join(root, f"{tag}.csv"))   # folded readout in c* (cf*)


def rows_for(run_replay, run_ladder, wording, transform="original"):
    return E.wording_rows(run_replay if wording == "baseline" else run_ladder, wording, transform)


def compare(ref: pd.DataFrame, new: pd.DataFrame, key) -> dict:
    m = ref.merge(new, on=key, suffixes=("_ref", "_new"))
    both = (m["c1_ref"].abs() >= STEP) & (m["c1_new"].abs() >= STEP)
    return {"n": int(len(m)), "a1_changed": float((m["a1_ref"] != m["a1_new"]).mean()),
            "r_expected": float(np.corrcoef(m["c1_ref"], m["c1_new"])[0, 1]),
            "sign_agreement_decided": float((np.sign(m["c1_ref"]) == np.sign(m["c1_new"]))[both].mean()),
            "median_abs_diff_steps": float(((m["c1_ref"] - m["c1_new"]).abs() / STEP).median())}


def stats_for(rows, geo):
    e = E.evaluate(rows, geo, value="cf1", floor=STEP, unit=STEP, n_boot=0)
    return {k: e[k] for k in ("opp_n", "opp_both_correct", "opp_same_sign", "contrast_pts", "contrast_p",
                              "splits", "splits_right_way", "splits_p", "signed_median_bins", "signed_p")} | {
        "right_share": {k: e["by_layout"][k]["right_share"] for k in R.CONFIGS}}


def same_sign_by_scene(rows, geo):
    out = R.scene_outcomes(rows.assign(c1=rows["cf1"].astype(float)), geo, floor=STEP)
    out = out[out["resolved"]]
    return pd.DataFrame({"frame": out["base_scene_id"].to_numpy(), "layout": out["configuration"].to_numpy(),
                         "same": out["same_sign"].astype(float).to_numpy()}, index=out.index)


def frame_sums(df, frames):
    """Per base frame: same-sign sums and counts for same-side and opposite scenes."""
    ss = df["layout"].str.startswith("same_side")
    out = np.zeros((len(frames), 4))
    idx = {f: i for i, f in enumerate(frames)}
    for (f, is_ss), g in df.groupby([df["frame"], ss]):
        out[idx[f], 0 if is_ss else 2] += g["same"].sum()
        out[idx[f], 1 if is_ss else 3] += len(g)
    return out


def contrast_from(sums):
    """Same-side minus opposite same-sign rate (points) from weighted frame sums (..., 4)."""
    return 100 * (sums[..., 0] / sums[..., 1] - sums[..., 2] / sums[..., 3])


def o4(lr: pd.DataFrame, pl: pd.DataFrame, rng) -> dict:
    """Placebo-corrected contrast: LR contrast minus placebo contrast. Base-frame bootstrap CI (frames
    resampled with replacement) and a permutation test that swaps the two pair labels per base frame."""
    frames = np.array(sorted(set(lr["frame"]) | set(pl["frame"])))
    a, b = frame_sums(lr, frames), frame_sums(pl, frames)
    obs_lr, obs_pl = float(contrast_from(a.sum(0))), float(contrast_from(b.sum(0)))
    w = rng.multinomial(len(frames), np.full(len(frames), 1 / len(frames)), size=N_BOOT).astype(float)
    boots = contrast_from(np.einsum('nf,fk->nk', w, a)) - contrast_from(np.einsum('nf,fk->nk', w, b))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    swap = rng.random((N_PERM, len(frames))) < 0.5
    keep, sw = (~swap).astype(float), swap.astype(float)
    sa = np.einsum('nf,fk->nk', keep, a) + np.einsum('nf,fk->nk', sw, b)
    sb = np.einsum('nf,fk->nk', sw, a) + np.einsum('nf,fk->nk', keep, b)
    null = np.abs(contrast_from(sa) - contrast_from(sb))
    obs = obs_lr - obs_pl
    p = float(max((np.sum(null > abs(obs)) + 0.5 * np.sum(np.isclose(null, abs(obs)))) / N_PERM, 1 / N_PERM))
    return {"contrast_lr": obs_lr, "contrast_placebo": obs_pl, "o4": float(obs),
            "ci": [float(lo), float(hi)], "p_perm_two_sided": p, "n_perm": N_PERM, "n_frames": int(len(frames))}


def main():
    rng = np.random.default_rng(SEED)
    data = E.load_all(os.path.join(REPO, "google_drive", "v2"), os.path.join(REPO, "outputs", "runs"))
    geo = data["geo"]
    gh_replay = R.load_run(os.path.join(REPO, "outputs", "runs", "replay_bf16.csv"))
    gh_ladder = R.load_run(os.path.join(REPO, "outputs", "runs", "ladder_bf16.csv"))
    a_replay, a_ladder = load("replay_bf16_a100"), load("ladder_bf16_a100")
    res = {"b41": {}, "x2": {}}

    # (a) GPU only, bf16, same folded readout
    res["b41"]["gpu_replay"] = compare(gh_replay, a_replay, ["scene_id", "condition", "role", "image_scene_id"])
    ladder_conds = [c for c in a_ladder["condition"].unique() if not c.startswith("ladder_placebo")]
    res["b41"]["gpu_ladder"] = compare(gh_ladder, a_ladder[a_ladder["condition"].isin(ladder_conds)],
                                       ["scene_id", "condition", "role", "image_transform"])
    # (b) per wording, GH200 vs A100
    res["b41"]["wordings"] = {}
    for w in WORDINGS:
        res["b41"]["wordings"][w] = {"gh200": stats_for(rows_for(gh_replay, gh_ladder, w), geo),
                                     "a100": stats_for(rows_for(a_replay, a_ladder, w), geo)}
    # (c) O4 for every wording on the A100; the original wording also on the GH200
    res["b41"]["o4"] = {}
    for w in WORDINGS:
        lr = same_sign_by_scene(rows_for(a_replay, a_ladder, w), geo)
        pl = same_sign_by_scene(R.wording_log(a_ladder, PLACEBO[w], "original"), geo)
        res["b41"]["o4"][f"a100|{w}"] = o4(lr, pl, rng)
    lr = same_sign_by_scene(rows_for(gh_replay, gh_ladder, "baseline"), geo)
    pl = same_sign_by_scene(R.wording_log(gh_ladder, "paraphrase", "original"), geo)
    res["b41"]["o4"]["gh200|baseline"] = o4(lr, pl, rng)

    # X2: raw vs lower-cased natural frames, same A100s
    for prec in ("bf16", "nf4"):
        raw, low = load(f"natural_{prec}_a100"), load(f"natural_lower_{prec}_a100")
        m = raw.merge(low, on=["scene_id", "role", "image_transform"], suffixes=("_raw", "_low"))
        changed = m[m["instruction_raw"] != m["instruction_low"]]
        res["x2"][prec] = {"n": int(len(m)), "text_changed": int(len(changed)),
                           "a1_changed_among_changed": float((changed["a1_raw"] != changed["a1_low"]).mean()),
                           "r_expected_among_changed": float(np.corrcoef(changed["c1_raw"], changed["c1_low"])[0, 1])}
    with open(os.path.join(HERE, "a100.json"), "w") as f:
        json.dump(E.jclean(res), f, indent=2)
    print(json.dumps(E.jclean(res), indent=1)[:6000])


if __name__ == "__main__":
    main()
