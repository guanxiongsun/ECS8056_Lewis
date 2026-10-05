#!/usr/bin/env python3
"""Ledger §13: BtP plan B5 (pre-processing), B4 (compute ladder) and B6 (screened-out scenes).

    .venv/bin/python docs/btp_paper/evidence/btp_runs_round2.py

Inputs: outputs/btp/a100/*.shard*.csv. Every setting is compared with B4.1 (bf16, eager, A100, same
stimuli), never with the GH200 reference. Writes docs/btp_paper/evidence/round2.json.
  B5  pre_official    OpenVLA's Bridge evaluation pre-processing applied to our frames
      pre_bridgeorig  256x256 first (an approximation of the training release), then the same
  B4  fp4_official (OpenVLA's load_in_4bit default), int8, fp16, fp32, bf16 with sdpa attention,
      and a bf16 repeat on the same GPUs (determinism)
  B6  approved vs rejected composites, both scored by recorded arrangement on the same A100s
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
SETTINGS = ("pre_official", "pre_bridgeorig", "fp4_official", "int8", "fp16", "fp32", "bf16sdpa", "repeat")
FILES = {"pre_official": ("replay_bf16_a100_pre_official", "ladder_bf16_a100_pre_official"),
         "pre_bridgeorig": ("replay_bf16_a100_pre_bridgeorig", "ladder_bf16_a100_pre_bridgeorig"),
         "fp4_official": ("replay_fp4_official_a100", "ladder_fp4_official_a100"),
         "int8": ("replay_int8_a100", "ladder_int8_a100"), "fp16": ("replay_fp16_a100", "ladder_fp16_a100"),
         "fp32": ("replay_fp32_a100", "ladder_fp32_a100"), "bf16sdpa": ("replay_bf16sdpa_a100", "ladder_bf16sdpa_a100"),
         "repeat": ("replay_bf16_a100_repeat", None)}


def load(tag):
    try:
        return R.load_run(os.path.join(A100, f"{tag}.csv"))
    except FileNotFoundError:
        return None


def compare(ref, new, key):
    m = ref.merge(new, on=key, suffixes=("_ref", "_new"))
    both = (m["c1_ref"].abs() >= STEP) & (m["c1_new"].abs() >= STEP)
    return {"n": int(len(m)), "a1_changed": float((m["a1_ref"] != m["a1_new"]).mean()),
            "a_all_identical": float(np.all([m[f"a{i}_ref"] == m[f"a{i}_new"] for i in range(7)], axis=0).mean()),
            "r_expected": float(np.corrcoef(m["c1_ref"], m["c1_new"])[0, 1]),
            "max_abs_diff_steps": float(((m["c1_ref"] - m["c1_new"]).abs() / STEP).max()),
            "sign_agreement_decided": float((np.sign(m["c1_ref"]) == np.sign(m["c1_new"]))[both].mean())}


def wording_stats(replay, ladder, geo):
    out = {}
    for w in WORDINGS:
        run = replay if w == "baseline" else ladder
        if run is None:
            continue
        rows = E.wording_rows(run, w, "original")
        if rows.empty:
            continue
        e = E.evaluate(rows, geo, value="cf1", floor=STEP, unit=STEP, n_boot=0)
        out[w] = {"opp_both_correct": e["opp_both_correct"], "opp_same_sign": e["opp_same_sign"],
                  "right_share": [e["by_layout"][k]["right_share"] for k in R.CONFIGS],
                  "contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                  "splits": f"{e['splits_right_way']}/{e['splits']}", "splits_p": e["splits_p"],
                  "signed_median": e["signed_median_bins"], "signed_p": e["signed_p"],
                  "direction": ("toward" if e["signed_median_bins"] > 0 else "away") if e["signed_p"] < .05 else "null"}
    return out


def manifest_geo(data_dir):
    man = pd.read_csv(os.path.join(data_dir, "constructed", "constructed_manifest.csv")).set_index("construct_id")
    geo = man[["base_scene_id", "configuration", "target_sign_a_image", "target_sign_b_image", "image_width",
               "x_source", "x_pasted", "x_gripper", "gripper_source", "separation_px", "noun"]].copy()
    geo["recorded_configuration"] = geo["configuration"]
    geo["human_configuration"] = np.nan
    geo["relabelled"] = False
    geo["x_left"] = geo[["x_source", "x_pasted"]].min(axis=1)
    geo["x_right"] = geo[["x_source", "x_pasted"]].max(axis=1)
    geo["paste_is_right"] = geo["x_pasted"] > geo["x_source"]
    geo["detected_gripper"] = geo["gripper_source"] == "detected"
    geo.index.name = "scene_id"
    return geo


def main():
    data = E.load_all(os.path.join(REPO, "google_drive", "v2"), os.path.join(REPO, "outputs", "runs"))
    geo = data["geo"]
    base_replay, base_ladder = load("replay_bf16_a100"), load("ladder_bf16_a100")
    conds = ["baseline", "neutral", "mirror", "mirror_neutral"]
    res = {"reference": "B4.1: bf16, eager, A100 (same stimuli)", "settings": {},
           "b41_wordings": wording_stats(base_replay, base_ladder, geo)}
    for s in SETTINGS:
        rtag, ltag = FILES[s]
        rep, lad = load(rtag), (load(ltag) if ltag else None)
        if rep is None:
            continue
        cell = {"replay_vs_b41": compare(base_replay[base_replay["condition"].isin(conds)], rep,
                                         ["scene_id", "condition", "role", "image_scene_id"])}
        if lad is not None:
            lad_conds = [c for c in lad["condition"].unique()]
            ref = base_ladder[(base_ladder["condition"].isin(lad_conds)) & (base_ladder["image_transform"] == "original")]
            cell["ladder_vs_b41"] = compare(ref, lad, ["scene_id", "condition", "role", "image_transform"])
        cell["wordings"] = wording_stats(rep, lad, geo)
        res["settings"][s] = cell

    man = load("manifest_bf16_a100")
    if man is not None:
        mgeo = manifest_geo(os.path.join(REPO, "google_drive", "v2"))
        man = man.copy()
        man["condition"] = R.BASELINE
        res["b6"] = {}
        for decision in ("approved", "rejected"):
            rows = man[man["screen_decision"] == decision]
            e = E.evaluate(rows, mgeo, value="cf1", floor=STEP, unit=STEP, n_boot=2000)
            res["b6"][decision] = {"n_scenes": e["n_scenes"], "opp_n": e["opp_n"],
                                   "opp_both_correct": e["opp_both_correct"],
                                   "opp_both_correct_ci": e["by_layout"]["opposite"].get("both_correct_ci"),
                                   "opp_same_sign": e["opp_same_sign"],
                                   "right_share": [e["by_layout"][k]["right_share"] for k in R.CONFIGS],
                                   "contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                                   "splits": f"{e['splits_right_way']}/{e['splits']}", "splits_p": e["splits_p"],
                                   "signed_median": e["signed_median_bins"], "signed_p": e["signed_p"]}
    with open(os.path.join(HERE, "round2.json"), "w") as f:
        json.dump(E.jclean(res), f, indent=2)
    for s, cell in res["settings"].items():
        c = cell["replay_vs_b41"]
        dirs = {w: v["direction"] for w, v in cell["wordings"].items()}
        print(f"{s:15s} replay a1 changed {c['a1_changed']:.1%} r={c['r_expected']:.3f} | directions {dirs}")
    if "b6" in res:
        for d, v in res["b6"].items():
            print(f"B6 {d:9s} scenes {v['n_scenes']} opp both-correct {v['opp_both_correct']:.3f} same-sign {v['opp_same_sign']:.3f} "
                  f"right {['%.2f' % x for x in v['right_share']]} splits {v['splits']} signed {v['signed_median']:+.2f} (p={v['signed_p']:.3g})")


if __name__ == "__main__":
    main()
