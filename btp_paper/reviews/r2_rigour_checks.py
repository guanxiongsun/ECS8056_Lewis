#!/usr/bin/env python3
"""
r2_rigour_checks.py: extra checks for the R2 rigour audit of the BtP draft.

    cd /Users/s3057498/code/ECS8056_Lewis && .venv/bin/python docs/btp_paper/reviews/r2_rigour_checks.py

Imports `btp_evidence.py` (read-only) and recomputes, from the same data, numbers
that are not in `evidence.json` but that the audit needs:

  A  base-frame structure of the 340 scenes (clusters behind every test)
  B  cluster-level sign-flip p-values for the signed effect (O1) next to the
     scene-level Wilcoxon p the draft reports
  C  the argmax signed-effect p with zeros kept (Pratt) vs dropped (Wilcox)
  D  Holm within run for the sign-agreement contrast (six wordings; five alternatives)
  E  the bf16 replay predictions that fall between the /255 step and the /254 width
  F  the argmax zero-motion bin under the "any nonzero" threshold
  G  a placebo: the sign-agreement contrast on the grab/take paraphrase pair
  H  the 24-cell count of "significant positive contrast with backwards splits"
  I  the values missing from the draft's Table VII
  J  misc. (object 4-bit-mirror p, paraphrase floor source, "both" subset size)
  K  the label-flip invariance of the sign-agreement contrast (exact)

Writes `r2_rigour_checks.json` next to this file. Modifies nothing else.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = "/Users/s3057498/code/ECS8056_Lewis"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "btp_paper", "evidence"))
os.chdir(REPO)

import analysis as A  # noqa: E402
import btp_evidence as E  # noqa: E402
import reanalysis as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20261002)
N_PERM = 20000


def clean(o):
    return E.jclean(o)


# --------------------------------------------------------------------------- #
data = E.load_all("google_drive/v2", "outputs/runs")
log, geo, runs = data["log"], data["geo"], data["runs"]
out: dict = {}

# A. base-frame structure ---------------------------------------------------- #
g = geo.copy()
per_frame = g.groupby("base_scene_id")["configuration"].apply(list)
n_by_frame = per_frame.apply(len)
opp_per_frame = per_frame.apply(lambda xs: sum(x == "opposite" for x in xs))
ss_per_frame = per_frame.apply(lambda xs: sum(x.startswith("same_side") for x in xs))
out["A_clusters"] = {
    "n_scenes": int(len(g)), "n_base_frames": int(g["base_scene_id"].nunique()),
    "scenes_per_frame": {int(k): int(v) for k, v in n_by_frame.value_counts().sort_index().items()},
    "opposite_scenes_per_frame": {int(k): int(v) for k, v in opp_per_frame.value_counts().sort_index().items()},
    "same_side_scenes_per_frame": {int(k): int(v) for k, v in ss_per_frame.value_counts().sort_index().items()},
    "frames_with_both_ssl_and_ssr": int(per_frame.apply(
        lambda xs: ("same_side_left" in xs) and ("same_side_right" in xs)).sum()),
    "detected_and_not_relabelled": int((g["detected_gripper"] & ~g["relabelled"].astype(bool)).sum()),
}


# B. cluster sign-flip for the signed effect --------------------------------- #
def oriented_table(rows, value="cf1", images="original", scoring="correct", unit=E.ONE_BIN, scenes=None):
    work = rows.copy()
    if scenes is not None:
        work = work[work["scene_id"].isin(set(scenes))]
    work["c1"] = work[value].astype(float)
    gg = geo if scenes is None else geo[geo.index.isin(set(scenes))]
    mirrored = images == "mirror" and scoring == "correct"
    o = R.scene_outcomes(work, gg, mirrored=mirrored)
    return pd.DataFrame({"d": o["oriented"].to_numpy() / unit,
                         "frame": o["base_scene_id"].to_numpy()}, index=o.index)


def signflip_p(tab: pd.DataFrame, level: str, n_perm=N_PERM):
    """Two-sided sign-flip test of the Wilcoxon statistic T = sum sign(d) rank|d| (zeros dropped)."""
    t = tab[tab["d"] != 0]
    ranks = stats.rankdata(np.abs(t["d"].to_numpy()))
    sgn = np.sign(t["d"].to_numpy())
    T = float((sgn * ranks).sum())
    if level == "scene":
        flips = RNG.choice([-1.0, 1.0], size=(n_perm, len(t)))
    else:
        codes, uniq = pd.factorize(t["frame"])
        fl = RNG.choice([-1.0, 1.0], size=(n_perm, len(uniq)))
        flips = fl[:, codes]
    Tn = (flips * (sgn * ranks)).sum(axis=1)
    p = (np.sum(np.abs(Tn) > abs(T)) + 0.5 * np.sum(np.abs(Tn) == abs(T)) + 1) / (n_perm + 1)
    return {"T": T, "p": float(p), "n_nonzero": int(len(t)), "n_frames": int(t["frame"].nunique())}


def cluster_boot_median(tab: pd.DataFrame, n_boot=4000):
    frames = tab["frame"].unique()
    groups = {f: tab.loc[tab["frame"] == f, "d"].to_numpy() for f in frames}
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(frames, size=len(frames), replace=True)
        meds[b] = np.median(np.concatenate([groups[f] for f in pick]))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return [float(lo), float(hi)]


cases = {
    "ref_bf16_original": (E.wording_rows(runs["replay_bf16"], "baseline"), "cf1", "original"),
    "nf4_GH200_original": (E.wording_rows(runs["replay_nf4"], "baseline"), "cf1", "original"),
    "mir_bf16": (E.wording_rows(runs["replay_bf16"], "baseline", "mirror"), "cf1", "mirror"),
    "prenominal_bf16": (E.wording_rows(runs["ladder_bf16"], "prenominal"), "cf1", "original"),
    "object_bf16": (E.wording_rows(runs["ladder_bf16"], "object"), "cf1", "original"),
    "table_side_bf16": (E.wording_rows(runs["ladder_bf16"], "table_side"), "cf1", "original"),
    "absent_noun_bf16": (E.wording_rows(runs["ladder_bf16"], "absent_noun"), "cf1", "original"),
    "argmax_bf16": (E.wording_rows(runs["replay_bf16"], "baseline"), "a1", "original"),
}
B = {}
for name, (rows, value, images) in cases.items():
    tab = oriented_table(rows, value=value, images=images)
    w = A.wilcoxon_paired(tab["d"].to_numpy())
    B[name] = {"median": float(np.median(tab["d"])), "wilcoxon_p_scene": w["p_value"],
               "signflip_scene": signflip_p(tab, "scene"), "signflip_frame": signflip_p(tab, "frame"),
               "median_cluster_boot_ci": cluster_boot_median(tab)}
out["B_signed_cluster"] = B

# splits: are opposite scenes one per frame? (if so, the binomial is frame-level)
out["B_splits_note"] = {"max_opposite_scenes_per_frame": int(opp_per_frame.max())}

# C. argmax signed effect with zeros kept (Pratt) ------------------------------- #
tab = oriented_table(E.wording_rows(runs["replay_bf16"], "baseline"), value="a1")
d = tab["d"].to_numpy()
out["C_argmax_zeros"] = {
    "n": int(len(d)), "share_zero": float(np.mean(d == 0)),
    "wilcox_drop_zeros_p": float(stats.wilcoxon(d[d != 0]).pvalue),
    "pratt_p": float(stats.wilcoxon(d, zero_method="pratt").pvalue),
    "zsplit_p": float(stats.wilcoxon(d, zero_method="zsplit").pvalue),
    "sign_test_p": float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue),
    "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
}
tab4 = oriented_table(E.logged_rows(log), value="a1")
d4 = tab4["d"].to_numpy()
out["C_argmax_zeros"]["logged4bit"] = {
    "share_zero": float(np.mean(d4 == 0)),
    "wilcox_drop_zeros_p": float(stats.wilcoxon(d4[d4 != 0]).pvalue),
    "pratt_p": float(stats.wilcoxon(d4, zero_method="pratt").pvalue)}

# D. Holm for the contrast within run ------------------------------------------ #
grid = json.load(open(os.path.join(REPO, "docs/btp_paper/evidence/evidence.json")))["wording_grid"]["cf1_convention"]
D = {}
for run in ("4-bit", "4-bit mirror", "bf16", "bf16 mirror"):
    ps6 = {w: grid[f"{run}|{w}"]["contrast_p"] for w in E.WORDINGS}
    ps5 = {w: p for w, p in ps6.items() if w != "baseline"}
    sp6 = {w: grid[f"{run}|{w}"]["splits_p"] for w in E.WORDINGS}
    sg6 = {w: grid[f"{run}|{w}"]["signed_p"] for w in E.WORDINGS}
    D[run] = {"contrast_raw": ps6,
              "contrast_holm_six": E.holm_adjusted(ps6), "contrast_holm_five_alt": E.holm_adjusted(ps5),
              "splits_holm_six": E.holm_adjusted(sp6), "signed_holm_six": E.holm_adjusted(sg6)}
all24 = {f"{r}|{w}": grid[f"{r}|{w}"]["signed_p"] for r in ("4-bit", "4-bit mirror", "bf16", "bf16 mirror")
         for w in E.WORDINGS}
D["signed_holm_all24"] = E.holm_adjusted(all24)
c24 = {f"{r}|{w}": grid[f"{r}|{w}"]["contrast_p"] for r in ("4-bit", "4-bit mirror", "bf16", "bf16 mirror")
       for w in E.WORDINGS}
D["contrast_holm_all24"] = E.holm_adjusted(c24)
out["D_holm"] = D

# E. predictions between the /255 step and the /254 width ---------------------- #
rb = runs["replay_bf16"]
mask = (rb["cf1"].abs() >= E.STEP_255[1]) & (rb["cf1"].abs() < E.ONE_BIN)
out["E_between_step_and_width"] = rb.loc[mask, ["scene_id", "condition", "role", "image_transform", "cf1"]] \
    .to_dict("records")
for tag in ("replay_nf4", "ladder", "ladder_bf16"):
    r = runs[tag]
    m = (r["cf1"].abs() >= E.STEP_255[1]) & (r["cf1"].abs() < E.ONE_BIN)
    out[f"E_between_{tag}"] = r.loc[m, ["scene_id", "condition", "role", "image_transform"]].to_dict("records")

# F. argmax zero-motion bin (128) under the 'any nonzero' threshold ------------ #
bl = E.wording_rows(runs["replay_bf16"], "baseline")
b1 = bl["b1"].astype(int)
out["F_argmax_bin128"] = {
    "bf16_baseline_share_bin128": float((b1 == 128).mean()),
    "bf16_baseline_share_bin127": float((b1 == 127).mean()),
    "bin128_dy": float(E.NORM_ZERO_DY + 1 * E.STEP_255[1]),
}
e_thr0 = E.evaluate(bl, geo, value="a1", floor=0.0, n_boot=0)
bl_z = bl.copy()
bl_z.loc[bl_z["b1"].astype(int) == 128, "a1"] = 0.0  # treat the zero-motion token as zero
e_thr0_z = E.evaluate(bl_z, geo, value="a1", floor=0.0, n_boot=0)
pick = lambda e: {"opp_n": e["opp_n"], "opp_same_sign": e["opp_same_sign"], "opp_both_correct": e["opp_both_correct"],
                  "right": [e["by_layout"][k]["right_share"] for k in R.CONFIGS],
                  "contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                  "splits": f'{e["splits_right_way"]}/{e["splits"]}', "splits_p": e["splits_p"]}
out["F_argmax_thr0"] = {"as_designed": pick(e_thr0), "bin128_as_zero": pick(e_thr0_z)}

# G. placebo: contrast on the grab/take paraphrase pair -------------------------- #
G = {}
for tag in ("ladder_bf16", "ladder"):
    for tr in ("original", "mirror"):
        rows = R.wording_log(runs[tag], "paraphrase", tr)
        instr = rows.groupby("role")["instruction"].first().to_dict()
        e = E.evaluate(rows, geo, value="cf1", images=tr, n_boot=0)
        G[f"{tag}|{tr}"] = {"instructions_example": instr,
                            "contrast_pts": e["contrast_pts"], "contrast_p": e["contrast_p"],
                            "pairs": e["contrast_pairs"], "discordant": e["contrast_discordant"],
                            "opp_same_sign": e["opp_same_sign"],
                            "ss_same_sign": [e["by_layout"]["same_side_left"]["same_sign"],
                                             e["by_layout"]["same_side_right"]["same_sign"]],
                            "signed_median_grab_minus_take": e["signed_median_bins"], "signed_p": e["signed_p"]}
out["G_placebo_paraphrase"] = G

# H. 24 cells: significant positive contrast with backwards splits (cf1) ------- #
H = []
for key, v in grid.items():
    if v["contrast_p"] is not None and v["contrast_p"] < .05 and v["contrast_pts"] > 0 \
            and v["splits"] and v["splits_right_way"] / v["splits"] < .5:
        H.append({"cell": key, "contrast_pts": v["contrast_pts"], "contrast_p": v["contrast_p"],
                  "splits": f'{v["splits_right_way"]}/{v["splits"]}', "splits_p": v["splits_p"]})
out["H_backwards_but_significant"] = H

# I. Table VII gaps ------------------------------------------------------------- #
I = {}
for key, v in grid.items():
    I[key] = {"splits": f'{v["splits_right_way"]}/{v["splits"]}', "splits_share": v["splits_share"],
              "splits_p": v["splits_p"], "signed": v["signed_median_bins"], "signed_p": v["signed_p"],
              "both_correct": v["opp_both_correct"], "contrast": v["contrast_pts"], "contrast_p": v["contrast_p"]}
out["I_wording_grid_cf1"] = I

# K. label-flip invariance of the sign-agreement contrast ----------------------- #
ref_rows = E.wording_rows(runs["replay_bf16"], "baseline")
base = E.evaluate(ref_rows, geo, value="cf1", n_boot=0)
flipped = ref_rows.copy()
pick_scenes = set(RNG.choice(flipped["scene_id"].unique(), size=170, replace=False))
sel = flipped["scene_id"].isin(pick_scenes)
flipped.loc[sel, "role"] = flipped.loc[sel, "role"].map({"a": "b", "b": "a"})
fl = E.evaluate(flipped, geo, value="cf1", n_boot=0)
out["K_label_flip_invariance"] = {
    "observed": {"contrast_pts": base["contrast_pts"], "contrast_p": base["contrast_p"],
                 "splits": f'{base["splits_right_way"]}/{base["splits"]}', "signed": base["signed_median_bins"]},
    "half_the_scenes_flipped": {"contrast_pts": fl["contrast_pts"], "contrast_p": fl["contrast_p"],
                                "splits": f'{fl["splits_right_way"]}/{fl["splits"]}', "signed": fl["signed_median_bins"]},
}

ev = json.load(open(os.path.join(REPO, "docs/btp_paper/evidence/evidence.json")))
out["J_misc"] = {
    "paraphrase_floor": ev["claim10_example"]["paraphrase_floor"],
    "positive_control_bf16": ev["claim1_action_frame"]["current_definition"]["dy_cf1_bf16"]["paste_displacement"],
    "dx_bf16_manipulation": ev["claim1_action_frame"]["current_definition"]["dx_cf0_bf16_own_bin"]["manipulation_check"],
}

json.dump(clean(out), open(os.path.join(HERE, "r2_rigour_checks.json"), "w"), indent=1)
print(json.dumps(clean({k: out[k] for k in ("A_clusters", "B_signed_cluster", "B_splits_note", "C_argmax_zeros",
                                            "E_between_step_and_width", "F_argmax_bin128", "F_argmax_thr0",
                                            "G_placebo_paraphrase", "H_backwards_but_significant",
                                            "K_label_flip_invariance")}), indent=1))


# --------------------------------------------------------------------------- #
# L. frame-level sign-flip p for the signed effect in all 24 wording x run cells,
#    and Holm within run / over 24 with those p-values
# --------------------------------------------------------------------------- #
RUNMAP = {"4-bit": ("replay_nf4", "ladder", "original"), "4-bit mirror": ("replay_nf4", "ladder", "mirror"),
          "bf16": ("replay_bf16", "ladder_bf16", "original"), "bf16 mirror": ("replay_bf16", "ladder_bf16", "mirror")}
L = {}
for run, (rtag, ltag, tr) in RUNMAP.items():
    for w in E.WORDINGS:
        rows = E.wording_rows(runs[rtag] if w == "baseline" else runs[ltag], w, tr)
        tab = oriented_table(rows, value="cf1", images=tr)
        L[f"{run}|{w}"] = {"median": float(np.median(tab["d"])),
                           "wilcoxon_p": A.wilcoxon_paired(tab["d"].to_numpy())["p_value"],
                           "frame_flip_p": signflip_p(tab, "frame")["p"],
                           "median_cluster_boot_ci": cluster_boot_median(tab, n_boot=2000)}
out["L_cells_frame_level"] = L
LH = {}
for run in RUNMAP:
    LH[run] = E.holm_adjusted({w: L[f"{run}|{w}"]["frame_flip_p"] for w in E.WORDINGS})
LH["all24"] = E.holm_adjusted({k: v["frame_flip_p"] for k, v in L.items()})
out["L_holm_frame_level"] = LH


# M. McNemar contrast: which duplicate scene per frame is kept --------------- #
def contrast_with_choice(rows, seed, floor=E.ONE_BIN):
    """Re-run the paired contrast keeping a random scene per frame where a frame has two of a kind."""
    rng = np.random.default_rng(seed)
    work = rows.copy()
    work["c1"] = work["cf1"].astype(float)
    pivot = A._pivot_condition(work, "baseline", True)
    meta = work[["scene_id", "configuration", "base_scene_id"]].drop_duplicates(subset="scene_id")
    merged = pivot.merge(meta, on="scene_id", how="left")
    merged = merged.assign(same_sign=(merged["a"] * merged["b"] > 0),
                           resolved=(A._decided(merged["a"].to_numpy(), floor) & A._decided(merged["b"].to_numpy(), floor)))
    ss = merged[merged["configuration"].str.startswith("same_side") & merged["resolved"]]
    op = merged[(merged["configuration"] == "opposite") & merged["resolved"]]
    ss = ss.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    op = op.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    return A._paired_contrast(ss, op)


ref_rows = E.wording_rows(runs["replay_bf16"], "baseline")
res = [contrast_with_choice(ref_rows, s) for s in range(400)]
ps = np.array([r["p_value"] for r in res])
diffs = np.array([100 * r["difference"] for r in res])
out["M_mcnemar_duplicate_choice"] = {
    "n_draws": len(res), "p_min": float(ps.min()), "p_median": float(np.median(ps)), "p_max": float(ps.max()),
    "share_p_below_05": float(np.mean(ps < .05)),
    "diff_min": float(diffs.min()), "diff_max": float(diffs.max()),
    "as_reported": {"pts": base["contrast_pts"], "p": base["contrast_p"]}}

json.dump(clean(out), open(os.path.join(HERE, "r2_rigour_checks.json"), "w"), indent=1)
print(json.dumps(clean({"L_holm_frame_level": LH, "M_mcnemar_duplicate_choice": out["M_mcnemar_duplicate_choice"]}),
                 indent=1))
for k, v in L.items():
    print(f"{k:28s} med {v['median']:+.3f}  wilcoxon {v['wilcoxon_p']:.4g}  frame-flip {v['frame_flip_p']:.4g}  "
          f"CI [{v['median_cluster_boot_ci'][0]:+.3f}, {v['median_cluster_boot_ci'][1]:+.3f}]")


# N. paraphrase floor with the folded readout (evidence.json uses c1) --------- #
def paraphrase_floor_col(replay, ladder, col):
    base = R.paired(R.wording_log(replay, R.BASELINE), R.BASELINE, col)
    para = R.paired(ladder[(ladder["condition"] == "ladder_paraphrase") & (ladder["image_transform"] == "original")],
                    "ladder_paraphrase", col)
    j = base.join(para, lsuffix="_lr", rsuffix="_pp").dropna()
    lr = (j["a_lr"] - j["b_lr"]).abs() / E.ONE_BIN
    pp = (j["a_pp"] - j["b_pp"]).abs() / E.ONE_BIN
    return {"n": int(len(j)), "lr_median": float(lr.median()), "pp_median": float(pp.median()),
            "share_lr_larger": float(np.mean(lr > pp)), "wilcoxon_p": float(stats.wilcoxon(lr, pp).pvalue)}


out["N_paraphrase_floor"] = {f"{prec}|{col}": paraphrase_floor_col(runs[r], runs[l], col)
                             for prec, (r, l) in E.RUN_PAIRS.items() for col in ("c1", "cf1")}
json.dump(clean(out), open(os.path.join(HERE, "r2_rigour_checks.json"), "w"), indent=1)
print(json.dumps(clean(out["N_paraphrase_floor"]), indent=1))
