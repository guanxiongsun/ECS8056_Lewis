#!/usr/bin/env python3
"""BtP plan B1: the specification curve, as pre-declared in docs/btp_paper/analysis_plan.md (B0).

    .venv/bin/python docs/btp_paper/evidence/btp_spec_curve.py [--flips 10000]

Stage 1 covers the 16 GH200 facets (wording x readout x precision) with outcomes O1-O3:
  O1  signed word effect toward the named twin's side (median of the oriented left-right
      difference, grid steps), with a two-sided base-frame sign-flip test of the Wilcoxon statistic
  O2  share of decided opposite-scene splits going to the named twins (binomial)
  O3  opposite-scene both-correct, descriptive, with a base-frame bootstrap upper limit
Joint test per facet: Stouffer's Z over the facet's O1 specifications, signs kept, two-sided,
against datasets whose left/right labels are flipped jointly within each base frame; Holm across
facets. O4 (placebo-corrected contrast), the A100 facets and the prefix facets come with the
B4.1/B2 outputs (stage 2).

Writes outputs/btp/spec_curve.{csv,json} and docs/btp_paper/figures/fig_spec_curve.{pdf,png}.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import btp_paper.evidence.btp_evidence as E  # noqa: E402

R = E.R
REPO = E.REPO
STEP = float(E.STEP_255[1])     # one grid step of dy, (q99 - q01) / 255
WORDINGS = ("baseline", "prenominal", "object", "table_side")
LABEL = {"baseline": "original (… on the left)", "prenominal": "the left {noun}",
         "object": "the object on the left", "table_side": "… left side of the table"}
PRECISION = {"bf16": ("replay_bf16", "ladder_bf16"), "nf4": ("replay_nf4", "ladder")}
READOUTS = ("expected", "argmax")
THRESHOLDS = (0, 1, 2)          # grid steps; 0 = any nonzero
SUBSETS = ("all", "detected", "not_relabelled", "both")
SEED = 20261003
Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #
def subsets(geo: pd.DataFrame) -> dict:
    det = set(geo.index[geo["detected_gripper"]])
    notrel = set(geo.index[~geo["relabelled"]])
    return {"all": None, "detected": sorted(det), "not_relabelled": sorted(notrel), "both": sorted(det & notrel)}


def readout_setup(rows: pd.DataFrame, readout: str):
    """(rows with the readout column, column name, unit, floor per threshold)."""
    if readout == "expected":
        return rows, "cf1", STEP, {k: k * STEP for k in THRESHOLDS}
    # Argmax: decisions count grid steps from bin 128, the bin holding physical zero (R-P5).
    rows = rows.assign(z1=rows["b1"].astype(float) - 128.0)
    return rows, "z1", 1.0, {0: 0.5, 1: 1.0, 2: 2.0}


def facet_rows(data: dict, wording: str, precision: str) -> pd.DataFrame:
    replay, ladder = PRECISION[precision]
    run = data["runs"][replay] if wording == "baseline" else data["runs"][ladder]
    return E.wording_rows(run, wording, "original")


def oriented(rows: pd.DataFrame, geo: pd.DataFrame, col: str, unit: float, scenes) -> pd.DataFrame:
    work = rows.assign(c1=rows[col].astype(float))
    g = geo if scenes is None else geo[geo.index.isin(set(scenes))]
    if scenes is not None:
        work = work[work["scene_id"].isin(set(scenes))]
    out = R.scene_outcomes(work, g)
    return pd.DataFrame({"d": out["oriented"].to_numpy() / unit, "frame": out["base_scene_id"].to_numpy()})


# --------------------------------------------------------------------------- #
# Frame-level inference
# --------------------------------------------------------------------------- #
def per_frame_signed_ranks(d: np.ndarray, frames: np.ndarray, frame_index: dict) -> np.ndarray:
    """Wilcoxon signed ranks (zeros dropped) summed per base frame, on the global frame index."""
    keep = d != 0
    s = np.sign(d[keep]) * stats.rankdata(np.abs(d[keep]))
    out = np.zeros(len(frame_index))
    np.add.at(out, [frame_index[f] for f in frames[keep]], s)
    return out


def tail_p(null: np.ndarray, obs: float) -> float:
    """Share of null values at least as large as obs, ties counted half, floored at 1/n."""
    n = len(null)
    p = (np.sum(null > obs) + 0.5 * np.sum(np.isclose(null, obs))) / n
    return float(max(p, 1.0 / n))


def joint_test(P: np.ndarray, flips: np.ndarray) -> dict:
    """P: frames x specs per-frame signed-rank sums. Two-sided Stouffer test with frame-level flips."""
    sd = np.sqrt((P ** 2).sum(axis=0))
    z_obs = P.sum(axis=0) / sd
    z_null = np.einsum("nf,fk->nk", flips, P) / sd   # einsum: macOS BLAS raises spurious FP warnings
    k = P.shape[1]
    s_obs = float(z_obs.sum() / math.sqrt(k))
    s_null = z_null.sum(axis=1) / math.sqrt(k)
    toward, away = float(np.mean(z_obs > Z95)), float(np.mean(z_obs < -Z95))
    return {"stouffer_z": s_obs, "p_two_sided": tail_p(np.abs(s_null), abs(s_obs)),
            "share_sig_toward": toward, "p_share_toward": tail_p(np.mean(z_null > Z95, axis=1), toward),
            "share_sig_away": away, "p_share_away": tail_p(np.mean(z_null < -Z95, axis=1), away),
            "z_per_spec": [float(z) for z in z_obs]}


def frame_boot_median_ci(d: np.ndarray, frames: np.ndarray, rng, n_boot=2000):
    codes, uniq = pd.factorize(frames)
    groups = [d[codes == i] for i in range(len(uniq))]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Variance decomposition
# --------------------------------------------------------------------------- #
def shapley_r2(df: pd.DataFrame, y: str, factors: list) -> dict:
    def r2(fs):
        if not fs:
            return 0.0
        X = pd.get_dummies(df[list(fs)].astype(str), drop_first=True).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(df)), X])
        beta, *_ = np.linalg.lstsq(X, df[y].to_numpy(dtype=float), rcond=None)
        resid = df[y].to_numpy(dtype=float) - X @ beta
        tss = np.sum((df[y] - df[y].mean()) ** 2)
        return float(1 - np.sum(resid ** 2) / tss) if tss > 0 else 0.0
    n = len(factors)
    cache = {fs: r2(fs) for r in range(n + 1) for fs in itertools.combinations(factors, r)}
    out = {}
    for f in factors:
        rest = [g for g in factors if g != f]
        val = 0.0
        for r in range(len(rest) + 1):
            for fs in itertools.combinations(rest, r):
                with_f = tuple(sorted(fs + (f,), key=factors.index))
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                val += w * (cache[with_f] - cache[fs])
        out[f] = val
    out["total"] = cache[tuple(factors)]
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "google_drive", "v2"))
    ap.add_argument("--runs", default=os.path.join(REPO, "outputs", "runs"))
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "btp"))
    ap.add_argument("--flips", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figure-only", action="store_true", help="redraw from outputs/btp/spec_curve.{csv,json}")
    args = ap.parse_args()
    if args.figure_only:
        specs = pd.read_csv(os.path.join(args.out, "spec_curve.csv"))
        with open(os.path.join(args.out, "spec_curve.json")) as f:
            facets = json.load(f)["facets"]
        figure(specs, facets, os.path.join(REPO, "docs", "btp_paper", "figures", "fig_spec_curve"))
        return
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = E.load_all(args.data, args.runs)
    geo = data["geo"]
    subs = subsets(geo)
    frames_all = sorted(geo["base_scene_id"].unique())
    frame_index = {f: i for i, f in enumerate(frames_all)}
    flips = rng.choice(np.array([-1.0, 1.0]), size=(args.flips, len(frames_all)))

    spec_rows, facets = [], {}
    for wording, readout, precision in itertools.product(WORDINGS, READOUTS, PRECISION):
        facet = f"{wording}|{readout}|{precision}"
        base_rows = facet_rows(data, wording, precision)
        rows, col, unit, floors = readout_setup(base_rows, readout)
        P_cols = []
        for sub in SUBSETS:
            o = oriented(rows, geo, col, unit, subs[sub])
            d, fr = o["d"].to_numpy(), o["frame"].to_numpy()
            P = per_frame_signed_ranks(d, fr, frame_index)
            P_cols.append(P)
            sd = math.sqrt(float((P ** 2).sum()))
            z = float(P.sum() / sd) if sd > 0 else float("nan")
            t_null = np.abs(np.einsum("nf,f->n", flips, P))
            lo, hi = frame_boot_median_ci(d, fr, rng, n_boot=args.n_boot)
            o1 = {"o1_median": float(np.median(d)), "o1_mean": float(np.mean(d)), "o1_ci_lo": lo, "o1_ci_hi": hi,
                  "o1_z": z, "o1_p_frame": tail_p(t_null, abs(float(P.sum()))), "o1_ties": float(np.mean(d == 0)),
                  "n_pairs": int(len(d))}
            for k in THRESHOLDS:
                e = E.evaluate(rows, geo, value=col, floor=floors[k], unit=unit, scenes=subs[sub], n_boot=args.n_boot)
                ci = e["by_layout"]["opposite"].get("both_correct_ci", {})
                lo3, hi3 = ci.get("lo", float("nan")), ci.get("hi", float("nan"))
                spec_rows.append({"facet": facet, "wording": wording, "readout": readout, "precision": precision,
                                  "subset": sub, "threshold": k, **o1,
                                  "o2_k": e["splits_right_way"], "o2_n": e["splits"], "o2_share": e["splits_share"],
                                  "o2_p": e["splits_p"], "o3": e["opp_both_correct"], "o3_ci_lo": lo3, "o3_ci_hi": hi3,
                                  "opp_n": e["opp_n"]})
        facets[facet] = joint_test(np.column_stack(P_cols), flips)
        facets[facet].update({"wording": wording, "readout": readout, "precision": precision})
        print(f"{facet:28s} Z={facets[facet]['stouffer_z']:+.2f}  p={facets[facet]['p_two_sided']:.4f}", flush=True)

    specs = pd.DataFrame(spec_rows)
    holm = E.holm_adjusted({f: v["p_two_sided"] for f, v in facets.items()})
    for f in facets:
        facets[f]["p_holm"] = float(holm[f])
        sub = specs[(specs["facet"] == f) & (specs["threshold"] == 1) & (specs["subset"] == "all")]
        facets[f]["o1_median_all"] = float(sub["o1_median"].iloc[0])
        # Pre-declared rule: the sign of the facet's median O1. Argmax facets can have a median of
        # exactly 0 (tied bins); their Stouffer sign is reported beside it (post hoc note, B0 file).
        facets[f]["direction"] = "toward" if facets[f]["o1_median_all"] > 0 else (
            "away" if facets[f]["o1_median_all"] < 0 else "none (median 0)")
        facets[f]["stouffer_sign"] = "toward" if facets[f]["stouffer_z"] > 0 else "away"

    # Reference specification must reproduce Table I's `ref` row (bf16, expected value, original wording,
    # all scenes, one grid step): O2 7/21, O3 7/112, O1 median -0.15 grid steps.
    ref = specs[(specs["facet"] == "baseline|expected|bf16") & (specs["subset"] == "all") & (specs["threshold"] == 1)].iloc[0]
    assert (ref["o2_k"], ref["o2_n"]) == (7, 21) and ref["opp_n"] == 112 and round(ref["o1_median"], 2) == -0.15, ref

    # "No reliable selection": the O3 upper limit is below 50% in every specification.
    o3_max_upper = float(specs["o3_ci_hi"].max())
    o1 = specs[specs["threshold"] == 1].drop_duplicates(["facet", "subset"])
    shap = shapley_r2(o1, "o1_median", ["wording", "readout", "precision", "subset"])
    summary = {
        "design": "docs/btp_paper/analysis_plan.md (B0), stage 1: 16 GH200 facets, O1-O3",
        "n_specs": {"o1": int(len(o1)), "o2_o3": int(len(specs))}, "flips": args.flips, "seed": SEED,
        "unit": f"grid steps of {STEP:.7f}", "facets": facets,
        "o3_max_upper_limit": o3_max_upper, "no_reliable_selection": bool(o3_max_upper < 0.5),
        "shapley_r2_o1": shap,
        "facets_significant_holm": {f: v["direction"] for f, v in facets.items() if v["p_holm"] < .05},
    }
    specs.to_csv(os.path.join(args.out, "spec_curve.csv"), index=False)
    with open(os.path.join(args.out, "spec_curve.json"), "w") as f:
        json.dump(E.jclean(summary), f, indent=2)
    figure(specs, facets, os.path.join(REPO, "docs", "btp_paper", "figures", "fig_spec_curve"))
    print(json.dumps({k: summary[k] for k in ("n_specs", "o3_max_upper_limit", "no_reliable_selection",
                                              "shapley_r2_o1", "facets_significant_holm")}, indent=1))


# --------------------------------------------------------------------------- #
# Figure: facet summary (top) and the bf16 expected-value curves per wording (bottom)
# --------------------------------------------------------------------------- #
def figure(specs: pd.DataFrame, facets: dict, stem: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    colors = {"baseline": "#7f7f7f", "prenominal": "#1b9e77", "object": "#66a61e", "table_side": "#d95f02"}
    fig = plt.figure(figsize=(7.0, 3.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.62, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    order = [f"{w}|{r}|{p}" for w in WORDINGS for r in READOUTS for p in PRECISION]
    for i, f in enumerate(order):
        v = facets[f]
        c = colors[v["wording"]]
        ax.plot([i, i], [0, v["stouffer_z"]], color=c, lw=1.2)
        ax.scatter([i], [v["stouffer_z"]], color=c if v["p_holm"] < .05 else "white", edgecolor=c, s=18, zorder=3,
                   marker="o" if v["readout"] == "expected" else "s")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{f.split('|')[1][:3]}\n{f.split('|')[2]}" for f in order], fontsize=5.5)
    for j, w in enumerate(WORDINGS):
        ax.text(4 * j + 1.5, ax.get_ylim()[1], LABEL[w], ha="center", va="bottom", fontsize=6.5, color=colors[w])
    ax.set_ylabel("facet joint test\n(Stouffer Z, + toward)")
    ax.set_xlim(-0.6, len(order) - 0.4)

    for j, w in enumerate(WORDINGS):
        axw = fig.add_subplot(gs[1, j])
        sub = specs[(specs["facet"] == f"{w}|expected|bf16")].sort_values("o2_share").reset_index(drop=True)
        for i, r in sub.iterrows():
            lo, hi = E.wilson(int(r["o2_k"]), int(r["o2_n"]))
            axw.plot([i, i], [lo, hi], color=colors[w], lw=0.8)
            axw.scatter([i], [r["o2_share"]], color=colors[w], s=8, zorder=3)
        axw.axhline(0.5, color="black", lw=0.5, ls=":")
        axw.set_ylim(0, 1)
        axw.set_xticks([])
        axw.set_title(LABEL[w], fontsize=6.5, color=colors[w])
        if j == 0:
            axw.set_ylabel("O2, splits toward\nnamed twin (bf16, EV)")
        # dashboard: threshold and subset behind each point
        for i, r in sub.iterrows():
            axw.text(i, -0.08, str(int(r["threshold"])), ha="center", va="top", fontsize=4.5,
                     transform=axw.get_xaxis_transform())
            axw.text(i, -0.17, {"all": "a", "detected": "d", "not_relabelled": "n", "both": "b"}[r["subset"]],
                     ha="center", va="top", fontsize=4.5, transform=axw.get_xaxis_transform())
    fig.savefig(stem + ".pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})


if __name__ == "__main__":
    main()
