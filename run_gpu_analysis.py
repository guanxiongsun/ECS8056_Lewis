#!/usr/bin/env python3
"""
run_gpu_analysis.py: summarise the GPU runs from scripts/run_gpu.py.

    python run_gpu_analysis.py --runs outputs/runs --data google_drive/v2

Reads whichever runs are present (replay_{nf4,bf16}, ladder, ladder_bf16,
natural, natural_bf16), writes `report.md` and `results.json` to --runs, and
prints the report. Pull the run logs first, e.g.

    rsync -a b5cs.aip2.isambard:/projects/b5cs/SGVLA/runs/ outputs/runs/
"""

from __future__ import annotations

import argparse
import json
import os

import reanalysis as R
from run_reanalysis import _clean, ci_str, pct

# Wordings scored against the named twins; `paraphrase` is a noise floor and is
# reported separately.
WORDINGS = ("baseline", "prenominal", "object", "table_side", "absent_noun", "move")
# Run files per precision: the replay of the original log, the ladder, the natural frames.
RUNS = {"nf4": ("replay_nf4", "ladder", "natural"), "bf16": ("replay_bf16", "ladder_bf16", "natural_bf16")}


def _load(runs_dir: str, tag: str):
    try:
        return R.load_run(os.path.join(runs_dir, f"{tag}.csv"))
    except FileNotFoundError:
        return None


def _p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def wording_table(replay, ladder, geo, transform: str):
    """Per-wording direction tests for one precision and image transform."""
    rows = {}
    for wording in WORDINGS:
        if wording == "baseline":
            if replay is None:
                continue
            condition = R.BASELINE if transform == "original" else R.MIRROR
            log = replay[(replay["condition"] == condition) & (replay["image_transform"] == transform)].copy()
            log["condition"] = R.BASELINE
        elif ladder is not None:
            log = R.wording_log(ladder, wording, transform)
        else:
            continue
        if not log.empty:
            rows[wording] = R.wording_tests(log, geo, mirrored=(transform == "mirror"))
    family = {w: e["contrast_paired"]["p_value"] for w, e in rows.items() if w != "baseline"}
    return rows, R.holm(family)


def wording_lines(rows: dict, holm: dict, title: str) -> list:
    L = [f"\n## Direction of the word effect per wording ({title})\n",
         "Paired contrast = same-side minus opposite same-direction rate (McNemar). Splits = opposite scenes "
         "where the two instructions move apart; right way = toward the named twins (binomial vs 50%). "
         "Signed median: + = 'left' more image-left.\n",
         "| wording | paired contrast, pts (p) | opposite splits right way (p) | both correct, opposite | "
         "signed median, bins (p) | image-right SSL / opp / SSR |",
         "|---|---|---|---|---|---|"]
    for wording, e in rows.items():
        c = e["contrast_paired"]
        shares = " / ".join(f"{100 * e['by_configuration'][k]['right_share']:.0f}" if k in e["by_configuration"]
                            else "–" for k in R.CONFIGS)
        L.append(f"| {wording} | {100 * c['difference']:+.1f} ({_p(c['p_value'])}) | "
                 f"{e['splits_right_way']}/{e['opposite_splits']} = {pct(e['splits_right_way_share'])} "
                 f"({_p(e['splits_binomial_p'])}) | {pct(e['opposite_both_correct'])} | "
                 f"{e['oriented_median_bins']:+.2f} ({_p(e['wilcoxon_p'])}) | {shares} |")
    if holm:
        L.append("\nHolm over the alternative wordings (paired contrast): "
                 + ", ".join(f"{w} {'reject' if h['reject'] else 'keep'}" for w, h in holm.items()))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--data", default="google_drive/v2")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    original = R.load_probe(args.data)
    constructed = R.load_constructed(args.data)
    geo = R.scene_geometry(original, constructed)
    bridge = R.load_bridge_manifest(args.data)
    res, L = {}, ["# GPU runs\n"]

    for tag in ("replay_nf4", "replay_bf16"):
        run = _load(args.runs, tag)
        if run is None:
            continue
        cmp = R.compare_replay(original, run)
        res[tag] = {"compare": cmp, "readout_fix": R.readout_fix_effect(run),
                    "gpu": sorted(run["gpu_name"].unique()), "precision": sorted(run["precision"].unique())}
        L.append(f"## {tag} ({cmp['n_matched']} of {len(original)} logged predictions replayed on "
                 f"{', '.join(res[tag]['gpu'])})\n")
        L.append(f"- argmax identical on all 6 movement dims: {pct(cmp['argmax_equal_translation_rotation'])}; "
                 f"lateral a1 identical {pct(cmp['argmax_equal_a1'])} (one-step changes "
                 f"{pct(cmp['a1_step_changes']['1'])}, ≥2 steps {pct(cmp['a1_step_changes']['>=2'])})")
        L.append(f"- continuous c1: |Δ| median {cmp['c1_abs_diff_bins']['median']:.2f} bins, p95 "
                 f"{cmp['c1_abs_diff_bins']['p95']:.2f}, max {cmp['c1_abs_diff_bins']['max']:.1f}; Pearson "
                 f"{cmp['c1_pearson']:.4f}; sign agreement where both ≥1 bin {pct(cmp['c1_sign_agreement_decided'])}")
        if len(run) >= len(original) * 0.99:
            h = R.headline_with_ci(run, geo, n_boot=args.n_boot)
            res[tag]["headline"] = h
            L.append("\n| arrangement | n | same direction | both correct | image-right share |\n|---|---|---|---|---|")
            for name, e in h.items():
                L.append(f"| {name} | {e['n']} | {ci_str(e['same_sign'])} | {ci_str(e['both_correct'])} | "
                         f"{ci_str(e['right_share'])} |")
        fix = res[tag]["readout_fix"]
        if "dim1" in fix:
            L.append(f"\n- corrected readout (token 31744 included): lateral max change {fix['dim1']['max_abs_change']:.2e}, "
                     f"lateral min mass {fix['dim1']['min_mass']:.4f}; gripper min mass {fix['dim6']['min_mass']:.4f}\n")

    ladder = _load(args.runs, "ladder")
    if ladder is not None:
        lad = R.ladder_summary(ladder, original, geo)
        res["ladder"] = lad
        L.append(f"## Instruction ladder, 4-bit (baseline left↔right median |Δ| = "
                 f"{lad['baseline_left_right_median_abs_bins']:.2f} bins)\n")
        for name, e in lad.items():
            if not isinstance(e, dict):
                continue
            line = f"- **{name}** n={e['n']}: median |Δ| {e['median_abs_diff_bins']:.2f} bins"
            if name == "move":
                line += (f"; 'move left' goes image-left {pct(e['move_left_goes_image_left'])}, 'move right' goes "
                         f"image-right {pct(e['move_right_goes_image_right'])}; signed median {e['oriented_median_bins']:+.2f}"
                         f" (p={e['wilcoxon_p']:.3g})")
            elif name == "paraphrase":
                v = e["vs_baseline_a_median_abs_bins"]
                line += f" between two wordings of the same referent; vs baseline 'left' {v['a']:.2f} / {v['b']:.2f}"
            else:
                hd = e["headline"]
                line += (f"; signed median {e['oriented_median_bins']:+.2f} (p={e['wilcoxon_p']:.3g}); same direction "
                         + " / ".join(f"{k[:13]} {pct(v['same_sign'])}" for k, v in hd.items())
                         + "; image-right " + " / ".join(f"{pct(v['right_share'])}" for v in hd.values()))
            L.append(line)

    res["phrase_counts"] = pc = R.phrase_counts(bridge, constructed["noun"].dropna().unique())
    L.append(f"\n## Ladder phrasings in Bridge instructions ({pc['episodes']} episodes)\n")
    for name, e in pc.items():
        if isinstance(e, dict):
            L.append(f"- {name}: {e['n']} (referent rule matches {e['referent_rule']}); e.g. {e['examples'][:2]}")

    res["wording"], res["paraphrase_floor"], res["natural_agreement"] = {}, {}, {}
    for precision, (replay_tag, ladder_tag, natural_tag) in RUNS.items():
        replay, lad_run = _load(args.runs, replay_tag), _load(args.runs, ladder_tag)
        for transform in ("original", "mirror"):
            if lad_run is None or not (lad_run["image_transform"] == transform).any():
                continue
            rows, holm = wording_table(replay, lad_run, geo, transform)
            res["wording"][f"{precision}_{transform}"] = {"rows": rows, "holm": holm}
            L += wording_lines(rows, holm, f"{precision}, {transform} images")
        if replay is not None and lad_run is not None:
            pf = res["paraphrase_floor"][precision] = R.paraphrase_floor(replay, lad_run)
            L.append(f"\nParaphrase floor ({precision}, n={pf['n']}): left↔right median |Δ| "
                     f"{pf['left_right_median_bins']:.2f} bins vs grab↔take {pf['paraphrase_median_bins']:.2f}; "
                     f"left↔right larger in {pct(pf['share_left_right_larger'])} (Wilcoxon p={_p(pf['wilcoxon_p'])})")

        natural = _load(args.runs, natural_tag)
        if natural is None:
            continue
        nat = R.natural_summary(natural, bridge)
        agree = R.natural_agreement(natural, bridge)
        res.setdefault("natural", {})[precision] = nat
        res["natural_agreement"][precision] = agree
        ag, mc, sh = agree["agreement"], agree["mcnemar"], agree["image_left_share"]
        L.append(f"\n## Unedited validation frames, antonym swap ({precision}, n={nat['n']})\n\n- median |Δ| "
                 f"{nat['median_abs_diff_bins']:.2f} bins; signed median {nat['oriented_median_bins']:+.2f} "
                 f"(p={nat['wilcoxon_p']:.3g}); 'left' more leftward {pct(nat['share_left_more_leftward'])}; same "
                 f"direction {pct(nat['same_sign_decided'])} (n={nat['n_decided']})")
        L.append(f"- agreement with the demonstrated first motion on {agree['n_common']} frames where all versions "
                 f"and the demonstration move: own instruction {pct(ag['a'])}, term removed {pct(ag['n'])} "
                 f"(McNemar p={_p(mc['a_vs_n']['p_value'])}), antonym {pct(ag['b'])} (p={_p(mc['a_vs_b']['p_value'])})")
        for label, e in sh.items():
            L.append(f"- image-left after {label.replace('_', ' ')}s: demonstrations {pct(e['demo'])} (n={e['n_demo']}); "
                     f"model with own {pct(e['a'])}, term removed {pct(e['n'])}, antonym {pct(e['b'])}")
        by_role = res.setdefault("natural_by_role", {})[precision] = R.natural_by_role(natural)
        for kind in ("source", "destination", "other"):
            if kind in by_role:
                e = by_role[kind]
                L.append(f"- {kind} word (n={e['n']}): 'left' version more image-left in "
                         f"{pct(e['share_left_more_leftward'])}, mean {e['mean_bins']:+.2f} bins (Wilcoxon p={_p(e['wilcoxon_p'])})")

    os.makedirs(args.runs, exist_ok=True)
    with open(os.path.join(args.runs, "results.json"), "w") as f:
        json.dump(_clean(res), f, indent=2)
    with open(os.path.join(args.runs, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
