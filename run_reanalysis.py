#!/usr/bin/env python3
"""
run_reanalysis.py: run the Tier-1 reanalysis on the logged probe data.

    python run_reanalysis.py --data google_drive/v2 --out outputs/reanalysis

Writes `report.md` (human-readable) and `results.json` (every number) to --out.
Needs only the CSVs from the Drive cache; no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import reanalysis as R


def _clean(obj):
    """Make numpy scalars and tuples JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def pct(x):
    return "nan" if x != x else f"{100 * x:.1f}%"


def ci_str(entry):
    return f"{pct(entry['estimate'])} [{pct(entry['lo'])}, {pct(entry['hi'])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="google_drive/v2")
    ap.add_argument("--out", default="outputs/reanalysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = R.load_probe(args.data)
    constructed = R.load_constructed(args.data)
    bridge = R.load_bridge_manifest(args.data)
    geo = R.scene_geometry(log, constructed)

    res = {}
    res["inventory"] = {
        "predictions": int(len(log)), "scenes": int(geo.shape[0]),
        "base_frames": int(geo["base_scene_id"].nunique()),
        "by_configuration": geo["configuration"].value_counts().to_dict(),
        "relabelled": int(geo["relabelled"].sum()),
        "detected_gripper": int(geo["detected_gripper"].sum()),
    }

    # Headline rates with base-frame bootstrap intervals.
    res["headline"] = {
        "one_bin": R.headline_with_ci(log, geo, n_boot=args.n_boot),
        "nonzero": R.headline_with_ci(log, geo, floor=0.0, n_boot=args.n_boot),
        "mirror_fixed": R.headline_with_ci(log, geo, condition=R.MIRROR, mirrored=True,
                                           n_boot=args.n_boot),
    }
    # Robustness subsets (point estimates).
    outcomes = R.scene_outcomes(log, geo)
    subsets = {
        "all": outcomes,
        "detected_gripper": outcomes[outcomes["detected_gripper"]],
        "not_relabelled": outcomes[~outcomes["relabelled"]],
        "detected_and_not_relabelled": outcomes[outcomes["detected_gripper"] & ~outcomes["relabelled"]],
    }
    res["subsets"] = {k: R.headline(v) for k, v in subsets.items()}
    res["subset_sizes"] = {k: int(v["resolved"].sum()) for k, v in subsets.items()}

    # R1 corrected checks.
    res["manipulation_check_c1"] = R.manipulation_check(log, geo)
    res["paste_displacement"] = {
        "neutral": R.paste_displacement(log, geo),
        "baseline_left": R.paste_displacement(log, geo, condition=R.BASELINE, role="a"),
        "baseline_right": R.paste_displacement(log, geo, condition=R.BASELINE, role="b"),
    }
    res["resolution"] = R.resolution_corrected(log)
    res["signed_contrast"] = {c: R.signed_contrast(log, c) for c in (R.BASELINE, R.SWAPPED, R.MIRROR)}
    res["screening"] = R.screening_by_configuration(args.data)

    # R2 drivers of the lateral action.
    frame = R.design_matrix(log, geo, bridge)
    res["drivers_all"] = R.drivers(frame)
    res["drivers_detected_gripper"] = R.drivers(frame[frame["detected_gripper"]])
    res["within_scene_word"] = R.within_scene_word_effect(frame)
    alt = frame.assign(layout=frame["layout_img"])
    res["drivers_image_centred_detected"] = R.drivers(alt[alt["detected_gripper"]])

    # R3 priors.
    res["demonstration_prior"] = R.demonstration_prior(bridge)
    res["tracking_prior"] = R.tracking_prior(args.data, bridge)
    res["language_audit"] = R.language_audit(bridge)
    res["role_association"] = R.role_association(bridge)

    # R4 and R5.
    res["outer_inner"] = R.outer_inner(log, geo)
    res["distribution"] = R.distribution_diagnostics(log, geo)

    res = _clean(res)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ------------------------------------------------------------------ report
    L = []
    inv = res["inventory"]
    L.append(f"# Tier-1 reanalysis\n\n{inv['predictions']} predictions, {inv['scenes']} scenes, "
             f"{inv['base_frames']} base frames; by arrangement {inv['by_configuration']}; "
             f"{inv['relabelled']} relabelled; {inv['detected_gripper']} with a detected gripper.\n")
    L.append("## Headline (≥1 bin), 95% base-frame bootstrap CIs\n")
    L.append("| arrangement | n | same direction | both correct | image-right share |\n|---|---|---|---|---|")
    for name, e in res["headline"]["one_bin"].items():
        L.append(f"| {name} | {e['n']} | {ci_str(e['same_sign'])} | {ci_str(e['both_correct'])} | "
                 f"{ci_str(e['right_share'])} |")
    L.append("\n## Mirrored replication, corrected scoring (≥1 bin)\n")
    L.append("| arrangement | n | same direction | both correct | image-right share |\n|---|---|---|---|---|")
    for name, e in res["headline"]["mirror_fixed"].items():
        L.append(f"| {name} | {e['n']} | {ci_str(e['same_sign'])} | {ci_str(e['both_correct'])} | "
                 f"{ci_str(e['right_share'])} |")
    L.append("\n## Robustness subsets (≥1 bin; point estimates)\n")
    L.append("| subset | n | opp same-dir | SSL same-dir | SSR same-dir | opp both-correct | right share SSL/opp/SSR |"
             "\n|---|---|---|---|---|---|---|")
    for key, h in res["subsets"].items():
        g = lambda c, m: pct(h.get(c, {}).get(m, float("nan")))  # noqa: E731
        L.append(f"| {key} | {res['subset_sizes'][key]} | {g('opposite','same_sign')} | "
                 f"{g('same_side_left','same_sign')} | {g('same_side_right','same_sign')} | "
                 f"{g('opposite','both_correct')} | {g('same_side_left','right_share')} / "
                 f"{g('opposite','right_share')} / {g('same_side_right','right_share')} |")
    mc = res["manipulation_check_c1"]
    L.append("\n## Manipulation check on the lateral channel (opposite scenes, ≥1 bin)\n")
    for cond, e in mc.items():
        L.append(f"- {cond}: toward paste {pct(e['toward_paste'])} (n={e['n']}, 95% CI "
                 f"{pct(e['ci'][0])}–{pct(e['ci'][1])}); P(right | paste right) = "
                 f"{pct(e['p_right_given_paste_right'])} (n={e['n_paste_right']}) vs P(right | paste left) = "
                 f"{pct(e['p_right_given_paste_left'])} (n={e['n_paste_left']}); difference "
                 f"{100 * e['difference']:+.1f} pts, Fisher p={e['fisher_p']:.3g}")
    L.append("\n## Within-frame paste displacement (same base frame; only the paste moves)\n")
    for cond, e in res["paste_displacement"].items():
        if not e.get("n_frames"):
            continue
        L.append(f"- {cond}: {e['n_frames']} frames; shift toward the paste median "
                 f"{e['median_shift_bins']:+.2f} bins, mean {e['mean_shift_bins']:+.2f} "
                 f"[{e['ci_mean_shift']['lo']:+.2f}, {e['ci_mean_shift']['hi']:+.2f}], "
                 f"p={e['wilcoxon_p']:.3g}; {pct(e['share_shifted_toward_paste'])} shifted toward it; "
                 f"net side changes toward paste {100 * e['net_side_changes_toward_paste']:+.1f} pts")
    rs = res["resolution"]
    L.append(f"\n## Resolution\n\nargmax grid step (from data) = {rs['argmax_step']:.7f} vs pipeline width "
             f"{rs['pipeline_bin_width']}.\n")
    a = rs["argmax"]
    L.append(f"- argmax: exact ties {pct(a['exact_zero'])}; one step {pct(a['one_step'])}; ≥2 steps "
             f"{pct(a['two_or_more_steps'])}; executable action changed {pct(a['changed_executable_action'])}; "
             f"'below one bin' with the pipeline width {pct(a['below_pipeline_width'])}")
    c = rs["continuous"]
    L.append(f"- continuous: exact zeros {pct(c['exact_zero'])}; below one step {pct(c['below_one_step'])}")
    L.append("\n## Signed left–right contrast (bins; positive = 'left' more leftward)\n")
    for cond, e in res["signed_contrast"].items():
        L.append(f"- {cond}: n={e['n']}; mean |Δ| {e['mean_abs_bins']:.1f}, median |Δ| {e['median_abs_bins']:.1f}; "
                 f"median signed {e['median_oriented_bins']:+.2f} (p={e['wilcoxon_p']:.3g}); "
                 f"'left' more leftward in {pct(e['share_left_more_leftward'])}; opposite-sign "
                 f"{pct(e['opposite_sign_one_bin'])} (≥1 bin, n={e['n_one_bin']})")
    sc = res["screening"]
    L.append("\n## Blind screening by arrangement\n")
    for name in R.CONFIGS:
        e = sc[name]
        L.append(f"- {name}: {e['approved']}/{e['screened']} approved ({pct(e['rate'])}); rejections {e['reasons']}")
    L.append(f"- opposite vs same-side: {pct(sc['opposite']['rate'])} vs "
             f"{pct(sc['opposite_vs_same_side']['same_side_rate'])}, χ² p={sc['opposite_vs_same_side']['p']:.2g}")
    for key in ("drivers_all", "drivers_detected_gripper", "drivers_image_centred_detected"):
        d = res[key]
        L.append(f"\n## Drivers of the image-rightward action ({key}; n={d['n']}; bins)\n")
        L.append("| term | OLS coef (cluster SE) | p | mixed coef | Shapley R² |\n|---|---|---|---|---|")
        for term in ("Intercept", "layout", "word", "grounded", "source", "demo", "mirror"):
            o = d["ols"][term]
            m = d.get("mixed", {}).get(term, {}).get("coef", float("nan"))
            s = d["shapley_r2"].get(term, float("nan"))
            L.append(f"| {term} | {o['coef']:+.2f} ({o['se']:.2f}) | {o['p']:.2g} | {m:+.2f} | "
                     f"{'' if s != s else f'{s:.3f}'} |")
        L.append(f"| total R² | | | | {d['shapley_r2']['total_r2']:.3f} |")
    w = res["within_scene_word"]
    L.append(f"\n## Within-scene word effect vs twin separation\n\nn={w['n']} pairs; slope "
             f"{w['slope_bins_per_width']:+.2f} bins per image width (p={w['slope_p']:.3g}); intercept "
             f"{w['intercept_bins']:+.2f} bins; median right-minus-left {w['median_diff_bins']:+.2f} bins.")
    dp = res["demonstration_prior"]
    L.append(f"\n## Demonstration prior (Bridge harvest)\n\n- image-right share of early motion: "
             f"{pct(dp['image_right_share'])} (n={dp['n']}); over 5 mm: {pct(dp['image_right_share_over_5mm'])} "
             f"(n={dp['n_over_5mm']}); by split {dp['by_split']}")
    for term, e in dp["word_vs_demo"].items():
        L.append(f"- natural '{term}' instructions: demonstration moves image-left in {pct(e['demo_image_left'])} "
                 f"(n={e['n']}; CI {pct(e['ci'][0])}–{pct(e['ci'][1])})")
    tp = res["tracking_prior"]
    L.append(f"\n## Single-object frames (NB03, n={tp['n']})\n\n- demos: image-right {pct(tp['demo_image_right'])}, "
             f"toward object {pct(tp['demo_toward_object'])}; objects right of gripper {pct(tp['object_right_share'])}")
    for cond in ("neutral", "mirror_neutral"):
        e = tp[cond]
        L.append(f"- model {cond}: image-right among decided {pct(e['model_image_right_decided'])}; sub-bin "
                 f"{pct(e['sub_bin_share'])}; reached object-left {pct(e['object_left']['reached_counting_sub_bin_as_miss'])}"
                 f" (sub-bin {pct(e['object_left']['sub_bin_share'])}), object-right "
                 f"{pct(e['object_right']['reached_counting_sub_bin_as_miss'])} (sub-bin {pct(e['object_right']['sub_bin_share'])})")
    mv = tp["magnitude_vs_distance"]
    L.append(f"- step size vs distance to the object: model ρ={mv['model_rho']:+.2f} (p={mv['model_p']:.2g}, "
             f"n={mv['model_n']}); demos ρ={mv['demo_rho']:+.2f} (p={mv['demo_p']:.2g})")
    la = res["language_audit"]
    L.append(f"\n## How Bridge uses lateral words ({la['episodes']} harvested episodes)\n\n"
             f"- {la['with_lateral_word']} instructions ({pct(la['share_with_lateral_word'])}) contain left/right\n"
             f"- destination or motion direction: {la['destination_or_direction']}; object's starting place "
             f"('from the left side ...'): {la['source_location']}; unclassified (mostly 'right' as an intensifier): "
             f"{la['unclassified']}; object-selecting rule matches: {la['referent_rule_matches']}")
    for t in la["referent_examples"]:
        L.append(f"  - referent-rule match: \"{t}\"")
    ra = res["role_association"]
    L.append("\nDemonstrated first motion (>5 mm) after a single lateral word, by the word's role:\n")
    for role in ("source", "destination"):
        e = ra[role]
        L.append(f"- {role}: image-left after 'left' {pct(e['left']['image_left'])} (n={e['left']['n']}) vs after "
                 f"'right' {pct(e['right']['image_left'])} (n={e['right']['n']}); Fisher p={e['fisher_p']:.2g}")
    oi = res["outer_inner"]
    L.append("\n## Outer vs inner twin (same-side scenes; reach toward the twins' side, bins)\n")
    for name, e in oi.items():
        L.append(f"- {name}: n={e['n']}; median outer−inner {e['median_outer_minus_inner_bins']:+.2f}; "
                 f"outer further in {pct(e['share_outer_further'])}; p={e['wilcoxon_p']:.3g}")
    ds = res["distribution"]
    L.append(f"\n## Readout diagnostics\n\n- argmax and expected value disagree in sign on "
             f"{pct(ds['sign_disagreement_argmax_vs_expected'])} of baseline predictions (n={ds['n']})")
    ec = ds["entropy_conflict"]
    L.append(f"- dy entropy when the named side conflicts with the term-free action: {ec['median_h_conflict']:.2f} vs "
             f"{ec['median_h_no_conflict']:.2f} nats (n={ec['n']}, p={ec['wilcoxon_p']:.3g})")
    dx = ds["dx_prefix"]
    L.append(f"- the greedy dx token differs between left and right in {pct(dx['share_dx_token_differs'])}; median "
             f"|Δc1| {dx['median_abs_dc1_same_dx']:.1f} bins with the same dx token vs "
             f"{dx['median_abs_dc1_diff_dx']:.1f} with a different one")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
