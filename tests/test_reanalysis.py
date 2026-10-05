"""Tests for reanalysis.py: corrected checks on synthetic scenes with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reanalysis as R

W = 640
STEP = 0.001  # well above ONE_BIN, so every synthetic action is "decided"


def _constructed(rows):
    """Minimal constructed manifest from (construct_id, base, config, x_src, x_paste, x_grip)."""
    out = []
    for cid, base, config, xs, xp, xg in rows:
        out.append({
            "construct_id": cid, "base_scene_id": base, "configuration": config,
            "image_width": W, "x_source": xs, "x_pasted": xp, "x_gripper": xg,
            "gripper_source": "detected", "separation_px": abs(xp - xs), "noun": "cup",
            "human_configuration": config,
        })
    return pd.DataFrame(out)


def _sides(xs, xp, xg):
    """Image sides of the left-named and right-named twins relative to the gripper."""
    x_left, x_right = min(xs, xp), max(xs, xp)
    return int(np.sign(x_left - xg)), int(np.sign(x_right - xg))


def _log(rows, values):
    """Probe-log rows. `values` maps (scene, condition, role) -> c1."""
    geo = {r[0]: r for r in rows}
    out = []
    for (cid, condition, role), c1 in values.items():
        _, base, config, xs, xp, xg = geo[cid]
        t_a, t_b = _sides(xs, xp, xg)
        out.append({"scene_id": cid, "base_scene_id": base, "configuration": config,
                    "condition": condition, "role": role, "c1": c1, "a1": c1,
                    "target_sign_a_image": t_a, "target_sign_b_image": t_b})
    return pd.DataFrame(out)


# Image-left motion is a positive lateral value (IMAGE_X_TO_LATERAL_SIGN = -1).
LEFT, RIGHT = +STEP, -STEP


def test_mirrored_scoring_swaps_the_named_twins():
    # Opposite scene: after a flip "left" names the twin that used to be on the right.
    rows = [("opp", 1, "opposite", 200, 440, 320),
            ("ssl", 2, "same_side_left", 100, 200, 320)]
    # A grounded policy on the mirrored images: in "opp" left->left, right->right;
    # in "ssl" both twins now sit image-right, so both instructions go right.
    values = {("opp", "mirror", "a"): LEFT, ("opp", "mirror", "b"): RIGHT,
              ("ssl", "mirror", "a"): RIGHT, ("ssl", "mirror", "b"): RIGHT}
    log = _log(rows, values)
    geo = R.scene_geometry(log, _constructed(rows))
    fixed = R.scene_outcomes(log, geo, condition="mirror", mirrored=True).set_index(
        pd.Index(["opp", "ssl"]))
    assert fixed.loc["opp", "both_correct"] and fixed.loc["ssl", "both_correct"]
    # The notebook's scoring (negate each role's own side) inverts the opposite scene.
    naive = log.copy()
    naive[["target_sign_a_image", "target_sign_b_image"]] *= -1
    geo_naive = R.scene_geometry(naive, _constructed(rows))
    wrong = R.scene_outcomes(naive, geo_naive, condition="mirror")
    assert not wrong.set_index(pd.Index(["opp", "ssl"])).loc["opp", "agree_a"]


def test_manipulation_check_reads_the_paste_side_from_instance_order():
    # Two opposite scenes, paste on the right in one and on the left in the other.
    rows = [("p_right", 1, "opposite", 200, 440, 320),
            ("p_left", 2, "opposite", 440, 200, 320)]
    # A policy that always reaches for the pasted twin; under the mirror it reverses.
    values = {("p_right", "neutral", "n"): RIGHT, ("p_left", "neutral", "n"): LEFT,
              ("p_right", "mirror_neutral", "n"): LEFT, ("p_left", "mirror_neutral", "n"): RIGHT}
    log = _log(rows, values)
    geo = R.scene_geometry(log, _constructed(rows))
    out = R.manipulation_check(log, geo)
    for condition in ("neutral", "mirror_neutral"):
        assert out[condition]["toward_paste"] == 1.0
        assert out[condition]["difference"] == 1.0


def test_manipulation_check_ignores_same_side_scenes():
    rows = [("ssl", 1, "same_side_left", 100, 200, 320)]
    log = _log(rows, {("ssl", "neutral", "n"): LEFT, ("ssl", "mirror_neutral", "n"): RIGHT})
    geo = R.scene_geometry(log, _constructed(rows))
    assert R.manipulation_check(log, geo)["neutral"]["n"] == 0


def test_paste_displacement_detects_a_policy_that_follows_the_paste():
    rows, values = [], {}
    for base in (1, 2, 3):
        same, opp = f"s{base}", f"o{base}"
        # Source left of the gripper; the paste sits beside it, or across the gripper.
        rows += [(same, base, "same_side_left", 200, 100, 320),
                 (opp, base, "opposite", 200, 440, 320)]
        values[(same, "neutral", "n")] = LEFT * base
        values[(opp, "neutral", "n")] = RIGHT * base
    log = _log(rows, values)
    geo = R.scene_geometry(log, _constructed(rows))
    out = R.paste_displacement(log, geo)
    assert out["n_frames"] == 3
    assert out["share_shifted_toward_paste"] == 1.0
    assert out["mean_shift_bins"] > 0


def test_argmax_step_recovers_the_grid_spacing():
    rng = np.random.default_rng(0)
    step, low = 0.0003238, -0.0417
    values = low + (rng.integers(0, 255, size=500) + 0.5) * step
    assert R.argmax_step(values.astype(np.float32)) == pytest.approx(step, rel=1e-3)


def test_outer_inner_is_positive_for_a_grounded_policy():
    rows = [("ssl", 1, "same_side_left", 100, 220, 320),
            ("ssr", 2, "same_side_right", 420, 540, 320)]
    # Outer twin: further from the gripper. "left" names it in ssl, "right" in ssr.
    values = {("ssl", "baseline", "a"): 3 * LEFT, ("ssl", "baseline", "b"): LEFT,
              ("ssr", "baseline", "a"): RIGHT, ("ssr", "baseline", "b"): 3 * RIGHT}
    log = _log(rows, values)
    geo = R.scene_geometry(log, _constructed(rows))
    out = R.outer_inner(log, geo)
    assert out["same_side_left"]["share_outer_further"] == 1.0
    assert out["same_side_right"]["share_outer_further"] == 1.0


def test_signed_contrast_separates_magnitude_from_direction():
    rows = [(f"s{i}", i, "opposite", 200, 440, 320) for i in range(4)]
    # Large left/right differences in alternating directions: big |Δ|, zero median.
    values = {}
    for i in range(4):
        flip = 1 if i % 2 == 0 else -1
        values[(f"s{i}", "baseline", "a")] = flip * 5 * STEP
        values[(f"s{i}", "baseline", "b")] = -flip * 5 * STEP
    out = R.signed_contrast(_log(rows, values), "baseline")
    assert out["mean_abs_bins"] > 20
    assert out["median_oriented_bins"] == pytest.approx(0.0)


def test_language_audit_tells_referents_from_destinations():
    bridge = pd.DataFrame({"instruction": [
        "pick up the cup on the left",             # referent
        "move the cup to the left of the pot",     # destination
        "place the spoon in the top right corner",  # destination
        "turn faucet left",                        # direction
        "put the pot right on top of the napkin",  # "right" as an intensifier
        "pick up the red cup",                     # no lateral word
    ]})
    out = R.language_audit(bridge)
    assert out["with_lateral_word"] == 5
    assert out["referent_rule_matches"] == 1
    assert out["destination_or_direction"] == 3
    assert out["unclassified"] == 1


def _as_run(log, transform="original"):
    """Give probe-log rows the columns a GPU run carries."""
    extra = {f"c{i}": 0.0 for i in (0, 2, 3, 4, 5, 6)}
    return log.assign(axis_index=1, image_transform=transform, **extra)


def test_wording_tests_reads_the_direction_of_opposite_scene_splits():
    rows = [(f"o{i}", i, "opposite", 200, 440, 320) for i in range(3)]
    rows += [(f"s{i}", i, "same_side_left", 100, 200, 320) for i in range(3)]
    # A backwards wording: on opposite scenes "left" goes image-right and "right"
    # image-left; on same-side scenes both go toward the twins.
    values = {}
    for i in range(3):
        values.update({(f"o{i}", "baseline", "a"): RIGHT, (f"o{i}", "baseline", "b"): LEFT,
                       (f"s{i}", "baseline", "a"): LEFT, (f"s{i}", "baseline", "b"): LEFT})
    log = _as_run(_log(rows, values))
    out = R.wording_tests(log, R.scene_geometry(log, _constructed(rows)))
    # The sign-blind decisive contrast scores a perfect separation...
    assert out["contrast_unpaired"]["difference"] == pytest.approx(1.0)
    # ...but every opposite-scene split goes to the wrong twin.
    assert out["opposite_splits"] == 3 and out["splits_right_way"] == 0
    assert out["oriented_median_bins"] < 0


def test_wording_tests_relabels_mirrored_arrangements():
    rows = [("ssl", 1, "same_side_left", 100, 200, 320), ("opp", 1, "opposite", 200, 440, 320)]
    # A grounded policy on mirrored images: the both-left scene now shows both
    # twins on the right, and in the opposite scene "left" names the old right twin.
    values = {("ssl", "baseline", "a"): RIGHT, ("ssl", "baseline", "b"): RIGHT,
              ("opp", "baseline", "a"): LEFT, ("opp", "baseline", "b"): RIGHT}
    log = _as_run(_log(rows, values), "mirror")
    out = R.wording_tests(log, R.scene_geometry(log, _constructed(rows)), mirrored=True)
    assert set(out["by_configuration"]) == {"opposite", "same_side_right"}
    assert out["by_configuration"]["same_side_right"]["right_share"] == 1.0
    assert out["splits_right_way"] == 1


def test_holm_steps_down_and_stops_at_the_first_failure():
    out = R.holm({"x": 0.001, "y": 0.02, "z": 0.04})
    assert all(v["reject"] for v in out.values())
    out = R.holm({"x": 0.03, "y": 0.04})
    assert not out["x"]["reject"] and not out["y"]["reject"]


def test_natural_agreement_scores_each_version_against_the_demonstration():
    run = pd.DataFrame([{"scene_id": "b000001", "role": r, "c1": v, "image_transform": "original",
                         "spatial_term": "left"} for r, v in (("a", LEFT), ("b", RIGHT), ("n", LEFT))])
    bridge = pd.DataFrame({"episode_index": [1], "gt_dy": [0.01]})  # demonstration moves image-left
    out = R.natural_agreement(run, bridge)
    assert out["n_common"] == 1
    assert out["agreement"] == {"a": 1.0, "b": 0.0, "n": 1.0}
    assert out["image_left_share"]["left_instruction"]["demo"] == 1.0


def test_phrase_counts_match_each_ladder_wording():
    bridge = pd.DataFrame({"instruction": [
        "move the cup to the left side of the table",   # table_side and move
        "pick up the left cup",                         # prenominal
        "place the spoon near the bowl on the left",    # ends with "on the left"
        "put the object on the left edge",              # object
        "pick up the left side of the cloth",           # not a probe noun
    ]})
    out = R.phrase_counts(bridge, ["cup", "spoon"])
    assert {k: out[k]["n"] for k in ("table_side", "move", "prenominal", "baseline", "object")} == {
        "table_side": 1, "move": 1, "prenominal": 1, "baseline": 1, "object": 1}


def test_word_role_tells_sources_from_destinations():
    assert R.word_role("move the spoon from the left side of the table to the pot") == "source"
    assert R.word_role("move the croissant from the pot to the left of the cloth") == "destination"
    assert R.word_role("put the cup on the right side of the table") == "destination"
    assert R.word_role("move the cup from left to right") == "multiple"
    assert R.word_role("pick up the cup") == "none"


def test_role_association_reads_image_left_from_the_lateral_convention():
    bridge = pd.DataFrame({
        "instruction": ["move the cup from the left side to the pot", "move the cup from the right side to the pot",
                        "put the cup on the left side", "put the cup on the right side"],
        # Image-left motion is positive: sources reach toward the named side,
        # destinations start on the other side.
        "gt_dy": [0.01, -0.01, -0.01, 0.01]})
    out = R.role_association(bridge)
    assert out["source"]["left"]["image_left"] == 1.0 and out["source"]["right"]["image_left"] == 0.0
    assert out["destination"]["left"]["image_left"] == 0.0 and out["destination"]["right"]["image_left"] == 1.0


def test_language_audit_counts_starting_places_separately():
    bridge = pd.DataFrame({"instruction": ["move the spoon from the left side of the table to the pot",
                                           "move the spoon to the left side of the table"]})
    out = R.language_audit(bridge)
    assert out["source_location"] == 1 and out["destination_or_direction"] == 1


def test_load_run_folded_readout_swaps_columns(tmp_path):
    """GPU runs read the folded readout by default and keep the 255-token slice."""
    import pandas as pd
    import reanalysis as R

    row = {"scene_id": "s1", "condition": "baseline", "role": "a", "image_scene_id": "s1"}
    row.update({f"c{i}": 0.1 * i for i in range(7)})
    row.update({f"cf{i}": 0.1 * i + 1.0 for i in range(7)})
    pd.DataFrame([row]).to_csv(tmp_path / "run.csv", index=False)

    folded = R.load_run(str(tmp_path / "run.csv"))
    assert all(abs(folded[f"c{i}"].iloc[0] - (0.1 * i + 1.0)) < 1e-12 for i in range(7))
    assert all(abs(folded[f"c{i}_255"].iloc[0] - 0.1 * i) < 1e-12 for i in range(7))

    sliced = R.load_run(str(tmp_path / "run.csv"), readout="255")
    assert all(abs(sliced[f"c{i}"].iloc[0] - 0.1 * i) < 1e-12 for i in range(7))
    assert "c0_255" not in sliced


def test_load_run_leaves_logs_without_folded_columns(tmp_path):
    """The original 4-bit log has no cf* columns and is returned unchanged."""
    import pandas as pd
    import reanalysis as R

    row = {"scene_id": "s1", "condition": "baseline", "role": "a"}
    row.update({f"c{i}": 0.1 * i for i in range(7)})
    pd.DataFrame([row]).to_csv(tmp_path / "log.csv", index=False)
    log = R.load_run(str(tmp_path / "log.csv"))
    assert "c0_255" not in log and abs(log["c3"].iloc[0] - 0.3) < 1e-12
