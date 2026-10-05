"""
reanalysis.py: corrected and extended analyses of the constructed-scene probe.

Additive to `analysis.py`. The notebook pipeline is left unchanged; this module
fixes the issues listed in docs/review_issues.md and adds the analyses the
workshop papers need. Everything here is a pure function of the logged CSVs, so
it runs on a laptop without a GPU or the model.

Conventions used throughout:

  * The lateral channel is component 1 (`c1`, continuous; `a1`, argmax), and an
    object further right in the image implies a more negative value
    (`compose_scenes.IMAGE_X_TO_LATERAL_SIGN = -1`). Where a quantity is
    reported "image-right in bins", it is `-c1 / ONE_BIN`.
  * `configuration` and the target signs are read from the prediction log,
    where Notebook 05 wrote the hand-resolved arrangement. The recorded
    construction geometry (instance and gripper x) comes from the constructed
    manifest.
  * Instruction A always names the left instance and B the right one: every
    synthesised instruction uses the term "left", and B is its antonym swap.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import analysis as A
from compose_scenes import IMAGE_X_TO_LATERAL_SIGN

# The one-bin floor the dissertation's resolution-aware analyses used. Kept as
# the decision threshold for "the model picked a side" so results stay
# comparable; the true spacing of the argmax grid is derived from the data
# (`argmax_step`), because the pipeline's width divides by 254, not 255.
ONE_BIN = 0.000325

BASELINE, NEUTRAL = "baseline", "neutral"
MIRROR, MIRROR_NEUTRAL = "mirror", "mirror_neutral"
SWAPPED = "swapped_scene"
CONFIGS = ("same_side_left", "opposite", "same_side_right")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_probe(data_dir: str) -> pd.DataFrame:
    """Constructed-scene rows of the probe log that carry the continuous readout."""
    log = pd.read_csv(os.path.join(data_dir, "probe_predictions.csv"))
    covered = log[A.CONTINUOUS_COLS].notna().all(axis=1)
    log = log[covered & (log["scene_source"] == "constructed")].copy()
    log["scene_id"] = log["scene_id"].astype(str)
    log["base_scene_id"] = log["base_scene_id"].astype(int)
    return log


def load_constructed(data_dir: str) -> pd.DataFrame:
    """Constructed manifest, with a flag for membership of the frozen set."""
    path = os.path.join(data_dir, "constructed")
    manifest = pd.read_csv(os.path.join(path, "constructed_manifest.csv"))
    frozen = pd.read_csv(os.path.join(path, "evaluation_set.csv"))
    manifest["frozen"] = manifest["construct_id"].isin(set(frozen["construct_id"]))
    return manifest


def load_bridge_manifest(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "bridge", "manifest.csv"),
                       low_memory=False)


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def cluster_bootstrap(frame: pd.DataFrame, statistic, *, cluster: str = "base_scene_id",
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile interval for `statistic(frame)` resampling whole clusters.

    Scenes built from one base frame share a background, an object and a cutout,
    so they are not independent observations; resampling base frames keeps
    that dependence inside every replicate.
    """
    rng = np.random.default_rng(seed)
    groups = {key: g for key, g in frame.groupby(cluster)}
    keys = np.array(list(groups))
    point = statistic(frame)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        draws.append(statistic(pd.concat([groups[k] for k in sample], ignore_index=True)))
    draws = np.asarray(draws, dtype=float)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def argmax_step(values) -> float:
    """Spacing of the argmax grid, recovered from the logged argmax actions.

    De-normalised bin centres are equally spaced, so every difference between
    two distinct argmax values is a whole number of steps. The smallest
    positive difference among the distinct values is one step (float32 noise
    is removed by taking the median of the near-minimal gaps).
    """
    grid = np.unique(np.round(np.asarray(values, dtype=float), 9))
    gaps = np.diff(grid)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float("nan")
    smallest = gaps.min()
    near = gaps[gaps < 1.5 * smallest]
    return float(np.median(near))


# --------------------------------------------------------------------------- #
# Scene geometry
# --------------------------------------------------------------------------- #
def scene_geometry(log: pd.DataFrame, constructed: pd.DataFrame) -> pd.DataFrame:
    """One row per probed scene: resolved arrangement plus recorded geometry.

    `configuration` is the arrangement the probe ran under (hand label where it
    disagreed with the recorded one); `recorded_configuration` is what was
    placed. Instance positions are exact by construction; the gripper position
    is a detection or, where detection failed, the image centre.
    """
    per_scene = (log.drop_duplicates("scene_id")
                 .set_index("scene_id")[["base_scene_id", "configuration",
                                         "target_sign_a_image", "target_sign_b_image"]])
    manifest = constructed.set_index("construct_id")
    cols = ["configuration", "image_width", "x_source", "x_pasted", "x_gripper",
            "gripper_source", "separation_px", "noun", "human_configuration"]
    geo = per_scene.join(manifest[cols].rename(
        columns={"configuration": "recorded_configuration"}), how="left")
    geo["relabelled"] = geo["configuration"] != geo["recorded_configuration"]
    geo["x_left"] = geo[["x_source", "x_pasted"]].min(axis=1)
    geo["x_right"] = geo[["x_source", "x_pasted"]].max(axis=1)
    geo["paste_is_right"] = geo["x_pasted"] > geo["x_source"]
    geo["detected_gripper"] = geo["gripper_source"] == "detected"
    return geo


# --------------------------------------------------------------------------- #
# Paired predictions per scene
# --------------------------------------------------------------------------- #
def paired(log: pd.DataFrame, condition: str, col: str = "c1") -> pd.DataFrame:
    """One row per scene with the value for each role under `condition`."""
    sub = log[log["condition"] == condition]
    wide = sub.pivot_table(index="scene_id", columns="role", values=col, aggfunc="mean")
    return wide


def decided(values, floor: float = ONE_BIN) -> np.ndarray:
    """A prediction picks a side when it is at least `floor` away from zero."""
    values = np.abs(np.asarray(values, dtype=float))
    return values >= floor if floor > 0 else values > 0


def image_right(c1) -> np.ndarray:
    """True where the lateral prediction moves image-right."""
    return np.sign(np.asarray(c1, dtype=float)) * IMAGE_X_TO_LATERAL_SIGN > 0


def scene_outcomes(log: pd.DataFrame, geo: pd.DataFrame, *, condition: str = BASELINE,
                   floor: float = ONE_BIN, mirrored: bool = False) -> pd.DataFrame:
    """Per-scene outcomes of the left/right pair under one condition.

    For mirrored images the target sides are swapped as well as negated: after
    a horizontal flip "the X on the left" names the instance that used to be on
    the right, so its side is minus the original B target's side (and vice
    versa). Negating each role's own side, as Notebook 06 did, inverts the
    opposite-arrangement scores.
    """
    wide = paired(log, condition)
    out = wide[["a", "b"]].join(geo, how="inner")
    t_a = out["target_sign_a_image"].astype(int).to_numpy()
    t_b = out["target_sign_b_image"].astype(int).to_numpy()
    if mirrored:
        t_a, t_b = -t_b, -t_a
    # Side of the action space each target lies in (c1 sign convention).
    side_a = t_a * IMAGE_X_TO_LATERAL_SIGN
    side_b = t_b * IMAGE_X_TO_LATERAL_SIGN
    a, b = out["a"].to_numpy(), out["b"].to_numpy()
    out = out.assign(
        t_a=t_a, t_b=t_b,
        resolved=decided(a, floor) & decided(b, floor),
        same_sign=a * b > 0,
        agree_a=np.sign(a) == side_a,
        agree_b=np.sign(b) == side_b,
        right_a=image_right(a), right_b=image_right(b),
        # Positive when "left" moves further image-left than "right" does, the
        # direction both grounding and a word-to-direction mapping predict.
        oriented=(a - b) * (-IMAGE_X_TO_LATERAL_SIGN),
    )
    out["both_correct"] = out["agree_a"] & out["agree_b"]
    out["right_share"] = (out["right_a"].astype(float) + out["right_b"].astype(float)) / 2.0
    return out


def headline(outcomes: pd.DataFrame, *, resolved_only: bool = True) -> dict:
    """Headline rates per arrangement from `scene_outcomes`."""
    rows = outcomes[outcomes["resolved"]] if resolved_only else outcomes
    table = {}
    for name in CONFIGS:
        g = rows[rows["configuration"] == name]
        if g.empty:
            continue
        table[name] = {
            "n": int(len(g)),
            "same_sign": float(g["same_sign"].mean()),
            "both_correct": float(g["both_correct"].mean()),
            "agree_a": float(g["agree_a"].mean()),
            "agree_b": float(g["agree_b"].mean()),
            "right_share": float(g["right_share"].mean()),
        }
    return table


def headline_with_ci(log, geo, *, floor=ONE_BIN, n_boot=2000, seed=0,
                     condition=BASELINE, mirrored=False) -> dict:
    """Headline rates with 95% intervals from a base-frame bootstrap."""
    outcomes = scene_outcomes(log, geo, condition=condition, floor=floor, mirrored=mirrored)
    rows = outcomes[outcomes["resolved"]]
    result = {}
    for name in CONFIGS:
        g = rows[rows["configuration"] == name]
        if g.empty:
            continue
        entry = {"n": int(len(g))}
        for metric in ("same_sign", "both_correct", "right_share"):
            entry[metric] = cluster_bootstrap(
                g, lambda f, m=metric: f[m].astype(float).mean(), n_boot=n_boot, seed=seed)
        result[name] = entry
    return result


# --------------------------------------------------------------------------- #
# R1: corrected instrument checks
# --------------------------------------------------------------------------- #
def manipulation_check(log: pd.DataFrame, geo: pd.DataFrame, *, floor: float = ONE_BIN) -> dict:
    """Does the term-free action go toward the pasted twin, on the identified channel?

    Only opposite arrangements are informative: there the two twins lie in
    opposite directions, so a single lateral action points at one of them. The
    pasted twin's side is taken from the order of the two instances, which is
    exact by construction and does not depend on where the gripper was located
    (the hand-resolved arrangement already guarantees they straddle the arm).
    Under the mirror the side is negated.

    Because the policy leans image-right, a raw toward-paste rate mixes
    perception with that lean. The contrast P(right | paste right) minus
    P(right | paste left) separates them: zero if the paste is ignored or
    treated exactly like the original, negative if only the original pulls,
    positive if the paste pulls harder.
    """
    out = {}
    opposite = geo[geo["configuration"] == "opposite"]
    for condition, flip in ((NEUTRAL, 1), (MIRROR_NEUTRAL, -1)):
        sub = log[(log["condition"] == condition) & (log["role"] == "n")]
        sub = sub.merge(opposite[["paste_is_right"]], left_on="scene_id", right_index=True)
        sub = sub[decided(sub["c1"], floor)]
        paste_right = np.where(sub["paste_is_right"], 1, -1) * flip > 0
        right = image_right(sub["c1"])
        toward = right == paste_right
        k, n = int(toward.sum()), int(len(sub))
        r_pr = right[paste_right].mean() if paste_right.any() else float("nan")
        r_pl = right[~paste_right].mean() if (~paste_right).any() else float("nan")
        table = np.array([[int(right[paste_right].sum()), int((~right[paste_right]).sum())],
                          [int(right[~paste_right].sum()), int((~right[~paste_right]).sum())]])
        p = float(stats.fisher_exact(table)[1]) if table.sum(axis=1).min() > 0 else float("nan")
        out[condition] = {
            "n": n, "toward_paste": k / n if n else float("nan"),
            "ci": wilson_interval(k, n),
            "p_right_given_paste_right": float(r_pr), "n_paste_right": int(paste_right.sum()),
            "p_right_given_paste_left": float(r_pl), "n_paste_left": int((~paste_right).sum()),
            "difference": float(r_pr - r_pl), "fisher_p": p,
        }
    return out


def paste_displacement(log: pd.DataFrame, geo: pd.DataFrame, *, condition: str = NEUTRAL,
                       role: str = "n") -> dict:
    """Within one base frame, does moving only the pasted twin move the action?

    Each base frame contributes a same-side scene (paste next to the original)
    and an opposite scene (paste reflected across the gripper). Background,
    object, cutout and instruction are identical; only the paste's position
    differs. If the model sees the paste, the action shifts toward where the
    paste went. Uses recorded construction (what was physically placed), so it
    does not depend on the hand relabelling or on gripper detection.
    """
    sub = log[(log["condition"] == condition) & (log["role"] == role)][["scene_id", "c1"]]
    sub = sub.merge(geo[["base_scene_id", "recorded_configuration", "x_pasted"]],
                    left_on="scene_id", right_index=True)
    rows = []
    for base, g in sub.groupby("base_scene_id"):
        opp = g[g["recorded_configuration"] == "opposite"]
        same = g[g["recorded_configuration"].str.startswith("same_side")]
        if len(opp) != 1 or len(same) != 1:
            continue
        direction = np.sign(float(opp["x_pasted"].iloc[0]) - float(same["x_pasted"].iloc[0]))
        # Image-right displacement in bins, oriented along the paste's move.
        shift = (-(float(opp["c1"].iloc[0]) - float(same["c1"].iloc[0])) / ONE_BIN) * direction
        side_change = (int(image_right([opp["c1"].iloc[0]])[0])
                       - int(image_right([same["c1"].iloc[0]])[0])) * direction
        rows.append({"base_scene_id": base, "shift_bins": shift, "side_change": side_change})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_frames": 0}
    test = A.wilcoxon_paired(frame["shift_bins"].to_numpy())
    return {
        "n_frames": int(len(frame)),
        "median_shift_bins": float(frame["shift_bins"].median()),
        "mean_shift_bins": float(frame["shift_bins"].mean()),
        "wilcoxon_p": test["p_value"], "rank_biserial": test["rank_biserial"],
        "share_shifted_toward_paste": float((frame["shift_bins"] > 0).mean()),
        "net_side_changes_toward_paste": float(frame["side_change"].mean()),
        "ci_mean_shift": cluster_bootstrap(frame, lambda f: f["shift_bins"].mean()),
    }


def resolution_corrected(log: pd.DataFrame) -> dict:
    """Ties and sub-step differences under both readouts, with the true grid step."""
    step = argmax_step(log["a1"])
    out = {"argmax_step": step, "pipeline_bin_width": ONE_BIN}
    for readout, col in (("argmax", "a1"), ("continuous", "c1")):
        wide = paired(log, BASELINE, col).dropna(subset=["a", "b"])
        diff = (wide["a"] - wide["b"]).to_numpy()
        entry = {
            "n": int(diff.size),
            "exact_zero": float(np.mean(diff == 0)),
            "below_pipeline_width": float(np.mean(np.abs(diff) < ONE_BIN)),
        }
        if readout == "argmax":
            steps = np.rint(np.abs(diff) / step)
            entry.update({
                "one_step": float(np.mean(steps == 1)),
                "two_or_more_steps": float(np.mean(steps >= 2)),
                "changed_executable_action": float(np.mean(steps >= 1)),
            })
        else:
            entry["below_one_step"] = float(np.mean(np.abs(diff) < step))
        out[readout] = entry
    return out


def signed_contrast(log: pd.DataFrame, condition: str) -> dict:
    """Signed and unsigned summaries of the left-minus-right lateral contrast.

    `oriented` is positive when the "left" instruction moves further image-left
    than the "right" one, which grounding and a word-to-direction mapping both
    predict. A contrast that is large in magnitude but centred on zero moves
    the action without steering it.
    """
    wide = paired(log, condition).dropna(subset=["a", "b"])
    a, b = wide["a"].to_numpy(), wide["b"].to_numpy()
    oriented = (a - b) * (-IMAGE_X_TO_LATERAL_SIGN) / ONE_BIN
    test = A.wilcoxon_paired(oriented)
    out = {
        "n": int(len(wide)),
        "mean_abs_bins": float(np.mean(np.abs(oriented))),
        "median_abs_bins": float(np.median(np.abs(oriented))),
        "median_oriented_bins": float(np.median(oriented)),
        "share_left_more_leftward": float(np.mean(oriented > 0)),
        "wilcoxon_p": test["p_value"], "rank_biserial": test["rank_biserial"],
    }
    for label, floor in (("nonzero", 0.0), ("one_bin", ONE_BIN)):
        keep = decided(a, floor) & decided(b, floor)
        out[f"opposite_sign_{label}"] = float(np.mean(a[keep] * b[keep] < 0)) if keep.any() else float("nan")
        out[f"n_{label}"] = int(keep.sum())
    return out


def screening_by_configuration(data_dir: str) -> dict:
    """Blind approval rates per arrangement, with a test of their difference."""
    review = pd.read_csv(os.path.join(data_dir, "constructed", "constructed_review.csv"))
    decided_rows = review[review["decision"].isin(["approved", "rejected"])]
    out = {}
    for name in CONFIGS:
        g = decided_rows[decided_rows["configuration"] == name]
        k, n = int((g["decision"] == "approved").sum()), int(len(g))
        out[name] = {"screened": n, "approved": k, "rate": k / n if n else float("nan"),
                     "ci": wilson_interval(k, n),
                     "reasons": dict(g.loc[g["decision"] == "rejected", "reject_reason"].value_counts())}
    opp = out["opposite"]
    same_k = out["same_side_left"]["approved"] + out["same_side_right"]["approved"]
    same_n = out["same_side_left"]["screened"] + out["same_side_right"]["screened"]
    table = np.array([[opp["approved"], opp["screened"] - opp["approved"]],
                      [same_k, same_n - same_k]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    out["opposite_vs_same_side"] = {"same_side_rate": same_k / same_n, "chi2": float(chi2), "p": float(p)}
    return out


# --------------------------------------------------------------------------- #
# R2: what drives the lateral action
# --------------------------------------------------------------------------- #
def design_matrix(log: pd.DataFrame, geo: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Long table of predictions with geometric and linguistic predictors.

    Outcome `y` is the image-rightward lateral action in bins. Geometry is
    expressed in the coordinates of the image the model saw, so mirrored rows
    are reflected (x -> W - x). Predictors, all as fractions of image width:

      layout    midpoint of the two twins relative to the gripper
      layout_img  the same midpoint relative to the image centre
      word      -1 for "left", +1 for "right", 0 when the term is removed
      grounded  offset of the named twin from the midpoint (0 without a term)
      source    offset of the original (non-pasted) twin from the midpoint
      demo      the base episode's demonstrated early lateral motion,
                image-right positive, standardised (reflected under the mirror)

    Under referent grounding `grounded` carries the same weight as `layout`; a
    word-to-direction mapping loads on `word`; a policy that follows the objects
    but ignores the term loads on `layout` alone.
    """
    keep = log[log["condition"].isin([BASELINE, NEUTRAL, MIRROR, MIRROR_NEUTRAL])].copy()
    keep = keep.merge(geo[["image_width", "x_source", "x_pasted", "x_gripper", "configuration",
                           "detected_gripper", "relabelled"]].rename(
                               columns={"configuration": "cfg"}),
                      left_on="scene_id", right_index=True)
    gt = bridge.set_index("episode_index")["gt_dy"]
    keep["demo_raw"] = -keep["base_scene_id"].map(gt).astype(float)  # image-right positive
    mirror = keep["condition"].isin([MIRROR, MIRROR_NEUTRAL]).to_numpy()
    width = keep["image_width"].astype(float).to_numpy()

    def img(x):
        x = keep[x].astype(float).to_numpy()
        return np.where(mirror, width - x, x)

    xs, xp, xg = img("x_source"), img("x_pasted"), img("x_gripper")
    x_left, x_right = np.minimum(xs, xp), np.maximum(xs, xp)
    x_mid = 0.5 * (x_left + x_right)
    role = keep["role"].to_numpy()
    word = np.select([role == "a", role == "b"], [-1.0, 1.0], default=0.0)
    x_named = np.select([role == "a", role == "b"], [x_left, x_right], default=x_mid)
    demo = np.where(mirror, -keep["demo_raw"], keep["demo_raw"])
    demo = (demo - np.nanmean(demo)) / np.nanstd(demo)
    return pd.DataFrame({
        "y": -keep["c1"].to_numpy() / ONE_BIN,
        "layout": (x_mid - xg) / width,
        "layout_img": (x_mid - width / 2.0) / width,
        "word": word,
        "grounded": (x_named - x_mid) / width,
        "source": (xs - x_mid) / width,
        "demo": demo,
        "mirror": mirror.astype(float),
        "base_scene_id": keep["base_scene_id"].to_numpy(),
        "scene_id": keep["scene_id"].to_numpy(),
        "role": role,
        "condition": keep["condition"].to_numpy(),
        "cfg": keep["cfg"].to_numpy(),
        "detected_gripper": keep["detected_gripper"].to_numpy(),
        "relabelled": keep["relabelled"].to_numpy(),
    }).dropna()


def _r2(y, X) -> float:
    if X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(1.0 - resid.var() / y.var())


def shapley_r2(frame: pd.DataFrame, blocks: dict, outcome: str = "y") -> dict:
    """Share of explained variance attributed to each predictor block (LMG)."""
    y = frame[outcome].to_numpy(dtype=float)
    names = list(blocks)
    cache = {}

    def r2(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            cols = [c for name in key for c in blocks[name]]
            cache[key] = _r2(y, frame[cols].to_numpy(dtype=float)) if cols else 0.0
        return cache[key]

    k = len(names)
    shares = {}
    for name in names:
        others = [n for n in names if n != name]
        total = 0.0
        for size in range(k):
            weight = 1.0 / (k * len(list(itertools.combinations(others, size))))
            for subset in itertools.combinations(others, size):
                total += weight * (r2(subset + (name,)) - r2(subset))
        shares[name] = total
    shares["total_r2"] = r2(tuple(names))
    return shares


def drivers(frame: pd.DataFrame) -> dict:
    """Regression of the image-rightward action on layout, word and referent."""
    import statsmodels.formula.api as smf

    formula = "y ~ layout + word + grounded + source + demo + mirror"
    ols = smf.ols(formula, data=frame).fit(cov_type="cluster",
                                           cov_kwds={"groups": frame["base_scene_id"]})
    out = {"n": int(len(frame)), "ols": {
        name: {"coef": float(ols.params[name]), "se": float(ols.bse[name]),
               "p": float(ols.pvalues[name])} for name in ols.params.index}}
    try:
        mixed = smf.mixedlm(formula, frame, groups=frame["base_scene_id"]).fit(reml=True)
        out["mixed"] = {name: {"coef": float(mixed.params[name]), "p": float(mixed.pvalues[name])}
                        for name in ols.params.index}
        out["mixed"]["group_var"] = float(mixed.cov_re.iloc[0, 0])
        out["mixed"]["resid_var"] = float(mixed.scale)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        out["mixed"] = {"error": str(exc)}
    blocks = {"layout": ["layout"], "word": ["word"], "grounded": ["grounded"],
              "source": ["source"], "demo": ["demo"], "mirror": ["mirror"]}
    out["shapley_r2"] = shapley_r2(frame, blocks)
    return out


def within_scene_word_effect(frame: pd.DataFrame) -> dict:
    """The right-minus-left difference in one scene, against the twins' separation.

    Grounding predicts a difference that grows with the separation between the
    twins (slope > 0, intercept ~ 0); a word-to-direction mapping predicts a
    positive intercept that does not depend on separation.
    """
    pairs = frame[frame["role"].isin(["a", "b"])]
    wide = pairs.pivot_table(index=["scene_id", "condition"], columns="role",
                             values=["y", "grounded"], aggfunc="mean").dropna()
    d = wide[("y", "b")] - wide[("y", "a")]
    sep = wide[("grounded", "b")] - wide[("grounded", "a")]
    slope, intercept, r, p, se = stats.linregress(sep.to_numpy(), d.to_numpy())
    return {"n": int(len(d)), "slope_bins_per_width": float(slope), "intercept_bins": float(intercept),
            "slope_p": float(p), "r": float(r), "median_diff_bins": float(d.median())}


# --------------------------------------------------------------------------- #
# R3: demonstration prior
# --------------------------------------------------------------------------- #
def demonstration_prior(bridge: pd.DataFrame) -> dict:
    """How often Bridge demonstrations start by moving image-right.

    gt_dy is the net early motion (five steps after the recorded no-op). Under
    the identified convention a negative gt_dy is image-right.
    """
    from data import make_pair

    gt = bridge["gt_dy"].astype(float)
    out = {"n": int(len(gt)),
           "image_right_share": float((gt < 0).mean()),
           "image_right_share_over_5mm": float((gt[gt.abs() > 0.005] < 0).mean()),
           "n_over_5mm": int((gt.abs() > 0.005).sum())}
    by_split = {}
    for split, g in bridge.groupby("split"):
        by_split[split] = {"n": int(len(g)), "image_right_share": float((g["gt_dy"] < 0).mean())}
    out["by_split"] = by_split
    # Natural referent instructions: does "left" predict a leftward demonstration?
    val = bridge[bridge["split"] == "validation"].copy()
    val["term"] = val["instruction"].map(lambda s: (make_pair(str(s)) or ("", ""))[0])
    word = {}
    for term in ("left", "right"):
        g = val[val["term"] == term]
        k = int((g["gt_dy"] > 0).sum())
        word[term] = {"n": int(len(g)), "demo_image_left": k / len(g) if len(g) else float("nan"),
                      "ci": wilson_interval(k, len(g))}
    out["word_vs_demo"] = word
    return out


def tracking_prior(data_dir: str, bridge: pd.DataFrame, *, floor: float = ONE_BIN) -> dict:
    """Model vs demonstration lateral direction on the single-object frames (NB03)."""
    base = os.path.join(data_dir, "bridge")
    geo = pd.read_csv(os.path.join(base, "object_tracking_set.csv"))
    review = pd.read_csv(os.path.join(base, "object_tracking_review.csv"))
    approved = set(review.loc[review["approved"] == "yes", "scene_id"])
    geo = geo[geo["scene_id"].isin(approved)].copy()
    log = pd.read_csv(os.path.join(data_dir, "object_tracking.csv"))
    gt = bridge.set_index("episode_index")["gt_dy"]
    geo["gt_dy"] = geo["episode_index"].map(gt).astype(float)
    geo["side"] = np.sign(geo["x_object"] - geo["x_gripper"])
    out = {"n": int(len(geo))}
    for condition in ("neutral", "mirror_neutral"):
        sub = log[log["condition"] == condition][["scene_id", "c1"]].merge(
            geo[["scene_id", "side", "gt_dy", "x_object", "x_gripper"]], on="scene_id")
        side = sub["side"] * (-1 if condition == "mirror_neutral" else 1)
        right = image_right(sub["c1"])
        dec = decided(sub["c1"], floor)
        entry = {"n": int(len(sub)),
                 "model_image_right_decided": float(right[dec].mean()),
                 "sub_bin_share": float((~dec).mean())}
        for label, s in (("object_left", -1), ("object_right", 1)):
            mask = side.to_numpy() == s
            toward = right == (s > 0)
            entry[label] = {"n": int(mask.sum()),
                            "reached_counting_sub_bin_as_miss": float((toward & dec)[mask].mean()),
                            "sub_bin_share": float((~dec)[mask].mean())}
        out[condition] = entry
    out["demo_image_right"] = float((geo["gt_dy"] < 0).mean())
    out["demo_toward_object"] = float((np.sign(-geo["gt_dy"]) == geo["side"]).mean())
    out["object_right_share"] = float((geo["side"] > 0).mean())
    # Does the model's step size grow with distance to a lone object?
    neutral = log[log["condition"] == "neutral"][["scene_id", "c1"]].merge(geo, on="scene_id")
    dist = (neutral["x_object"] - neutral["x_gripper"]).abs() / neutral["image_width"]
    toward = image_right(neutral["c1"]) == (neutral["side"] > 0)
    rho, p = stats.spearmanr(dist[toward], neutral.loc[toward, "c1"].abs())
    rho_gt, p_gt = stats.spearmanr(dist, neutral["gt_dy"].abs())
    out["magnitude_vs_distance"] = {"model_rho": float(rho), "model_p": float(p), "model_n": int(toward.sum()),
                                    "demo_rho": float(rho_gt), "demo_p": float(p_gt)}
    return out


LATERAL_WORD = r"\b(?:left|right|leftmost|rightmost)\b"

# A lateral word names where to put or move something when it follows a
# destination preposition or sits in a destination construction.
_DESTINATION = (
    r"\b(to|towards|toward|into|onto|on|in|at|near|next to|beside|under|over)\b"
    r"[^.]*?\b(the |its |a )?(far |upper |lower |top |bottom |back |front |middle |center |centre )*"
    r"(left|right)\b"
    r"|\b(left|right)\s*(-?hand )?(side|edge|corner|part|half|of|burner|end|area|portion|middle|center|centre)\b"
    r"|\b(move|slide|push|turn|shift|rotate|drag|pull|fold)\b[^.]*\b(left|right)\b"
)
# The lateral word picks out the manipulated object: prenominal ("the left
# cup"), superlative ("leftmost"), or a locative directly on the object of a
# pick-up verb before any destination clause ("pick up the cup on the left").
_REFERENT = (
    r"\b(left|right)most\b(?! (side|edge|corner|of))"
    r"|^(pick|grab|take|lift|get)( up)? (the |a )?(\w+ ){0,3}(on|at) the (left|right)\b"
    r"(?! (side|edge|corner|of|part|half|burner))(?!.*\b(place|put|move|drop|keep)\b)"
    r"|\bthe (left|right) one\b|\b(which|that) is (on|to) the (left|right)\b"
)
# The lateral word gives the manipulated object's starting place: it follows
# "from" with no destination preposition in between ("from the left side of
# the table", but not "from the pot to the left").
_SOURCE = r"\bfrom (?:the )?(?:(?!(?:to|into|onto|toward|towards)\b)\w+ ){0,2}(?:left|right)\b"


def language_audit(bridge: pd.DataFrame) -> dict:
    """How Bridge instructions use lateral words: object selection or destination.

    A referent-selection probe presupposes that training language uses
    "left"/"right" to choose which object to act on. This counts how often it
    does. A word that gives the object's starting place ("from the left side of
    the table") is counted as `source_location`: it locates the manipulated
    object, but never to choose between identical ones. The rules are
    transparent regular expressions; `referent_examples`
    lists every instruction the referent rule matched so the count can be
    checked by eye (in the v2 harvest those matches were themselves
    destinations or object locations, not choices between identical objects).
    """
    import re

    t = (bridge["instruction"].fillna("").astype(str).str.lower()
         .str.replace(r"\s+", " ", regex=True).str.strip())
    lateral = t.map(lambda s: bool(re.search(LATERAL_WORD, s)))
    sub = t[lateral]
    referent_rx, destination_rx = re.compile(_REFERENT), re.compile(_DESTINATION)
    source_rx = re.compile(_SOURCE)
    referent = sub.map(lambda s: bool(referent_rx.search(s)))
    source = sub.map(lambda s: bool(source_rx.search(s))) & ~referent
    destination = sub.map(lambda s: bool(destination_rx.search(s))) & ~referent & ~source
    return {
        "episodes": int(len(t)),
        "with_lateral_word": int(lateral.sum()),
        "share_with_lateral_word": float(lateral.mean()),
        "referent_rule_matches": int(referent.sum()),
        "source_location": int(source.sum()),
        "destination_or_direction": int(destination.sum()),
        "unclassified": int((~referent & ~source & ~destination).sum()),
        "referent_examples": sorted(set(sub[referent])),
        "source_examples": sorted(set(sub[source]))[:10],
        "unclassified_examples": sorted(set(sub[~referent & ~source & ~destination]))[:40],
    }


# --------------------------------------------------------------------------- #
# R4: outer vs inner twin (instance binding at the action level)
# --------------------------------------------------------------------------- #
def outer_inner(log: pd.DataFrame, geo: pd.DataFrame) -> dict:
    """On same-side scenes, does naming the outer twin move further than the inner one?

    Both twins lie in one direction, so a grounded policy moves the same way for
    both instructions and further for the twin that is further from the arm.
    `reach` is the lateral action toward the twins' side, in bins.
    """
    wide = paired(log, BASELINE).dropna(subset=["a", "b"]).join(geo, how="inner")
    rows = []
    for name, outer_role in (("same_side_left", "a"), ("same_side_right", "b")):
        g = wide[wide["configuration"] == name]
        toward = 1.0 if name == "same_side_right" else -1.0     # image direction of the twins
        reach = lambda c1: (-np.asarray(c1) / ONE_BIN) * toward  # noqa: E731
        inner_role = "b" if outer_role == "a" else "a"
        d = reach(g[outer_role]) - reach(g[inner_role])
        rows.append(pd.DataFrame({"configuration": name, "diff_bins": d,
                                  "base_scene_id": g["base_scene_id"].to_numpy()}))
    frame = pd.concat(rows, ignore_index=True)
    out = {}
    for name, g in list(frame.groupby("configuration")) + [("pooled", frame)]:
        test = A.wilcoxon_paired(g["diff_bins"].to_numpy())
        out[name] = {"n": int(len(g)), "median_outer_minus_inner_bins": float(g["diff_bins"].median()),
                     "share_outer_further": float((g["diff_bins"] > 0).mean()),
                     "wilcoxon_p": test["p_value"], "rank_biserial": test["rank_biserial"]}
    out["pooled"]["ci_mean"] = cluster_bootstrap(frame, lambda f: f["diff_bins"].mean())
    return out


# --------------------------------------------------------------------------- #
# R5: what the output distribution hides
# --------------------------------------------------------------------------- #
def distribution_diagnostics(log: pd.DataFrame, geo: pd.DataFrame) -> dict:
    """Readout disagreements, hesitation under conflict, and the dx prefix."""
    base = log[log["condition"] == BASELINE]
    both = base[(base["a1"] != 0) & (base["c1"] != 0)]
    out = {"sign_disagreement_argmax_vs_expected": float((np.sign(both["a1"]) != np.sign(both["c1"])).mean()),
           "n": int(len(both))}
    # Entropy when the named side conflicts with the term-free action's side.
    n_side = (log[log["condition"] == NEUTRAL].set_index("scene_id")["c1"]
              .pipe(lambda s: pd.Series(image_right(s), index=s.index)))
    wide_h = paired(log, BASELINE, "h1").join(geo[["target_sign_a_image", "target_sign_b_image"]], how="inner")
    wide_h = wide_h.join(n_side.rename("neutral_right"), how="inner")
    conflict_a = (wide_h["target_sign_a_image"] > 0) != wide_h["neutral_right"]
    conflict_b = (wide_h["target_sign_b_image"] > 0) != wide_h["neutral_right"]
    mixed = conflict_a != conflict_b
    h_conf = np.where(conflict_a, wide_h["a"], wide_h["b"])[mixed]
    h_free = np.where(conflict_a, wide_h["b"], wide_h["a"])[mixed]
    test = A.wilcoxon_paired(h_conf - h_free)
    out["entropy_conflict"] = {"n": int(mixed.sum()), "median_h_conflict": float(np.median(h_conf)),
                               "median_h_no_conflict": float(np.median(h_free)),
                               "median_diff": float(np.median(h_conf - h_free)), "wilcoxon_p": test["p_value"]}
    # The dy readout is conditioned on the greedy dx token already emitted.
    b0 = paired(log, BASELINE, "b0").dropna(subset=["a", "b"])
    c1 = paired(log, BASELINE, "c1").loc[b0.index]
    same_prefix = b0["a"] == b0["b"]
    dc = (c1["a"] - c1["b"]).abs() / ONE_BIN
    out["dx_prefix"] = {"share_dx_token_differs": float((~same_prefix).mean()),
                        "median_abs_dc1_same_dx": float(dc[same_prefix].median()),
                        "median_abs_dc1_diff_dx": float(dc[~same_prefix].median())}
    return out


# --------------------------------------------------------------------------- #
# GPU runs (scripts/run_gpu.py): replay, ladder, natural swap
# --------------------------------------------------------------------------- #
REPLAY_KEY = ["scene_id", "condition", "role", "image_scene_id"]


def load_run(path: str, readout: str = "folded") -> pd.DataFrame:
    """A run's log, concatenating `<stem>.shard<i>.csv` files when the run was sharded.

    GPU runs log two expected-value readouts: `c*`, the original slice over
    tokens 31745-31999, and `cf*`, all 256 action tokens with 31744 folded into
    bin 254 as OpenVLA's decoder does. With `readout="folded"` (the papers'
    convention) the folded values replace `c*`, so every analysis reads them,
    and the slice values are kept as `c*_255`. `readout="255"` keeps the slice.
    Logs without `cf*`, such as the original 4-bit log, are returned unchanged.
    """
    import glob

    if readout not in ("folded", "255"):
        raise ValueError(f"readout must be 'folded' or '255', got {readout!r}")
    stem, ext = os.path.splitext(path)
    files = sorted(glob.glob(f"{stem}.shard*{ext}")) or ([path] if os.path.exists(path) else [])
    if not files:
        raise FileNotFoundError(path)
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame["scene_id"] = frame["scene_id"].astype(str)
    if "image_scene_id" in frame:
        frame["image_scene_id"] = frame["image_scene_id"].astype(str)
    if readout == "folded" and all(f"cf{i}" in frame for i in range(7)):
        for i in range(7):
            frame[f"c{i}_255"] = frame[f"c{i}"]
            frame[f"c{i}"] = frame[f"cf{i}"]
    return frame


def compare_replay(original: pd.DataFrame, replay: pd.DataFrame) -> dict:
    """Agreement between a replayed run and the logged predictions it repeats.

    Argmax agreement is exact equality of the executable action per dimension;
    the continuous readout is compared in lateral bins. A replay on different
    hardware or precision that agrees on the argmax but drifts slightly in the
    expected value is still reproducing the executable behaviour. The original
    log carries only the 255-token slice, so a replay loaded with the folded
    readout is compared through its `c*_255` columns.
    """
    if all(f"c{i}_255" in replay for i in range(7)):
        replay = (replay.drop(columns=[f"c{i}" for i in range(7)])
                  .rename(columns={f"c{i}_255": f"c{i}" for i in range(7)}))
    cols = REPLAY_KEY + [f"a{i}" for i in range(7)] + [f"c{i}" for i in range(7)]
    merged = original[cols].merge(replay[cols], on=REPLAY_KEY, suffixes=("_orig", "_new"))
    out = {"n_matched": int(len(merged)), "n_replay": int(len(replay))}
    step = argmax_step(original["a1"])
    exact = np.ones(len(merged), dtype=bool)
    for i in range(7):
        same = np.isclose(merged[f"a{i}_orig"], merged[f"a{i}_new"], rtol=0, atol=1e-7)
        out[f"argmax_equal_a{i}"] = float(same.mean())
        if i < 6:  # the gripper dimension is not part of the probe
            exact &= same
    out["argmax_equal_translation_rotation"] = float(exact.mean())
    d_steps = np.rint(np.abs(merged["a1_orig"] - merged["a1_new"]) / step)
    out["a1_step_changes"] = {"0": float((d_steps == 0).mean()), "1": float((d_steps == 1).mean()),
                              ">=2": float((d_steps >= 2).mean())}
    dc = (merged["c1_new"] - merged["c1_orig"]).abs() / ONE_BIN
    out["c1_abs_diff_bins"] = {"median": float(dc.median()), "p95": float(dc.quantile(0.95)),
                               "max": float(dc.max())}
    out["c1_pearson"] = float(np.corrcoef(merged["c1_orig"], merged["c1_new"])[0, 1])
    out["c1_spearman"] = float(stats.spearmanr(merged["c1_orig"], merged["c1_new"])[0])
    both = decided(merged["c1_orig"]) & decided(merged["c1_new"])
    out["c1_sign_agreement_decided"] = float((np.sign(merged["c1_orig"]) == np.sign(merged["c1_new"]))[both].mean())
    return out


def readout_fix_effect(run: pd.DataFrame) -> dict:
    """How much the corrected readout (token 31744 folded into bin 254) changes values."""
    out = {}
    for i in range(7):
        if f"cf{i}" not in run or f"m{i}" not in run:
            continue
        sliced = run[f"c{i}_255"] if f"c{i}_255" in run else run[f"c{i}"]
        diff = (run[f"cf{i}"] - sliced).abs()
        out[f"dim{i}"] = {"max_abs_change": float(diff.max()), "share_changed_1e-6": float((diff > 1e-6).mean()),
                          "min_mass": float(run[f"m{i}"].min()), "median_mass": float(run[f"m{i}"].median())}
    return out


def _pair_table(run: pd.DataFrame, condition: str, transform: str = "original") -> pd.DataFrame:
    sub = run[(run["condition"] == condition) & (run["image_transform"] == transform)]
    return sub.pivot_table(index="scene_id", columns="role", values="c1", aggfunc="mean")


def ladder_summary(run: pd.DataFrame, original: pd.DataFrame, geo: pd.DataFrame) -> dict:
    """Left/right statistics for each wording in the instruction ladder.

    `move` has no referent: a lexical channel would send "move left" image-left
    (positive c1) and "move right" image-right on every scene. `paraphrase`
    names the same (left) twin twice, so its |Δ| is a noise floor for how much a
    meaning-preserving rewording moves the action.
    """
    out = {}
    base = paired(original, BASELINE)
    floor_rows = {}
    for condition in sorted(run["condition"].unique()):
        wide = _pair_table(run, condition).dropna(subset=["a", "b"])
        name = condition.replace("ladder_", "")
        a, b = wide["a"].to_numpy(), wide["b"].to_numpy()
        entry = {"n": int(len(wide)),
                 "median_abs_diff_bins": float(np.median(np.abs(a - b)) / ONE_BIN),
                 "mean_abs_diff_bins": float(np.mean(np.abs(a - b)) / ONE_BIN)}
        if name == "move":
            dec_a, dec_b = decided(a), decided(b)
            entry["move_left_goes_image_left"] = float((a[dec_a] > 0).mean()) if dec_a.any() else float("nan")
            entry["move_right_goes_image_right"] = float((b[dec_b] < 0).mean()) if dec_b.any() else float("nan")
            entry["oriented_median_bins"] = float(np.median(a - b) / ONE_BIN)
            entry["share_left_more_leftward"] = float(np.mean(a > b))
            test = A.wilcoxon_paired((a - b) / ONE_BIN)
            entry["wilcoxon_p"] = test["p_value"]
        elif name == "paraphrase":
            joined = wide.join(base[["a"]].rename(columns={"a": "baseline_a"}), how="inner")
            entry["vs_baseline_a_median_abs_bins"] = {
                role: float(np.median(np.abs(joined[role] - joined["baseline_a"])) / ONE_BIN) for role in ("a", "b")}
            floor_rows = joined
        else:
            # Same named twins as the baseline pair: score like the main analysis.
            fake = run[(run["condition"] == condition) & (run["image_transform"] == "original")].copy()
            fake["condition"] = BASELINE
            outcomes = scene_outcomes(fake, geo)
            entry["headline"] = headline(outcomes)
            sc = fake.assign()  # signed contrast on this wording
            wide2 = paired(sc, BASELINE).dropna(subset=["a", "b"])
            oriented = (wide2["a"] - wide2["b"]).to_numpy() / ONE_BIN
            entry["oriented_median_bins"] = float(np.median(oriented))
            entry["share_left_more_leftward"] = float(np.mean(oriented > 0))
            entry["wilcoxon_p"] = A.wilcoxon_paired(oriented)["p_value"]
        out[name] = entry
    lr = (base["a"] - base["b"]).abs().dropna() / ONE_BIN
    out["baseline_left_right_median_abs_bins"] = float(lr.median())
    return out


def natural_summary(run: pd.DataFrame, bridge: pd.DataFrame) -> dict:
    """Antonym swap on unedited frames: does the first action change with the word?

    Role a is the episode's own instruction, b its antonym swap, n the term
    removed. `left` is the version whose term says left (a or b, depending on
    the original term).
    """
    sub = run[run["image_transform"] == "original"]
    wide = sub.pivot_table(index="scene_id", columns="role", values="c1", aggfunc="mean")
    terms = sub.drop_duplicates("scene_id").set_index("scene_id")["spatial_term"].astype(str).str.lower()
    wide = wide.join(terms)
    wide = wide.dropna(subset=["a", "b"])
    left_first = wide["spatial_term"].isin(["left", "leftmost"])
    left = np.where(left_first, wide["a"], wide["b"])
    right = np.where(left_first, wide["b"], wide["a"])
    oriented = (left - right) / ONE_BIN
    dec = decided(left) & decided(right)
    out = {"n": int(len(wide)),
           "median_abs_diff_bins": float(np.median(np.abs(oriented))),
           "oriented_median_bins": float(np.median(oriented)),
           "share_left_more_leftward": float(np.mean(oriented > 0)),
           "wilcoxon_p": A.wilcoxon_paired(oriented)["p_value"],
           "same_sign_decided": float((np.sign(left) == np.sign(right))[dec].mean()) if dec.any() else float("nan"),
           "n_decided": int(dec.sum())}
    gt = bridge.assign(scene_id=bridge["episode_index"].map(lambda e: f"b{int(e):06d}")).set_index("scene_id")["gt_dy"]
    g = wide.join(gt.rename("gt_dy"), how="inner")
    for role in ("a", "b"):
        both = decided(g[role]) & (g["gt_dy"].abs() > 0.005)
        out[f"agrees_with_demo_{role}"] = float((np.sign(g[role]) == np.sign(g["gt_dy"]))[both].mean())
    return out


# --------------------------------------------------------------------------- #
# E1/E2 follow-ups: direction of the word effect per wording
# --------------------------------------------------------------------------- #
# A horizontal flip turns a both-left scene into a both-right one.
_MIRROR_CONFIG = {"same_side_left": "same_side_right", "same_side_right": "same_side_left",
                  "opposite": "opposite"}


def wording_log(run: pd.DataFrame, wording: str, transform: str = "original") -> pd.DataFrame:
    """One wording's rows, relabelled as the baseline pair so they score like it.

    Every ladder wording except `move` and `paraphrase` names the same twins as
    the baseline pair (a = left twin, b = right twin). `baseline` itself is read
    from a replay run, where it is a condition of its own.
    """
    want = BASELINE if wording == BASELINE else f"ladder_{wording}"
    log = run[(run["condition"] == want) & (run["image_transform"] == transform)].copy()
    log["condition"] = BASELINE
    return log


def wording_tests(log: pd.DataFrame, geo: pd.DataFrame, *, mirrored: bool = False,
                  floor: float = ONE_BIN) -> dict:
    """Decisive contrast, split direction and layout shares for one wording.

    The decisive test (same-side minus opposite same-direction rate) reads only
    whether the two actions share a sign, so it cannot tell a wording that
    separates the twins correctly from one that separates them backwards.
    `splits_right_way` supplies the direction: of the opposite scenes where the
    two instructions move in different directions, how many move toward the
    twins they name. On mirrored images the arrangements are relabelled by what
    the model saw.
    """
    outcomes = scene_outcomes(log, geo, floor=floor, mirrored=mirrored)
    if mirrored:
        outcomes["configuration"] = outcomes["configuration"].map(_MIRROR_CONFIG)
    rows = outcomes[outcomes["resolved"]]
    opposite = rows[rows["configuration"] == "opposite"]
    splits = opposite[~opposite["same_sign"]]
    right_way = int(splits["both_correct"].sum())
    oriented = outcomes["oriented"].to_numpy() / ONE_BIN
    test = A.same_side_test(log, min_magnitude=floor)
    by_config = {}
    for name in CONFIGS:
        g = rows[rows["configuration"] == name]
        if len(g):
            by_config[name] = {"n": int(len(g)), "same_sign": float(g["same_sign"].mean()),
                               "right_share": float(g["right_share"].mean())}
    return {
        "n_scenes": int(len(outcomes)),
        "by_configuration": by_config,
        "contrast_unpaired": test["contrast"],
        "contrast_paired": test["contrast_paired"],
        "opposite_n": int(len(opposite)),
        "opposite_splits": int(len(splits)),
        "splits_right_way": right_way,
        "splits_right_way_share": right_way / len(splits) if len(splits) else float("nan"),
        "splits_binomial_p": float(stats.binomtest(right_way, len(splits)).pvalue) if len(splits) else float("nan"),
        "opposite_both_correct": right_way / len(opposite) if len(opposite) else float("nan"),
        "median_abs_diff_bins": float(np.median(np.abs(oriented))),
        "oriented_median_bins": float(np.median(oriented)),
        "share_left_more_leftward": float(np.mean(oriented > 0)),
        "wilcoxon_p": A.wilcoxon_paired(oriented)["p_value"],
    }


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down decisions for a family of named p values."""
    order = sorted((k for k, p in pvalues.items() if np.isfinite(p)), key=pvalues.get)
    out, still = {}, True
    for i, name in enumerate(order):
        threshold = alpha / (len(order) - i)
        still = still and pvalues[name] <= threshold
        out[name] = {"p_value": float(pvalues[name]), "threshold": threshold, "reject": bool(still)}
    return out


def paraphrase_floor(replay: pd.DataFrame, ladder: pd.DataFrame) -> dict:
    """Left/right |Δ| against a meaning-preserving rewording, on the same scenes."""
    base = paired(wording_log(replay, BASELINE), BASELINE)
    para = paired(ladder[(ladder["condition"] == "ladder_paraphrase")
                         & (ladder["image_transform"] == "original")], "ladder_paraphrase")
    j = base.join(para, lsuffix="_lr", rsuffix="_pp").dropna()
    lr = (j["a_lr"] - j["b_lr"]).abs() / ONE_BIN
    pp = (j["a_pp"] - j["b_pp"]).abs() / ONE_BIN
    return {"n": int(len(j)), "left_right_median_bins": float(lr.median()),
            "paraphrase_median_bins": float(pp.median()),
            "share_left_right_larger": float(np.mean(lr > pp)),
            "wilcoxon_p": float(stats.wilcoxon(lr, pp).pvalue)}


def natural_agreement(run: pd.DataFrame, bridge: pd.DataFrame, *, demo_floor: float = 0.005,
                      floor: float = ONE_BIN) -> dict:
    """Unedited frames: which wording reproduces the demonstrated first motion?

    Compares the episode's own instruction (a), its antonym (b) and the term
    removed (n) on the frames where all three actions are decided and the
    demonstration moves more than `demo_floor` laterally, with exact McNemar
    tests between versions. Also tabulates how often the first motion is
    image-left after "left" vs "right" instructions, for the demonstrations and
    for the model under each version.
    """
    sub = run[run["image_transform"] == "original"]
    wide = sub.pivot_table(index="scene_id", columns="role", values="c1", aggfunc="mean")
    terms = sub.drop_duplicates("scene_id").set_index("scene_id")["spatial_term"].astype(str).str.lower()
    gt = bridge.assign(scene_id=bridge["episode_index"].map(lambda e: f"b{int(e):06d}")).set_index(
        "scene_id")["gt_dy"].rename("gt_dy")
    w = wide.join(terms).join(gt, how="inner").dropna(subset=["a", "b", "n"])
    common = w[decided(w["a"], floor) & decided(w["b"], floor) & decided(w["n"], floor)
               & (w["gt_dy"].abs() > demo_floor)]
    agree = {r: (image_right(common[r]) == image_right(common["gt_dy"])) for r in ("a", "b", "n")}
    out = {"n_common": int(len(common)), "agreement": {r: float(v.mean()) for r, v in agree.items()},
           "mcnemar": {}}
    for x, y in (("a", "b"), ("a", "n"), ("n", "b")):
        x_only, y_only = int((agree[x] & ~agree[y]).sum()), int((~agree[x] & agree[y]).sum())
        out["mcnemar"][f"{x}_vs_{y}"] = {
            "x_only": x_only, "y_only": y_only,
            "p_value": float(stats.binomtest(x_only, x_only + y_only).pvalue) if x_only + y_only else float("nan")}
    left_term = w["spatial_term"].isin(["left", "leftmost"])
    shares = {}
    for label, mask in (("left_instruction", left_term), ("right_instruction", ~left_term)):
        g = w[mask]
        demo = g[g["gt_dy"].abs() > demo_floor]["gt_dy"]
        entry = {"demo": float(np.mean(~image_right(demo))), "n_demo": int(len(demo))}
        for r in ("a", "n", "b"):
            v = g[r][decided(g[r], floor)]
            entry[r], entry[f"n_{r}"] = float(np.mean(~image_right(v))), int(len(v))
        shares[label] = entry
    out["image_left_share"] = shares
    return out


def phrase_counts(bridge: pd.DataFrame, nouns) -> dict:
    """How often Bridge instructions use each ladder wording's lateral phrase.

    `prenominal` counts "the left/right <noun>" for the probe's own nouns.
    `referent_rule` counts how many of the matches the R3 referent rule
    (`_REFERENT`) classifies as selecting an object rather than a destination.
    """
    import re

    t = (bridge["instruction"].fillna("").astype(str).str.lower()
         .str.replace(r"\s+", " ", regex=True).str.strip())
    noun_rx = "|".join(sorted({re.escape(str(n).lower()) for n in nouns if str(n).strip()}, key=len, reverse=True))
    phrases = {
        "baseline": r"\bon the (?:left|right)\W*$",
        "prenominal": rf"\bthe (?:left|right) (?:{noun_rx})\b",
        "object": r"\bobject on the (?:left|right)\b",
        "table_side": r"\b(?:left|right) side of the (?:table|counter|stove|sink)\b",
        "move": r"\b(?:move|slide|push)\b.*\b(?:left|right)\b",
    }
    referent_rx = re.compile(_REFERENT)
    out = {"episodes": int(len(t))}
    for name, rx in phrases.items():
        pattern = re.compile(rx)
        hits = t[t.map(lambda s: bool(pattern.search(s)))]
        out[name] = {"n": int(len(hits)),
                     "referent_rule": int(hits.map(lambda s: bool(referent_rx.search(s))).sum()),
                     "examples": sorted(set(hits))[:3]}
    return out


def word_role(instruction: str) -> str:
    """How an instruction's single lateral word is used.

    'source' (the object's starting place), 'destination' (where it goes, or a
    motion direction), or 'other'. Instructions with no lateral word or with
    more than one are 'none' and 'multiple'.
    """
    import re

    s = re.sub(r"\s+", " ", str(instruction).lower()).strip()
    words = re.findall(r"\b(?:left|right)\b", s)
    if not words:
        return "none"
    if len(words) > 1:
        return "multiple"
    if re.search(_SOURCE, s):
        return "source"
    if re.search(_DESTINATION, s):
        return "destination"
    return "other"


def role_association(bridge: pd.DataFrame, *, demo_floor: float = 0.005) -> dict:
    """Does the demonstrated first motion go toward the side a lateral word names?

    Splits single-lateral-word instructions by the word's role. When the word
    gives the object's starting place, the reach should go toward it; when it
    names a destination, the object (and so the reach) usually starts on the
    other side. The source rows also check the language frame: if "from the
    left" demonstrations move image-left, annotators' "left" is image-left.
    """
    t = bridge["instruction"].fillna("").astype(str)
    roles = t.map(word_role)
    word = t.str.lower().str.extract(r"\b(left|right)\b", expand=False)
    out = {}
    for role in ("source", "destination"):
        entry = {}
        for w in ("left", "right"):
            g = bridge.loc[(roles == role) & (word == w), "gt_dy"].astype(float)
            g = g[g.abs() > demo_floor]
            k = int(np.sum(~image_right(g)))
            entry[w] = {"n": int(len(g)), "image_left": k / len(g) if len(g) else float("nan"),
                        "ci": wilson_interval(k, len(g))}
        if entry["left"]["n"] and entry["right"]["n"]:
            table = [[round(entry[w]["image_left"] * entry[w]["n"]), entry[w]["n"] - round(entry[w]["image_left"] * entry[w]["n"])]
                     for w in ("left", "right")]
            entry["fisher_p"] = float(stats.fisher_exact(table)[1])
        out[role] = entry
    out["counts"] = {k: int(v) for k, v in roles.value_counts().items()}
    return out


def natural_by_role(run: pd.DataFrame) -> dict:
    """Natural-frame antonym swap, split by how the episode's lateral word is used.

    + means the "left" version moves further image-left than the "right" one.
    """
    sub = run[run["image_transform"] == "original"]
    wide = sub.pivot_table(index="scene_id", columns="role", values="c1", aggfunc="mean")
    info = sub[sub["role"] == "a"].drop_duplicates("scene_id").set_index("scene_id")[["instruction", "spatial_term"]]
    w = wide.join(info).dropna(subset=["a", "b"])
    left_first = w["spatial_term"].astype(str).str.lower().isin(["left", "leftmost"])
    left_more = (np.where(left_first, w["a"] - w["b"], w["b"] - w["a"]) * -IMAGE_X_TO_LATERAL_SIGN) / ONE_BIN
    w = w.assign(left_more=left_more, kind=w["instruction"].map(word_role))
    out = {}
    for kind, g in w.groupby("kind"):
        v = g["left_more"].to_numpy()
        out[str(kind)] = {"n": int(len(v)), "share_left_more_leftward": float(np.mean(v > 0)),
                          "median_bins": float(np.median(v)), "mean_bins": float(np.mean(v)),
                          "wilcoxon_p": A.wilcoxon_paired(v)["p_value"]}
    return out
