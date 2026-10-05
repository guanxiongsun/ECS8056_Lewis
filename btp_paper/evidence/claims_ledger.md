# BtP claims ledger: evidence audit

**Date:** 2 Oct 2026. **Scope:** every number the BtP paper ("Beneath the probe") intends to state, checked against the logged predictions and the GPU runs.
**Regenerate:** `.venv/bin/python docs/btp_paper/evidence/btp_evidence.py` (43 s on a laptop, no GPU). It writes `evidence.json` (every number below), `one_at_a_time.csv` and `one_at_a_time.md` next to this file. The §11 addenda then come from `.venv/bin/python docs/btp_paper/evidence/btp_evidence_addenda.py` (10 s), which writes `addenda.json`.
**Self-checks run on every execution** (`evidence.json: checks`, `claim10_example.variance_shares.matches_published_results_json`):
- `evaluate()` reproduces `reanalysis.wording_tests` in 24 of 24 run × wording cells.
- The fast bootstrap draws exactly as `reanalysis.cluster_bootstrap` does.
- The Shapley shares equal `outputs/reanalysis/results.json`.

**Data**
- **4-bit logs:** `google_drive/v2/probe_predictions.csv`, 2,720 predictions. These are 4-bit NF4 runs with bf16 compute on a Colab A100-SXM4-40GB. They carry only the 255-token readout `c*`.
- **GPU runs:** `outputs/runs/*.shard*.csv`, all on a GH200:
  - `replay_nf4` and `replay_bf16`: 2,720 rows each;
  - `ladder` and `ladder_bf16`: 8,160 rows each (340 scenes × 6 wordings × 2 roles × 2 image transforms);
  - `natural` and `natural_bf16`: 3,103 rows each.
- Each GPU run carries `c*` (token 31744 omitted), `cf*` (token 31744 folded into bin 254) and `m*` (token mass).

**Conventions** (done.md). These are used throughout unless a row says otherwise.
- **Channel:** dy = component 1. Image-right is negative.
- **Decided:** |dy| ≥ 0.000325 = (q99 − q01)/254 ("one bin").
- **Argmax grid step:** (q99 − q01)/255 = 0.0003238.
- **Rates:** computed over scenes where both instructions are decided. CIs are 95% base-frame cluster bootstraps with 2,000 draws.
- **Paired contrast:** same-side minus opposite same-sign rate, paired within base frame, exact McNemar test.
- **Splits:** opposite scenes where the two actions have opposite signs. A split goes the right way when each instruction moves toward its own twin; tested with a binomial test against 50%.
- **Signed median:** over all 340 left/right pairs, in bins. + means "left" moves further image-left. Tested with a Wilcoxon signed-rank test.
- **Mirrored images:** targets are swapped and negated. Layouts are labelled as the model saw them.

**Status legend**
- **VERIFIED:** the number and the wording stand.
- **NEEDS REWORDING:** the numbers are right but the sentence over- or under-states them; use the recommended wording.
- **UNSUPPORTED:** the data do not support the claim.
- **NEEDS GPU:** the claim needs a run that has not been done.

Recommended wordings are phrased for the paper: counterfactual and neutral, never as anyone's mistake. Provenance notes, such as "NB06", are for authors only and must not appear in the paper.

---

## 0. Read first: findings that affect several claims

**G1. The stated readout is not the one the reported numbers use.**
- `done.md` states the convention: expected value over 256 tokens, with token 31744 folded into bin 254 (`cf*`).
- The numbers actually reported use `c1`, the 255-token readout with 31744 omitted:
  - every number in `outputs/reanalysis/` (unavoidable, because the 4-bit logs have no `cf*`);
  - every number in `outputs/runs/`, because `run_gpu_analysis.py` → `reanalysis.paired/scene_outcomes` → `analysis.same_side_test` all read `c1`.
- **Effect of switching the GPU-run numbers to `cf1`:**
  - Counts change in 11 of 24 wording × run cells. Examples: 4-bit `object` splits 26/33 → 28/35; bf16 `object` 21/28 → 23/31.
  - No raw p < .05 decision flips.
  - **One Holm decision flips:** the 4-bit `object` paired contrast goes from reject to keep (p .013 → .031).
  - The `table_side` bound "p ≤ 6×10⁻⁶" holds only for `c1`; with `cf1` it is p ≤ 1.1×10⁻⁵.
- **Recommendation:** a paper about the token-map pitfall must itself use the folded readout.
  - For GPU runs, use the `cf1` numbers in this ledger (Table 1, claims 8 and 9).
  - Say explicitly that the 4-bit (Colab) logs carry only the 255-token readout.
  - Sources: `evidence.json: wording_grid.{c1_as_reported,cf1_convention}` and `claim3_token_map.headline_bf16`.

**G2. Which run is "the" result.**
- The plan proposes bf16 as the primary precision, but done.md's headline uses the 4-bit logs.
- Table 1's reference row is bf16 (`cf1`). The worked-example numbers in claim 10 are from the 4-bit logs.
- Always name the run.

**G3. In bf16 the original wording passes the sign-agreement test.** The paired contrast is +10.7 points (McNemar p = .039), whereas in 4-bit it is +4.1 (p = .55). Its splits still go the wrong way (7/21), and its signed effect is reversed (−0.15 bins, p = .011). A "null decisive contrast" is therefore precision-dependent. This strengthens claim 8.

**G4. Addition: where zero is.**
- dy's q01 and q99 are asymmetric (−0.04170 and +0.04086). OpenVLA's normalised zero (bin 127) therefore de-normalises to dy = −0.000424 = −1.3 bins.
- Physical zero is encoded as bin 128, which decodes to −0.31 bins.
- **Consequence for the argmax readout with a 1-bin floor:** one grid step image-right of the zero-motion token counts as a decided image-right action (bin 127, −1.3 bins). One step image-left does not (bin 129, +0.69 bins).
  - 7.2% of all logged argmax predictions sit on bin 127 (9.0% of baseline predictions).
  - Among decided argmax predictions, the image-right share is 69.0% with bin 127 and 66.1% without it.
- **For the expected value:** measuring from normalised zero instead of physical zero changes the image-right shares from 44 / 71 / 99% to 24 / 57 / 99% (Table 1, row `zero`, with one bin = the /255 grid step; 24 / 56 / 99% with the /254 width, §11.11).
- Physical zero is the correct reference. The row shows that the "rightward lean" statistic rests partly on actions within about 1–2 bins of zero.
- Source: `evidence.json: claim2_readout.argmax_grid`.

**G5. Stimulus overlaps in the ladder (minor).**
- "object" is one of the probe's nouns.
  - For 4 scenes the `object` wording is identical to the original wording (8 prompts).
  - 5 `absent_noun` scenes drew "object" as the absent noun, so their prompt equals the `object` wording. These 20 duplicate stimuli also give a determinism check (inventory b).
- Source: `evidence.json: inventory.ladder_overlaps`.

---

## 1. Action frame: dx vs dy

**Claim as worded.** Plan v2: "Reading `dx` instead of `dy` puts the manipulation check at chance (50%)."
The brief's earlier note on dx scored as lateral: "47% toward the object, 26% flips under mirroring, dx negative in about 80% of frames".

**Numbers.** Term-free ("neutral") instruction, 4-bit logs unless marked.

| Check | dx (c0) | dy (c1) |
|---|---|---|
| **A. Original pipeline check** (`compose_scenes.manipulation_rate`, as called in NB06 §10). Scenes whose recorded instances straddle the recorded gripper x (image-centre fallback included); any nonzero action counts. | **83/166 = 50.0%** [42.5, 57.5], binomial p = 1.0. By hand label: opposite 72/133 = 54.1%; both-left 3/12 = 25.0%; both-right 8/21 = 38.1% (reproduces the originally printed 50.0 / 54.1 / 25.0 / 38.1% exactly) | **66/166 = 39.8%** [32.6, 47.4], p = .010 (below one half). By hand label: opposite 49/133 = 36.8%; both-left 6/12 = 50.0%; both-right 11/21 = 52.4% |
| **B. Current check, toward-paste rate** (`reanalysis.manipulation_check`). Hand-resolved opposite scenes; paste side from the order of the two instances; ≥1 bin of the channel. | 78/149 = 52.3% [44.4, 60.2] (dx floor 0.000225) | 50/126 = 39.7% [31.6, 48.4] |
| **C. Paste-side contrast:** P(image-right ∣ paste right) − P(image-right ∣ paste left) | 86.1% (n = 79) vs 85.7% (n = 70): **+0.4 pts, Fisher p = 1.0** | 54.0% (n = 63) vs 74.6% (n = 63): **−20.6 pts, p = .025** |
| **D. Within-frame paste displacement:** 149 base frames; only the paste moves (`reanalysis.paste_displacement`) | median −1.4 dx bins, mean −2.6 [−5.2, 0.0]; 32.9% toward the paste; Wilcoxon p = .0007. The sign of "toward" depends on the sign convention assumed for dx; with the originally assumed +1 this reads 67.1% toward | median +1.5 bins, mean +7.6 [3.5, 12.1]; 70.5% toward the paste; p = 1.9×10⁻⁶ |
| **B–D in the bf16 replay** (`cf*`) | 76/144 = 52.8%; +1.0 pts, p = 1.0; displacement median −0.9 dx bins, p = .0025 | 46/121 = 38.0% [29.9, 46.9]; 58.6% vs 81.0%, −22.3 pts, p = .0096; displacement median +1.4, mean +4.9 [1.1, 9.2], 73.2%, p = 5.6×10⁻⁷ |

**Axis identification.** Recomputed from `object_tracking.csv` and the Bridge manifest; it reproduces the NB03 outputs exactly. The data are 98 approved unedited single-object frames.
- **Demonstrations:** the early motion points toward the object along dy in 96.9% of frames with image-right negative (97.5% of 40 object-left, 96.6% of 58 object-right), and in 3.1% under the opposite sign. dx manages at most 57.1% under either sign (+1: 52.5% object-left / 60.3% object-right). dz manages at most 56.1% (sign −1); under +1 it gives 43.9%, with a constant-sign pattern of 80.0% object-left vs 19.0% object-right.
- **Model, dy:** reverses sign under a horizontal flip in 65.6% of 64 decided frame pairs. With image-right negative it points at the object in 91.4% of 81 decided original frames (76.0% of 25 object-left, 98.2% of 56 object-right).
- **Model, dx scored as lateral with the originally assumed +1 sign:**
  - It points at the object in 47.2% of decided frames: 81.6% of 38 object-left vs 21.6% of 51 object-right.
  - It flips in 25.9% of 85 pairs.
  - It is negative in 79.8% (original) and 78.7% (mirrored) of decided frames.
  - dz flips in 16.8%.
- **Language check from Bridge:** after "from the left …", the demonstration's first motion is image-left (positive dy) in 81.8% of 22 episodes, against 8.3% of 24 after "from the right …" (claim 10).

**Definitions.** As in the table rows. dx's own one-bin floor is (q99 − q01)/254 = 0.000225.

**Source.**
- `btp_evidence.py::claim1_action_frame`, which calls `compose_scenes.manipulation_rate`, `reanalysis.manipulation_check`, `reanalysis.paste_displacement` and `analysis.object_tracking_by_sign`.
- Output: `evidence.json: claim1_action_frame.{nb06_original_definition, current_definition, axis_identification}`.
- Also `outputs/reanalysis/results.json: manipulation_check_c1.neutral, paste_displacement.neutral`, and the printed outputs of NB06 cell 26 and NB03 cells 30, 35, 38 and 40.

**Status: NEEDS REWORDING.**
- The 50.0% is exact. But "at chance" implies that dy passes the same statistic, and it does not: dy's raw toward-paste rate is 39.8% (39.7% under current definitions), significantly below one half.
- The raw rate mixes perception with the image-right lean and the original twin's stronger pull.
- The informative contrast between the channels is C, plus the within-frame displacement on dy (D).
- The brief's dx note (47% / 26% / about 80%) is **VERIFIED**.
- **Caveat for writers:** do not say dx ignores the paste. dx does shift when the paste moves within a frame (D, p = .0007); it is not lateral, and its shift can be read as "toward" or "away" depending on the sign assumed.

**Recommended wording.**
- "Scored on dx (component 0) instead of dy, the sign of the term-free action is unrelated to which side the pasted twin is on. P(image-right) is 86.1% vs 85.7% (Fisher p = 1.0), and the toward-paste rate is 50.0% (83/166)."
- "On dy the action depends on the paste's side (image-right 54.0% vs 74.6%, p = .025). Moving only the paste within a frame shifts dy toward it (median +1.5 bins, 70.5% of 149 frames, Wilcoxon p = 2×10⁻⁶)."
- "Report the side contrast, not the raw toward-paste rate. On dy the raw rate is 39.7% (50/126), because the original instance pulls harder and the action leans image-right."
- Axis, two sentences:
  - "In 98 unedited single-object Bridge frames, the demonstrated early motion points toward the object along dy in 96.9% of frames when image-right is negative dy (97.5% object-left, 96.6% object-right; 3.1% under the opposite sign), against at most 57% for dx and 56% for dz. OpenVLA's dy reverses under a horizontal flip in 65.6% of decided frames (dx 25.9%, dz 16.8%), and under the same sign it points at the object in 91.4% of decided frames."
  - Optional third: "Bridge's starting-place phrases agree: 'from the left …' is followed by image-left motion in 82% of 22 episodes vs 8% of 24 after 'from the right …'."

---

## 2. Readout: argmax vs expected value; bin width /254 vs /255

**Claim as worded.**
- done.md: "With the argmax readout, 32.6% of left/right contrasts are exact ties and another 11.8% differ by a single grid step."
- Plan v2: "Argmax ties in 33% of contrasts; dividing by 254 misfiles another 12%."

**Numbers.** Baseline left/right pairs, 4-bit logs, n = 340.
- **Argmax |Δdy| in grid steps:** 111 ties (32.6%), 40 one-step (11.8%), 189 two or more steps (55.6%). The expected value has 0 ties.
- **Widths:** the pipeline width is (q99 − q01)/254 = 0.00032503; the de-normalised argmax grid step is /255 = 0.00032376 (0.0003238 recovered from the data).
- **Effect on contrasts:** a one-step argmax contrast (0.00032376) falls below the 0.000325 width. A "below one bin" count therefore reports 44.4% (151/340) of contrasts as sub-resolution instead of 32.6%.
- **Effect on individual predictions:**
  - 0 of 2,720 logged predictions have |dy| in [0.0003238, 0.000325), argmax or expected value; 2 of 2,720 do in the bf16 `cf1` replay.
  - The width choice changes no individual prediction's "decided" status, and no Table 1 statistic: rows `ref` and `w254` are identical (§11.11).

**Headline by readout** (≥1 bin; layouts both-left / opposite / both-right):

| Run, readout | n | Same sign | Both correct | Image-right | Paired contrast | Splits right way | Signed median |
|---|---|---|---|---|---|---|---|
| 4-bit logs, expected value (c1) | 56 / 117 / 86 | 83.9 / 83.8 [76.7, 90.5] / 93.0 | 53.6 / **5.1** [1.7, 9.3] / 90.7 | 38 / 71 / 94 | +4.1 (p = .55) | 6/19 (p = .17) | −0.02 (p = .49) |
| 4-bit logs, argmax (a1) | 58 / 117 / 86 | 82.8 / 80.3 [73.3, 87.3] / 93.0 | 48.3 / **6.0** [1.7, 10.4] / 89.5 | 43 / 70 / 93 | +8.0 (p = .18) | 7/23 (p = .093) | 0.00 (p = .70; 33% ties) |
| bf16, expected value (cf1) | 59 / 112 / 87 | 89.8 / 81.2 [73.6, 88.4] / 98.9 | 50.8 / **6.2** [1.8, 10.8] / 98.9 | 44 / 71 / 99 | **+10.7 (p = .039)** | 7/21 (p = .19) | −0.15 (p = .011) |
| bf16, argmax (a1) | 55 / 108 / 87 | 89.1 / 81.5 [74.1, 88.8] / 96.6 | 49.1 / **8.3** [3.6, 13.9] / 96.6 | 45 / 72 / 98 | +6.9 (p = .23) | 9/20 (p = .82) | 0.00 (p = .038; r = −0.16; 30% ties) |

- The readouts disagree in sign on 5.1% of 680 baseline predictions.
- See G4 for the asymmetric argmax grid around zero.

**Definitions.**
- "Tie": identical argmax actions for "left" and "right".
- "One step": |Δa1| equals one grid step.
- "Misfile": a one-step argmax contrast is executable (the two actions differ by one bin), yet a /254 width classes it as "below one bin", together with the true ties.

**Source.**
- `btp_evidence.py::claim2_readout` → `evidence.json: claim2_readout.{resolution, contrast_steps, width_254_vs_255, argmax_grid, headline_by_readout}`.
- The same numbers are in `outputs/reanalysis/results.json: resolution.argmax.{exact_zero, one_step, below_pipeline_width}`.

**Status.**
- Ties and one-step: **VERIFIED**.
- "Misfiles another 12%": **NEEDS REWORDING.** It is the same 11.8% (40/340) of one-step contrasts, and it matters only for a contrast-level resolution count, not for any decided/undecided decision.
- **Readout verdict:** the readout changes no headline conclusion: opposite both-correct stays 5–8%, same-sign 80–84%, and the layout gradient holds. It does change secondary inference: the bf16 sign-agreement contrast goes from p = .039 to p = .23.

**Recommended wording.**
- "With the argmax readout, 32.6% of the 340 left/right pairs are exact ties and a further 11.8% differ by a single bin. The expected-value readout has no ties."
- "Taking one bin as (q99 − q01)/254 rather than the grid's /255 would classify those single-bin differences as sub-resolution (44.4% instead of 32.6%), although no individual action changes status."
- "Switching readouts leaves the headline unchanged (opposite both-correct 6.0% vs 5.1%) but moves the significance of the sign-agreement contrast in bf16 (p = .039 vs .23)."

---

## 3. Token map (token 31744 folded or left out)

**Claim as worded.**
- done.md: "Leaving token 31744 out … changes `dy` by ≥1 bin in only 1.1% of predictions but corrupts the gripper: its expected value averages 0.33 instead of 0.995."
- Plan v2: "Dropping token 31744 corrupts every gripper value."

**Numbers.** `cf` is the expected value with 31744 folded into bin 254; `c` leaves 31744 out and renormalises over 31745–31999.

| Run | n | dy changed ≥1 bin | Max change (bins) | Decided-sign flips | Gripper mean, 31744 out → folded | Gripper changed > 0.1 | > 0.01 | Crosses 0.5 |
|---|---|---|---|---|---|---|---|---|
| replay_nf4 | 2,720 | **1.14%** (31) | 75 | 0.13% | **0.327 → 0.995** | 89.3% | 97.2% | 65.4% |
| replay_bf16 | 2,720 | 0.88% (24) | 93 | 0.04% | 0.362 → 0.995 | 89.0% | 97.3% | 61.5% |
| ladder (4-bit) | 8,160 | 1.30% | 121 | 0.07% | 0.264 → 0.994 | 92.7% | 98.0% | 72.3% |
| ladder_bf16 | 8,160 | 1.75% | 172 | 0.22% | 0.297 → 0.995 | 92.3% | 98.0% | 69.1% |
| natural (4-bit) | 3,103 | 0.48% | 82 | 0.08% | 0.303 → 0.996 | 93.7% | 99.0% | 68.5% |
| natural_bf16 | 3,103 | 0.45% | 103 | 0.04% | 0.335 → 0.996 | 93.1% | 99.0% | 64.0% |
| **all GPU runs** | 27,966 | **1.19%** | | | **0.301 → 0.995** | **92.0%** | | **68.3%** |

- **Gripper state:** the argmax gripper is "open" (0.996) in 99.8–100% of predictions in each run. In the 4-bit logs (`c` only), 65.1% of predictions have an open argmax gripper but a 31744-omitted expected value below 0.5.
- **Mass:** dy's mass on the 256 action tokens is at least 0.959 in every run.
- **Large outliers:** the few large dy changes, up to +172 bins, come from predictions with heavy dy mass on the extreme top bin (tokens 31744 and 31745).
- **Effect on inference:**
  - On the bf16 headline, the fold changes only the signed median (−0.16 → −0.15 bins).
  - Across the 24 wording × run cells it changes counts in 11, flips no raw p < .05 decision, and flips one Holm decision (G1).

**Source.**
- `btp_evidence.py::claim3_token_map` → `evidence.json: claim3_token_map.{per_run, pooled, logged4bit, headline_bf16}`.
- `outputs/runs/results.json: replay_*.readout_fix` has only the maximum change and mass. The 1.1% was not in any committed output before this script.

**Status.**
- dy 1.1% and gripper 0.33 vs 0.995: **VERIFIED** (4-bit replay).
- "Corrupts every gripper value": **NEEDS REWORDING.** 92% of values move by more than 0.1 and 68% fall below 0.5; "every" is literally true only for any change at all (97–99% move by more than 0.01).

**Recommended wording.**
- "Leaving out the edge token 31744, which OpenVLA's decoder folds into the top bin, changes the lateral expected value by at least one bin in only 1.2% of 27,966 predictions."
- "It pulls the gripper's expected value from 0.995 to 0.30 on average: the value moves by more than 0.1 in 92% of predictions and drops below 0.5 (closed) in 68%, although the executed (argmax) gripper is open in over 99.7%."
- "A token-map error can be invisible on the probed axis and obvious on another."

---

## 4. Numerical precision (4-bit NF4 vs bf16) and the cross-GPU replay

**Claim as worded.** Plan v2: "4-bit vs bf16 changes 58% of lateral actions, but no conclusion." done.md: "Lateral argmax identical 91.9% (4-bit, GH200)".

**Numbers.** Lateral argmax token (a1) agreement on identical stimuli.

| Comparison | n | a1 changed | one step / ≥2 steps | r of expected dy | Sign agreement, both decided |
|---|---|---|---|---|---|
| 4-bit logs (A100) vs 4-bit GH200 replay | 2,720 | **8.1%** (221; identical 91.9%) | 1.2% / 6.9% | .968 | 99.3% (n = 2,236) |
| 4-bit logs (A100) vs bf16 GH200 replay | 2,720 | **58.1%** (1,581) | 8.2% / 49.9% | .727 | 90.4% (n = 2,081) |
| 4-bit GH200 vs bf16 GH200, replay (precision only) | 2,720 | 58.3% (1,585) | 8.2% / 50.0% | .723 (cf1 .709) | 90.2% |
| 4-bit GH200 vs bf16 GH200, ladder | 8,160 | 60.8% (4,959) | 9.0% / 51.7% | .652 | 87.2% |
| 4-bit GH200 vs bf16 GH200, natural frames | 3,103 | 43.5% (1,349) | 7.2% / 36.3% | .727 | 92.5% |

All six movement dimensions are identical in 76.5% (4-bit replay) and 5.5% (bf16 replay).

**Headline across precisions** (≥1 bin):

| Run | Opp. both correct | Opp. same sign | Same sign, both-left / both-right | Image-right | Paired contrast | Splits | Signed median |
|---|---|---|---|---|---|---|---|
| 4-bit logs (c1) | 5.1 [1.7, 9.3] | 83.8 | 83.9 / 93.0 | 38 / 71 / 94 | +4.1 (.55) | 6/19 | −0.02 (.49) |
| 4-bit GH200 (cf1) | 6.0 [1.8, 10.4] | 83.6 | 85.7 / 92.9 | 39 / 70 / 94 | +4.1 (.55) | 7/19 | −0.03 (.22) |
| bf16 GH200 (cf1) | 6.2 [1.8, 10.8] | 81.2 | 89.8 / 98.9 | 44 / 71 / 99 | **+10.7 (.039)** | 7/21 | **−0.15 (.011)** |

**Conclusions that held** across all three: opposite both-correct ≤ 6.2%; opposite same-sign 81–84%; image-right rising with layout; and the direction of every wording effect (claim 9).

**Secondary results that changed with precision** (4-bit → bf16):
- **Original wording, paired contrast:** +4.1 (p = .55) → +10.7 (p = .039).
- **Original wording, signed median:** −0.03 (p = .22) → −0.15 (p = .011).
- **Both-right rates:** same-sign 92.9 → 98.9%, both-correct 90.6 → 98.9% and image-right 94 → 99%.
  - These are the largest shifts: +8.3 points for both-correct, logged → bf16 +8.2. Every opposite-scene rate moves by at most 2.5 points.
  - With r = .72 < .9, the plan's pre-declared agreement criterion fails (plan v2 §7).
- **`table_side` right-way splits:** 25% → 12.5%.
- **`object` paired contrast:** +14.3 (p = .013) → +2.4 (p = .80) with c1, as reported; +12.7 (p = .031) → +3.7 (p = .61) with cf1.
- **Holm decisions on the paired contrast** (the report's family of 5 alternative wordings), as `outputs/runs/report.md` reports them (c1):
  - 4-bit: prenominal, table_side and object rejected;
  - bf16: none rejected;
  - with cf1, 4-bit rejects prenominal and table_side only.

**What differed between the A100 log and the GH200 4-bit replay** (from the CSV metadata, the notebook outputs and `outputs/runs/logs`).
- **Identical on both machines:**
  - torch 2.11.0+cu128, transformers 4.40.1 and bitsandbytes 0.50.2 (NB05 run; an earlier NB01 smoke run used 0.50.1);
  - Python 3.12;
  - NF4 with double quantisation and bf16 compute; eager attention; seed 42;
  - greedy decoding, batch size 1 with no attention mask, and the same prompt template with the empty token appended;
  - the same PNG frames.
- **Different:**
  - GPU: A100-SXM4-40GB (sm_80) vs GH200 120GB (sm_90);
  - CPU architecture: x86-64 vs aarch64;
  - driver: GH200 565.57.01; Colab's driver is not logged;
  - runner: `run_gpu.py` sets `torch.backends.cuda.matmul.allow_tf32 = False`. That is PyTorch's default, so probably not a difference.
- **Not logged:** Pillow, timm, tokenizers, torchvision and numpy versions.
- **Evidence for a hardware origin:**
  - The GH200 smoke test reproduced the logged a1 but not c1: −9.468e-05 logged vs −9.680e-05 (`logs/sgvla-smoke_6987042.out`).
  - An A100 smoke test on termitech reportedly reproduced a logged prediction exactly. That is a single prediction, and it is recorded only in `docs/workshop_plan.md`.
- **Mechanism (association):** the dx prefix amplifies the numerical difference.
  - The greedy dx token differs from the log in 5.0% of replayed predictions.
  - When it differs, dy's token changes in 84.7% of cases (median |Δdy| 6.0 bins); when it does not, in 4.1% (median 0.07 bins).
  - 54.5% of the ≥2-step dy changes come with a dx change.
  - From 4-bit to bf16, dx changes in 46%, and dy changes in 83% of those cases vs 37% otherwise.
- **Within the GH200 runs:** 20 duplicate stimuli that ran on different GPUs gave bit-identical outputs.

**Source.**
- `btp_evidence.py::claim4_precision` → `evidence.json: claim4_precision.{comparisons, headline, dx_prefix_amplification, environment}`.
- Also `outputs/runs/results.json: replay_{nf4,bf16}.compare` and `wording.*.holm`.

**Status.**
- 58% and 91.9%: **VERIFIED** (denominator 2,720).
- "But no conclusion": **NEEDS REWORDING.** No headline conclusion changes, but which secondary tests are significant does.

**Recommended wording.**
- "Switching inference from 4-bit NF4 to bf16 changes the lateral action token in 58% of 2,720 predictions (correlation of expected values .72). The opposite-scene rates move by at most 2.5 points (both-correct 6.0% vs 6.2%), and no headline rate moves by more than 8.3 points (both-right both-correct 90.6% vs 98.9%)."
- "But the switch changes which secondary tests are significant. The sign-agreement contrast on the original wording goes from p = .55 to p = .039, and the Holm-corrected contrasts go from two or three wordings to none."
- "Even at the same precision, moving from an A100 to a GH200 changes 8.1% of lateral tokens."

---

## 5. Autoregressive prefix

**Claim as worded.** Plan v2: "60% of word swaps change the `dx` token, which carries most of the `dy` change." done.md: "those swaps carry most of the change (5.9 vs 0.7 bins)".

**Numbers.** 340 baseline left/right pairs.

| Run | dx token differs | Median \|Δdy\|, same dx / different dx | Share of summed \|Δdy\| in different-dx pairs | Opposite-sign dy (both decided), same / different dx |
|---|---|---|---|---|
| 4-bit logs (c1) | **59.7%** (203/340) | **0.75 / 5.86 bins** (Mann-Whitney p = 5×10⁻¹⁴) | 84.5% | 9.6% / 15.5% |
| 4-bit GH200 (cf1) | 59.4% | 0.77 / 5.78 | 84.5% | 9.6% / 15.0% |
| bf16 GH200 (cf1) | 59.7% | 0.95 / 3.80 | 82.3% | 4.9% / 14.8% |

When the dx tokens differ, they are a median 7–8 bins apart.

**How dy is read** (verified in code):
- `scripts/run_gpu.py:predict` (l. 73–110) and `model.predict_action_dist` both call `vla.generate(do_sample=False, output_scores=True)` for 7 tokens.
- The dy distribution (scores[1]) is computed after the greedy dx token has been appended.
- Every logged dy is therefore E[dy ∣ image, prompt, own greedy dx token]. No run teacher-forces a common dx token or marginalises over dx.

**Source.**
- `btp_evidence.py::claim5_prefix` → `evidence.json: claim5_prefix`.
- `outputs/reanalysis/results.json: distribution.dx_prefix` (59.7%, 0.75, 5.86).

**Status: NEEDS REWORDING.** The numbers are verified. "Carries" is causal, and the prefix-control experiment has not been run (Table 1, row `gpu_prefix`: NEEDS GPU).

**Recommended wording.**
- "OpenVLA decodes dx before dy, and dy is read conditioned on the model's own dx token. In 59.7% of the 340 left/right pairs the two instructions produce different dx tokens. In those pairs dy differs by a median of 5.9 bins, against 0.7 bins when the dx token is shared, and they account for 85% of the total dy change."
- "Whether the word acts on dy directly or through the dx prefix cannot be separated without forcing a common prefix (not yet run)."

---

## 6. Scoring of mirrored images

**Claim as worded.** Plan v2: "Naively negating the targets inverts the opposite-scene result."

**Definitions.**
- **(a) correct:** after a horizontal flip, "left" names the other twin, so the targets are swapped and negated: t_a, t_b ← −t_b, −t_a. This is `reanalysis.scene_outcomes(mirrored=True)`.
- **(b) naive:** each role's own target is negated, t_a, t_b ← −t_a, −t_b. This is what the original pipeline (NB06) did. Under it the scorer expects "left" to go to the image-right twin, so the signed effect's expected direction flips too.
- **(c) none:** the original targets are kept.
- In all three, layouts are labelled as the model saw them. A flipped both-left scene is both-right.

**Numbers.** "Same sign" and "image-right" are identical under (a), (b) and (c).

| Mirrored run | Scoring | Same sign (L / opp / R) | Both correct (L / opp / R) | Opposite both-correct [CI] | Splits right way | Signed median |
|---|---|---|---|---|---|---|
| 4-bit logs, ≥1 bin (n = 67 / 114 / 69) | (a) correct | 79.1 / 81.6 / 89.9 | 52.2 / 7.0 / 82.6 | 7.0 [2.7, 12.3] | 8/21 (p = .38) | −0.16 (p = .061) |
| | (b) naive | same | 52.2 / **11.4** / 82.6 | 11.4 [6.1, 17.7] | **13/21** | **+0.16** |
| | (c) none | same | **26.9** / 7.0 / **7.2** | 7.0 | 8/21 | −0.16 |
| 4-bit logs, nonzero threshold (the original pipeline's) | (a) / (b) | — | opp 9.9 / **14.6** | — | 15/37 / 22/37 | |
| **bf16 replay (cf1), ≥1 bin** (n = 60 / 115 / 74) | (a) correct | 75.0 / 78.3 / 82.4 | 48.3 / **3.5** / 77.0 | 3.5 [0.9, 7.0] | **4/25 = 16% (p = .0009)** | **−0.48 (p = .0008)** |
| | (b) naive | same | 48.3 / **18.3** / 77.0 | 18.3 [11.3, 25.9] | **21/25 = 84% (p = .0009)** | **+0.48 (p = .0008)** |
| | (c) none | same | **26.7** / 3.5 / **5.4** | 3.5 | 4/25 | −0.48 |

Image-right on mirrored images, by the layout seen: 37 / 73 / 88% (4-bit) and 39 / 79 / 86% (bf16).

**"Inverts", precisely.**
- Naive scoring exchanges the opposite-scene splits that go the right way with those that go the wrong way (k/n → (n − k)/n), and it flips the sign of the signed effect.
- It leaves same-sign rates and image-right shares untouched, and changes no same-side score.
- The opposite-scene both-correct rate becomes the both-wrong rate.
- Keeping the original targets (c) instead breaks only the same-side scores.
- The 14.6% printed by the original pipeline is the naive (b) value at the nonzero threshold; the correct value is 9.9%.

**Source.**
- `btp_evidence.py::claim6_mirror` → `evidence.json: claim6_mirror`.
- Also `outputs/reanalysis/results.json: headline.mirror_fixed` (correct scoring, labelled by original layout) and NB06 cell 18 (naive).

**Status: VERIFIED with the definition above.** The bf16 replay makes the point most strongly.

**Recommended wording.**
- "After a horizontal flip, 'the mug on the left' names the other twin. Scoring a mirrored scene by negating each instruction's target without swapping them exchanges the opposite-scene splits that go the right way with those that go the wrong way."
- "In the bf16 replay, 4 of 25 splits (16%, p = .0009) go the right way under correct scoring and 21 of 25 (84%) under naive scoring. The signed effect flips from −0.48 to +0.48 bins, and opposite-scene both-correct rises from 3.5% to 18.3%."
- "A significantly reversed result is scored as significant grounding."

---

## 7. Stimulus screening and the gripper fallback

**Claim as worded.** Plan v2: "Approval differs by layout (54 vs 46%, p < .001); 45.5% of scenes use the gripper fallback."

**Numbers.**
- **Blind screening:** decisions on 1,644 of 3,248 composites; 1,604 were never screened.

  | Layout | Approved | Rate [Wilson CI] |
  |---|---|---|
  | Both-left | 210/437 | 48.1% [43.4, 52.7] |
  | Opposite | 437/808 | **54.1%** [50.6, 57.5] |
  | Both-right | 173/399 | 43.4% [38.6, 48.3] |

  - Opposite vs pooled same-side: 54.1% vs **45.8%** (383/836), χ² = 11.2, **p = .0008** (Fisher p = .0009).
  - Three-layout χ² p = .0014.
- **Rejections:** 824 in total: 781 implausible paste, 26 wrong object, 17 no second instance.
- **Frozen set:** 400 scenes. 60 were hand-labelled "unclear" and dropped, leaving 340 analysed.
- **Gripper fallback (image centre):**
  - **182 of the 400 frozen scenes (45.5%)**, and 141 of the 340 analysed (41.5%).
  - By layout: both-left 31/93 (33%), opposite 54/151 (36%), both-right **56/96 (58%)**; χ² p = .0004.
  - 41 of the 60 dropped scenes used the fallback.
  - Hand relabelling changed 53 of the 340 scenes.
- **Headline on detected-gripper scenes only (199 scenes):**
  - 4-bit logs: opposite both-correct **4.3%** (3/70) [0.0, 10.0]; same-sign 85.7% [77.1, 92.9]; image-right 43 / 73 / 99%; paired contrast +4.0 (p = .73); splits 3/10.
  - bf16: 4.5% [0.0, 10.6], 83.3%, 50 / 73 / 100%, +6.1 (p = .38), 3/11, signed median −0.37 (p = .002).
- **Not-relabelled scenes (287):**
  - 4-bit: 3.0% [0, 6.9], 85.1%, 43 / 72 / 95%.
  - bf16: 4.2%, 82.1%, 48 / 71 / 100%; the contrast is +11.9 (p = .039) while splits go 4/17 (p = .049) the wrong way.

**Source.**
- `btp_evidence.py::claim7_screening` → `evidence.json: claim7_screening.{screening, gripper_fallback, headline_subsets}`.
- Also `outputs/reanalysis/results.json: screening, subsets`.

**Status.**
- Rates and p: **VERIFIED**.
- "45.5% of scenes": **NEEDS REWORDING.** It is 45.5% of the 400 frozen scenes but 41.5% of the 340 analysed, and the fallback is unevenly spread across layouts.
- The headline holds on detected-gripper scenes: **VERIFIED**.

**Recommended wording.**
- "Blind screening approved 54.1% of opposite-layout composites (437/808) but 45.8% of same-side ones (383/836; χ² p = .0008). 781 of the 824 rejections were for an implausible paste."
- "The gripper could not be detected in 141 of the 340 analysed scenes (41.5%), which use the image centre instead. This happened more often for both-right scenes (58%) than for the others (33–36%)."
- "On the 199 scenes with a detected gripper, the headline is unchanged: 4.3% (3/70) of opposite scenes have both instructions correct."

---

## 8. The decisive statistic

**Claim as worded.**
- Plan v2: "A sign-agreement contrast scores backwards behaviour as grounding."
- done.md: table_side (4-bit, original images): "+14.8 points, p = .004, even though 75% of its splits go to the wrong twin".

**Numbers.** Paired contrast = same-side minus opposite same-sign rate. "Disc." is the number of discordant pairs. GPU runs use cf1 (the c1 values are identical in these rows unless noted).

| Run, wording | Paired contrast (McNemar p; pairs, disc.) | Opposite splits right way (binomial p) | Signed median (Wilcoxon p) |
|---|---|---|---|
| **4-bit, table_side** | **+14.8 (p = .0042; 81, 16)** | **7/28 = 25.0% (p = .013)** | −0.53 (p = 4×10⁻⁷) |
| **bf16, original wording** | **+10.7 (p = .039; 75, 12)** | 7/21 = 33.3% (p = .19) | **−0.15 (p = .011)** |
| **bf16, table_side** | **+9.6 (p = .039; 83, 12)** | **3/24 = 12.5% (p = .0003)** | −0.48 (p = 1.1×10⁻⁵); c1: −0.51 (4.8×10⁻⁶) |
| bf16, not-relabelled scenes, original wording | +11.9 (p = .039) | 4/17 = 23.5% (p = .049) | −0.23 (p = .002) |
| for contrast: 4-bit, prenominal | +17.1 (p = .0002; c1) / +18.2 (p = .0001; cf1) | 22/28 = 78.6% (p = .004) | +0.74 (p = 1×10⁻⁵) |

Of the 24 wording × run cells, three have a significant positive contrast with backwards splits: 4-bit table_side, bf16 original and bf16 table_side.

**Source.**
- `btp_evidence.py::claim8_decisive` → `evidence.json: claim8_decisive.{c1_as_reported, cf1_convention}`.
- Also `outputs/runs/results.json: wording.nf4_original.rows.table_side`.

**Status: VERIFIED.** The bf16 rows make the point more strongly: there the original wording itself passes the sign-agreement test while its effect is reversed.

**Recommended wording.**
- "The same-side vs opposite contrast of sign agreement reads only whether two actions share a sign, so it cannot tell correct separation from backwards separation."
- "With '… on the left side of the table' it reports +14.8 points (McNemar p = .004), although only 7 of 28 opposite-scene splits (25%, p = .013) go to the named twins."
- "In bf16, the original wording passes the same test (+10.7, p = .039) while 7 of 21 splits go the right way and its signed effect is reversed (−0.15 bins, p = .011)."
- "Report the split direction and the signed effect with it."
- **Revision round:**
  - Neither bf16 contrast (+10.7 and +9.6, both p = .039) survives any Holm family (§11.13). Cite them as raw p values that illustrate the statistic.
  - A placebo pair naming the same twin gives +11.7 (p = .012) in bf16 (§11.10).

---

## 9. Wording: direction of the word's effect across four runs

**Claim as worded.**
- done.md and plan v2: splits going the right way, 4-bit / 4-bit mirror / bf16 / bf16 mirror: prenominal 79 / 68 / 77 / 76%; table_side 25 / 29 / 12.5 / 32%; original 37 / 32 / 33 / 16%.
- Signed effects: original "≈0, then weakly reversed (p ≤ .03)"; prenominal and object "significant in 3 of 4"; table_side "reversed in all 4 (p ≤ 6×10⁻⁶)".

**Numbers.** cf1 (the stated convention). Where c1, the reported value, differs, it is given in brackets.
- **Holm-run:** Holm correction within each run over its 6 wordings.
- **Global:** Holm correction over all 24 cells.

| Wording | Splits right way (4-bit / 4-bit mirror / bf16 / bf16 mirror) | Splits sig. raw / Holm-run | Signed median, bins | Signed p, raw | Signed sig. raw / Holm-run / global |
|---|---|---|---|---|---|
| prenominal "the left {noun}" | 22/28 = 78.6 / 23/34 = 67.6 / 20/26 = 76.9 / 20/26 = 76.9% [c1: 19/25 = 76.0] | 3 / 3 [c1: 3 / 2] | +0.74 / +0.02 / +0.35 / +0.20 | 1×10⁻⁵ / .58 / .0008 / .006 | 3 / 3 / 2 |
| object "the object on the left" | 28/35 = 80.0 / 18/30 = 60.0 / 23/31 = 74.2 / 16/24 = 66.7% [c1: 26/33, 18/30, 21/28, 17/25] | 2 / 2 [c1: 2 / 1] | +0.80 / +0.21 / +0.25 / +0.29 [c1: +0.77 / +0.21 / +0.22 / +0.08] | .0003 / .050 / .010 / .001 | 3 / 3 / 2 [c1: 3 / 2 / 2] |
| table_side "… left side of the table" | 7/28 = 25.0 / 7/24 = 29.2 / 3/24 = 12.5 / 9/28 = 32.1% | 2 / 1 | −0.53 / −0.81 / −0.48 / −0.80 | 4×10⁻⁷ / 6×10⁻⁷ / 1.1×10⁻⁵ / 1×10⁻⁶ [c1 bf16: 4.8×10⁻⁶] | **4 / 4 / 4** (global Holm p ≤ 2.2×10⁻⁴) |
| original "… on the left" | 7/19 = 36.8 / 7/22 = 31.8 / 7/21 = 33.3 / 4/25 = 16.0% | 1 / 1 | −0.03 / −0.13 / −0.15 / −0.48 | .22 / .026 / .011 / .0008 | 3 / 2 / 1 |
| absent_noun | 47.6 / 5.6 / 38.1 / 33.3% | 1 / 1 | −0.37 / −0.49 / −0.09 / −0.32 | all < .05 | 4 / 3 / 3 |
| move "move left/right" | 50.0 / 67.4 / 58.6 / 50.0% [c1: 48.3, 54.8 for runs 1 and 3] | 1 / 0 | −0.08 / +0.77 / +0.17 / +0.15 | .90 / .0001 / .15 / .18 | 1 / 1 / 1 |

- **Direction:** consistent across all four runs for every named-twin wording, on both the split and the signed statistic. prenominal and object always point toward the named twin; table_side and the original wording always point away. `move` is inconsistent.
- **Opposite both-correct:**
  - prenominal 19.5 / 17.7 / 18.7 / 15.9%;
  - object 24.3 / 14.4 / 20.4 / 13.1%;
  - table_side 6.2 / 6.1 / 2.7 / 7.7%;
  - original 6.0 / 6.2 / 6.2 / 3.5%.
- **Paired-contrast Holm** (the report's family, c1 as reported): 4-bit rejects prenominal, table_side and object (cf1: prenominal and table_side); 4-bit mirror rejects prenominal; bf16 and bf16 mirror reject nothing.

**Source.**
- `btp_evidence.py::wording_grid, claim9_wording` → `evidence.json: claim9_wording.{cf1_convention, c1_as_reported}.by_wording` and `wording_grid`.
- Also `outputs/runs/results.json: wording.{nf4,bf16}_{original,mirror}.rows`.

**Status.**
- Every split percentage in the claim: **VERIFIED.** They are c1; with cf1 only the bf16-mirror prenominal changes (76.0 → 76.9%).
- "Significant in 3 of 4" (raw) for prenominal and object: **VERIFIED.**
- "p ≤ 6×10⁻⁶" for table_side: **VERIFIED for c1 only.** With cf1 it is p ≤ 1.1×10⁻⁵.
- The summary needs Holm qualifiers, so **NEEDS REWORDING.**

**Recommended wording.**
- "In all four runs (4-bit and bf16, original and mirrored images), each wording's effect has the same direction. 'the left {noun}' and 'the object on the left' move the action toward the named twin (60–80% of opposite-scene splits; positive signed effect in 4 of 4 runs). '… on the left side of the table' and '… on the left' move it away (12.5–32% and 16–37%; negative in 4 of 4)."
- "After Holm correction within each run, the signed effect is significant in 4 of 4 runs for 'left side of the table', 3 of 4 for 'the left {noun}' and 'the object on the left', and 2 of 4 for the original wording."
- "A sign-agreement analysis would detect none of these effects in bf16 (no Holm-surviving contrast)."
- If c1 must be kept, say "object: 2 of 4".
- **Revision round (§11.9):** these counts use scene-level Wilcoxon p. With a sign-flip test by base frame (149 of 191 frames hold two scenes) they are 4 / 3 / 2 / 1 for table side / prenominal / object / original. Prefer the frame-level counts.

---

## 10. The worked example's finding

**Claim as worded** (done.md, plan v2): headline both-correct 5.1% in opposite scenes, same-sign rates, image-right 38 / 71 / 94%; paste displacement +7.6 bins, p = 2×10⁻⁶, 70.5%; variance shares (word Shapley R² 0.000); paraphrase floor; Bridge role statistics.

**Numbers.**
- **Headline (4-bit logs, ≥1 bin, bootstrap CIs):**

  | Layout | n | Same sign | Both correct | Image-right |
  |---|---|---|---|---|
  | Both-left | 56 | 83.9% [73.2, 92.9] | 53.6% [40.0, 66.1] | 38.4% [27.3, 50.0] |
  | Opposite | 117 | 83.8% [76.7, 90.5] | **5.1%** [1.7, 9.3] (6/117) | 71.4% [63.9, 78.6] |
  | Both-right | 86 | 93.0% [87.4, 97.7] | 90.7% [84.3, 96.4] | 94.2% [89.9, 97.7] |

  - In bf16 (cf1): both-left 89.8 / 50.8 / 44.1; opposite 81.2 / **6.2** [1.8, 10.8] / 71.0; both-right 98.9 / 98.9 / 99.4.
- **Within-frame paste displacement** (149 base frames, term-free instruction):
  - median **+1.47** bins; mean **+7.59** [3.51, 12.09];
  - 70.5% shifted toward the paste; Wilcoxon p = 1.9×10⁻⁶; rank-biserial +0.45.
- **Variance shares:** Shapley (LMG) decomposition of the OLS R², with cluster-robust SEs by base frame. The mixed model supplies coefficients only.

  | Fit | n | Layout | Demo direction | Word | Named-twin offset | Total R² |
  |---|---|---|---|---|---|---|
  | All | 2,040 | 0.053 | 0.068 | **0.0004** | 0.0008 | 0.128 |
  | Detected gripper | 1,194 | 0.081 | 0.029 | 0.0002 | 0.0004 | 0.119 |
  | Image-centred layout, detected | 1,194 | 0.028 | | | | 0.067 |

  - Word OLS coefficient: +0.02 bins (p = .99).
  - The mixed model (random intercept per base frame) **does not converge** on all 2,040 predictions (statsmodels ConvergenceWarning, |grad| = 97). It converges for the two detected-gripper fits (`evidence.json: claim10_example.variance_shares.*.mixed_model_converged`).
  - Do not cite the all-predictions "mixed coef" column of `outputs/reanalysis/report.md`. The Shapley shares come from OLS and are unaffected.
- **Paraphrase floor** (left↔right |Δ| vs grab↔take |Δ|, n = 340):
  - 4-bit: 2.57 vs 1.72 bins; left↔right larger in 58.5%; Wilcoxon p = .00035.
  - bf16: 2.46 vs 1.37; 58.8%; p = .002.
- **Bridge language** (17,035 harvested instructions):
  - **3,574 (21.0%)** contain left/right.
  - By rule: 3,333 destination or direction; 183 starting place; 2 match the referent rule (both on inspection a destination or location); **56 unclassified.**
  - Of the 56 unclassified (33 distinct strings), most use "right" as an intensifier ("right on top of the napkin"). A few are locations or orientations ("pepper is back right table", "metal part facing left"). One, "right pepper shaker", could be a prenominal referent.
- **Role statistics** (single lateral word, motion > 5 mm; first motion image-left):

  | Role | After "left" | After "right" | Fisher p |
  |---|---|---|---|
  | Starting place | 81.8% (n = 22) | 8.3% (n = 24) | 5.4×10⁻⁷ |
  | Destination | 39.3% (n = 1,693) | 58.8% (n = 1,463) | 6.1×10⁻²⁸ |

**Source.**
- `btp_evidence.py::claim10_example` → `evidence.json: claim10_example.*`.
- Equal to `outputs/reanalysis/results.json: headline.one_bin, paste_displacement.neutral, drivers_*, language_audit, role_association` and `outputs/runs/results.json: paraphrase_floor`.

**Status.** The headline, displacement, paraphrase floor and role statistics are **VERIFIED**. Three phrasings **NEED REWORDING**:
- done.md's "Shapley R², mixed model grouped by base frame" should be "Shapley decomposition of OLS R² (SEs clustered by base frame)".
- The displacement should be reported with its median (+1.5 bins), not the mean (+7.6) alone, because the mean is driven by a tail.
- "56 use 'right' to mean 'directly'" should be "56 are unclassified, mostly 'right' as an intensifier".
- Note also that the full model explains only 13% of the variance.

**Recommended wording.**
- "With the original wording (4-bit), both instructions move the same way in 83.8% of opposite scenes and both are correct in 5.1% (6/117). The action follows the layout: image-right in 38% / 71% / 94% of both-left / opposite / both-right scenes."
- "Moving only the pasted twin shifts the term-free action toward it (median +1.5 bins, 70.5% of 149 frames, p = 2×10⁻⁶)."
- "The word explains 0.04% of the variance in the lateral action (layout 5%, the demonstrated direction 7%; total R² 0.13)."

---

## Table 1: one-at-a-time sensitivity

The table is generated as `one_at_a_time.csv` / `one_at_a_time.md`.

- **Reference analysis:** dy channel; expected value with token 31744 folded (`cf1`); bf16 on GH200; original wording; original images; all 340 scenes; correct mirror scoring; split direction and signed effect reported next to the sign-agreement contrast.
- **One bin** is the grid step (q99 − q01)/255 = 0.0003238 since the revision round (§11.11). Before that it was the /254 width 0.000325, which is now row `w254`. thr2 is 2 × 0.0003238. dx's floor is its own /255 step, 0.000224.
- **Each row changes exactly one choice.** Exception: `nf4log` changes both GPU and readout, because the logs have no `cf`.
- **Rates:** over scenes where both actions are decided. CIs are base-frame cluster bootstraps with 2,000 draws.
- **Layouts:** as the model saw them.
- **gpu row (added in §11.1):** a two-run comparison at fixed NF4 precision and fixed readout `c1`, which cannot be expressed against the bf16 reference. It shows A100 values, then GH200 values (CSV columns `cmp_*`). Its A100 values equal row `nf4log`.
- **dx row:** "image-right" means negative dx (the dy convention applied to dx), and the signed median is in dx bins.
- **r:** rank-biserial, shown when the median is a tie.

| Row | Choice changed (one at a time) | Opp. n | Opp. both correct [95% CI] | Opp. same sign [95% CI] | Image-right SSL / opp / SSR | Same-side − opp. sign-agreement, pts (McNemar p) | Opp. splits the right way (binomial p) | Signed left−right median, bins (Wilcoxon p) |
|---|---|---|---|---|---|---|---|---|
| ref | **Reference:** dy, expected value (31744 folded), bf16 GH200, original wording, original images, all 340 scenes, ≥1 bin (grid step 0.0003238) | 112 | 6.2 [1.8, 10.8] | 81.2 [73.6, 88.4] | 44 / 71 / 99 | +10.7 (0.039) | 7/21 = 33% (0.189) | -0.15 (0.011) |
| dx | **Action channel:** dx (component 0) instead of dy; dx's own one-bin floor (q99-q01)/255 = 0.000224 | 142 | 4.2 [1.4, 7.7] | 91.5 [86.6, 95.8] | 82 / 82 / 78 | -1.8 (0.774) | 6/12 = 50% (1.000) | +0.09 (0.051) |
| sign | **Sign convention:** image-right = positive dy (flipped) | 112 | 12.5 [7.0, 18.9] | 81.2 [73.6, 88.4] | 56 / 29 / 1 | +10.7 (0.039) | 14/21 = 67% (0.189) | +0.15 (0.011) |
| zero | **Zero point:** dy measured from normalised zero (bin 127 = -0.000424) instead of physical zero | 112 | 6.2 [1.8, 10.8] | 81.2 [73.9, 88.3] | 24 / 57 / 99 | +12.0 (0.012) | 7/21 = 33% (0.189) | -0.15 (0.011) |
| argmax | **Readout:** argmax token instead of expected value | 108 | 8.3 [3.6, 13.9] | 81.5 [74.1, 88.8] | 45 / 72 / 98 | +6.9 (0.227) | 9/20 = 45% (0.824) | +0.00 (0.038); r = -0.16, 30% ties |
| thr0 | **Decision threshold:** any nonzero action counts as decided (0 bins) | 151 | 10.6 [5.9, 16.0] | 76.8 [70.0, 83.6] | 49 / 68 / 97 | +1.6 (0.864) | 16/35 = 46% (0.736) | -0.15 (0.011) |
| thr2 | **Decision threshold:** 2 bins (0.000648) | 90 | 6.7 [2.2, 12.2] | 81.1 [73.0, 88.9] | 37 / 73 / 99 | +14.0 (0.008) | 6/17 = 35% (0.332) | -0.15 (0.011) |
| w254 | **Bin width:** one bin taken as (q99-q01)/254 = 0.000325 instead of the grid step /255 = 0.0003238 | 112 | 6.2 [1.8, 10.8] | 81.2 [73.6, 88.4] | 44 / 71 / 99 | +10.7 (0.039) | 7/21 = 33% (0.189) | -0.15 (0.011) |
| tok | **Token map:** token 31744 left out (renormalise over 31745-31999) | 112 | 6.2 [1.8, 10.8] | 81.2 [73.6, 88.4] | 44 / 71 / 99 | +10.7 (0.039) | 7/21 = 33% (0.189) | -0.16 (0.010) |
| nf4 | **Precision:** 4-bit NF4 instead of bf16 (same GH200) | 116 | 6.0 [1.8, 10.4] | 83.6 [76.9, 90.4] | 39 / 70 / 94 | +4.1 (0.549) | 7/19 = 37% (0.359) | -0.03 (0.215) |
| nf4log | **Precision + GPU:** 4-bit NF4 logged on Colab A100 (token fold unavailable: c1) | 117 | 5.1 [1.7, 9.3] | 83.8 [76.7, 90.5] | 38 / 71 / 94 | +4.1 (0.549) | 6/19 = 32% (0.167) | -0.02 (0.485) |
| gpu | **GPU:** Colab A100 instead of GH200, at fixed NF4 precision and the 255-token readout (c1); comparator is GH200 NF4 c1, not the reference | A100 117; GH200 115 | A100 5.1 [1.7, 9.3]; GH200 6.1 [1.8, 11.0] | A100 83.8 [76.7, 90.5]; GH200 83.5 [76.1, 90.4] | A100 38 / 71 / 94; GH200 39 / 71 / 94 | A100 +4.1 (0.549); GH200 +4.1 (0.549) | A100 6/19 = 32% (0.167); GH200 7/19 = 37% (0.359) | A100 -0.02 (0.485); GH200 -0.03 (0.240) |
| mir | **Images (replication):** mirrored images, targets swapped and negated (correct scoring) | 115 | 3.5 [0.9, 7.0] | 78.3 [70.2, 86.0] | 39 / 79 / 86 | +1.3 (1.000) | 4/25 = 16% (0.00091) | -0.49 (0.00077) |
| mir_naive | **Mirror scoring:** mirrored images, each role's target negated without swapping (NB06) | 115 | 18.3 [11.3, 25.9] | 78.3 [70.2, 86.0] | 39 / 79 / 86 | +1.3 (1.000) | 21/25 = 84% (0.00091) | +0.49 (0.00077) |
| mir_none | **Mirror scoring:** mirrored images, original targets kept | 115 | 3.5 [0.9, 7.0] | 78.3 [70.2, 86.0] | 39 / 79 / 86 | +1.3 (1.000) | 4/25 = 16% (0.00091) | -0.49 (0.00077) |
| w_prenominal | **Wording:** "pick up the left {noun}" | 107 | 18.7 [12.0, 26.4] | 75.7 [67.6, 83.2] | 34 / 69 / 99 | +15.7 (0.013) | 20/26 = 77% (0.009) | +0.35 (0.00075) |
| w_object | **Wording:** "pick up the object on the left" | 113 | 20.4 [12.6, 27.7] | 72.6 [64.4, 80.9] | 48 / 62 / 82 | +3.7 (0.607) | 23/31 = 74% (0.011) | +0.25 (0.010) |
| w_table_side | **Wording:** "pick up the {noun} on the left side of the table" | 113 | 2.7 [0.0, 6.1] | 78.8 [70.8, 85.7] | 49 / 73 / 95 | +9.6 (0.039) | 3/24 = 12% (0.00028) | -0.48 (1.1e-05) |
| w_absent_noun | **Wording:** "pick up the {other noun} on the left" (extra) | 101 | 7.9 [3.0, 13.7] | 79.2 [70.9, 87.1] | 60 / 66 / 83 | +6.8 (0.481) | 8/21 = 38% (0.383) | -0.09 (0.041) |
| w_move | **Wording:** "move left" / "move right" (extra; no referent) | 93 | 18.3 [10.6, 26.4] | 68.8 [59.1, 78.3] | 57 / 66 / 72 | -3.2 (0.832) | 17/29 = 59% (0.458) | +0.17 (0.151) |
| sub_det | **Scene subset:** detected-gripper scenes only (199 of 340) | 66 | 4.5 [0.0, 10.6] | 83.3 [74.2, 92.4] | 50 / 73 / 100 | +6.1 (0.375) | 3/11 = 27% (0.227) | -0.37 (0.002) |
| sub_notrel | **Scene subset:** scenes not relabelled by hand (287 of 340) | 95 | 4.2 [1.1, 8.4] | 82.1 [73.7, 89.5] | 48 / 71 / 100 | +11.9 (0.039) | 4/17 = 24% (0.049) | -0.23 (0.002) |
| gpu_prefix | **Autoregressive prefix:** dy read with a common (teacher-forced) dx token, or marginalised over dx | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU |
| gpu_screen | **Stimulus screening:** include the 824 rejected (and 1,604 unscreened) composites | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU |
| gpu_dist | **Readout:** median or mode of the full 256-way dy distribution | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU | NEEDS GPU |

The CSV also gives the same-side both-correct rates, the McNemar pair and discordant counts, the rank-biserial and the tie share.

**Which choices move which conclusion.** Tags follow the call: direction, strength, scope.

**"Neither grounded."** Reference: opposite both-correct 6.2%.
- At most 12.5% in every row except:
  - the wordings Bridge never uses (prenominal 18.7%, object 20.4%; strength);
  - the referent-free "move left/right" (18.3%);
  - naive mirror scoring (18.3%, an artefact; a pitfall).
- The nonzero threshold raises it to 10.6%.
- The flipped sign gives 12.5%, which is the both-wrong rate.
- **Robust:** no row exceeds 20.4%.

**"Not lexical."** Reference: opposite same-sign 81.2%.
- 69–92% in every row.
- Reading dx inflates it (91.5%) because dx barely changes sign.
- **Robust.**

**"Follows the layout, leans image-right."** Reference: image-right 44 / 71 / 99%.
- dx flattens the gradient to 82 / 82 / 78% (scope: the layout effect disappears).
- The flipped sign reverses it to 56 / 29 / 1% (direction: a leftward lean).
- Normalised zero shrinks the opposite-scene lean to 57% (strength; 56% with the /254 width, §11.11).
- Every other row keeps the gradient (opposite 62–79%).

**Sign-agreement contrast (direction-blind).** Reference: +10.7 points, p = .039. This is the most fragile statistic; its significance turns on and off with:
- threshold: 0 bins p = .86; 2 bins p = .008;
- readout: argmax p = .23;
- precision: 4-bit p = .55;
- images: mirror p = 1.0;
- subset: detected gripper p = .38; not relabelled p = .039;
- zero point: p = .012;
- wording: prenominal p = .013; object p = .61; table_side p = .039.

It is never informative about direction (strength).

**Split direction.** Reference: 7/21, not significant.
- Reversed by the flipped sign (14/21) and by naive mirror scoring (21/25 = 84%, p = .0009); direction.
- Significantly backwards on mirrored images (4/25), for table_side (3/24) and on not-relabelled scenes (4/17).
- Significantly correct for prenominal (20/26) and object (23/31); direction, by wording.

**GPU (row `gpu`, §11.1).** At fixed NF4 precision and readout, A100 vs GH200 changes 8.1% of lateral tokens but no verdict. Every rate moves by at most 1.8 points, and every p stays on the same side of .05.

**Signed effect.** Reference: −0.15 bins, p = .011.
- Flips sign under the flipped sign convention (+0.15) and naive mirror scoring (+0.48); direction.
- Loses significance at 4-bit (−0.03, p = .22) and on dx (+0.09, p = .051); strength.
- Under argmax the median is 0 with 30% ties, but still reversed (r = −0.16, p = .038).
- Set by the wording: +0.35, +0.25, −0.48; direction.
- **No effect:** token map, bin width (/254 vs /255), threshold.

**Pitfalls** (conventions with a right answer): channel, sign, zero point, token map, bin width, mirror scoring, direction-blind statistic.
- The channel, sign and mirror scoring change the direction of conclusions.
- The token map and bin width change nothing material here.

**Forks** (legitimate choices): readout, threshold, precision, wording, subsets.
- Readout, threshold, precision and subsets change the strength (significance) of secondary statistics but not the headline.
- Wording changes the direction of the word effect.
- **Prefix and screening: NEEDS GPU.**

---

## Inventory for planning further experiments

**(a) What the GPU CSVs store.**
- No full distributions for any dimension. Per dimension (0–6) they store:
  - `a` (argmax action) and `b` (argmax bin);
  - `c` (expected value, 255 tokens, 31744 omitted) and `cf` (expected value, 31744 folded);
  - `p` (top-bin probability) and `h` (entropy), both within the 255-token slice, so also token-map-affected;
  - `m` (mass on all 256 action tokens);
  - plus `action_mass_min`.
- The 4-bit logs lack `cf` and `m`.
- `scripts/run_gpu.py:predict` computes the 7 × 256 probabilities (l. 103) and discards them.
- Dimensions 1–6 are conditioned on the greedy prefix.

**(b) Batching and padding.**
- **No batching.** Each prediction tokenises one prompt, drops the attention mask (l. 81), appends the empty token, and calls `vla.generate` with batch size 1 (l. 88–89). There is no padding, so the padding side does not arise.
- **Sharding:** items are spread across GPUs by index modulo `nshards` (l. 134), and each GPU still runs batch 1.
- **Determinism:** 20 duplicate stimuli that ran on different GH200s gave bit-identical outputs (`evidence.json: inventory.cross_gpu_duplicates`).
- **If batching is added, outputs could change:**
  1. OpenVLA's remote code splices the image-patch embeddings in after the first (BOS) token, so left padding would misplace them unless handled. This is from the public `modeling_prismatic.py`; check the copy in `$HF_HOME/modules/transformers_modules` on the GPU machine.
  2. bitsandbytes 4-bit takes a different kernel path for single-row and multi-row inputs, so batch composition can change the logits at float level.
  3. Near-tie argmax tokens flip easily. A GPU change alone flipped 8.1% of lateral tokens; dx is often near-tied (median top probability 0.69, 27% below 0.5).
- **Any batched mode must be gated against batch-1 outputs.**

**(c) Images of screened-out scenes.**
- **Yes.** `google_drive/v2/constructed/frames/` holds a PNG for all 3,248 composites: 820 approved, 824 rejected and 1,604 never screened, plus the 400 frozen.
- `constructed_manifest.csv` has `instr_a/instr_b`, the recorded arrangement, the instance and gripper x, the gripper source and the recorded target signs.
- Hand labels (`human_configuration`) exist only for the 400 frozen scenes, so screened-out scenes would be scored by their recorded arrangement. 1,274 of 3,248 composites use the gripper fallback.
- A run needs a new subcommand that iterates the manifest instead of `probe_predictions.csv`. The 824 rejected scenes × 2 roles is about 1,650 predictions, roughly 5 minutes on 4 GH200s at the replay's throughput.

**(d) Run modes, and what a prefix-control mode needs.**
- **Current subcommands:**
  - `smoke`: load, action-space report, readout gate;
  - `replay`, with `--conditions` and `--limit`;
  - `ladder`, with `--mirror`;
  - `natural`, with `--mirror`.
- **Shared options:** `--precision {nf4,bf16}`, `--out`, `--shard/--nshards` and `--n`. Runs are resumable.
- **Prefix control:** add a `--prefix {greedy,force,marginal}` option to replay and ladder, or a `prefix` subcommand, that replaces `generate` with a manual two-step decode:
  1. **Prompt pass:** one forward pass over the prompt (with the empty token appended), `use_cache=True`, to get the dx logits and the KV cache.
  2. **Force:** feed a chosen dx token t* and read the dy logits. Two natural choices: the greedy dx token of the term-free instruction on the same image, or for each left/right pair, the same token for both members, which removes the prefix difference behind claim 5.
  3. **Marginal:** take the dx tokens that cover ≥ 99% of p(dx). The dx perplexity has median 2.6 and p90 about 7, so expect roughly 10–30 tokens. Run them as one batched step on the cache expanded to K rows, and mix: p(dy) = Σ p(dx = k) p(dy ∣ dx = k) / Σ p(dx = k).
  4. **Log** the forced and marginal expected values (for example `ct1`, `cm1`), K and the covered mass.
  5. **Gate:** with t* equal to the greedy token, the forced path must reproduce `c1/cf1` from `generate`, as `verify_readout` does. Validate the K-batched step against sequential single-token steps, because 4-bit kernels depend on batch shape.
  - Cost: about one replay-sized job, 5–10 minutes on 4 GH200s.
- **Full-distribution logging:** a `--save-dist` flag that appends the 7 × 256 array `p` for each prediction to a per-shard sidecar file, in CSV-row order with a `dist_row` index column. In float16 that is 3.5 KB per prediction, 29 MB for the full 8,160-row ladder. This enables the `gpu_dist` row (median or mode readouts) without re-running.

---

## 11. Addenda (2 Oct, revision round)

**Regenerate.**
1. `.venv/bin/python docs/btp_paper/evidence/btp_evidence.py` writes Table 1, which now includes row `gpu`.
2. `.venv/bin/python docs/btp_paper/evidence/btp_evidence_addenda.py` takes 10 s, writes `addenda.json` and prints the §11.7 tables.

Both use existing outputs only (no GPU), and all definitions are as in the sections above.

### 11.1 GPU at fixed precision and readout (Table 1 row `gpu`)

**Request.** Isolate the GPU. Compare 4-bit NF4 logged on the Colab A100 with 4-bit NF4 replayed on the GH200, both read with the 255-token readout `c1`, because the logs have no `cf1`.

**Numbers.** Original wording, original images, ≥1 bin. CIs are base-frame bootstraps with 2,000 draws.

| Statistic | A100 (Colab log), NF4, c1 | GH200 (replay), NF4, c1 |
|---|---|---|
| Opposite scenes, both decided | 117 | 115 |
| Opposite both-correct | 5.1% [1.7, 9.3] (6/117) | 6.1% [1.8, 11.0] (7/115) |
| Opposite same-sign | 83.8% [76.7, 90.5] | 83.5% [76.1, 90.4] |
| Same-sign, both-left / both-right | 83.9 / 93.0% | 85.7 / 92.9% |
| Image-right, both-left / opposite / both-right | 38.4 / 71.4 / 94.2% | 39.3 / 70.9 / 94.1% |
| Sign-agreement contrast (McNemar p; pairs, discordant) | +4.1 points (p = .55; 74, 11) | +4.1 points (p = .55; 73, 11) |
| Opposite splits the right way (binomial p) | 6/19 = 31.6% (p = .17) | 7/19 = 36.8% (p = .36) |
| Signed median (Wilcoxon p) | −0.02 bins (p = .48) | −0.03 bins (p = .24) |

**Per-prediction agreement** (identical stimuli).
- **All 2,720 predictions:**
  - the lateral argmax token changed in **8.1%** (221): one step in 1.2%, two or more steps in 6.9%;
  - correlation of `c1`: r = **.968** (Spearman .972); median |Δc1| 0.09 bins;
  - sign agreement **99.3%** where both runs are decided (n = 2,236).
- **The 680 baseline predictions behind the table:** token changed in 6.2% (42); r = .960; sign agreement 99.3% (n = 569).
- **Link to the prefix (claim 4):** the greedy dx token changed in 5.0% of predictions. Where it changed, dy's token changed in 84.7% of cases (median |Δdy| 6.0 bins); where it did not, in 4.1% (0.07 bins).

**What else differs between the two runs.**
- **Identical:**
  - torch 2.11.0+cu128, transformers 4.40.1 and bitsandbytes 0.50.2, from both runs' CSV metadata;
  - tokenizers 0.19.1, timm 0.9.10, huggingface_hub 0.23.4 and accelerate 0.30.1, printed by NB01/03/05 on Colab and pinned and installed on Isambard;
  - NF4 with double quantisation and bf16 compute; eager attention; seed 42;
  - greedy decoding at batch size 1, with the attention mask dropped and the empty token appended;
  - `bridge_orig` q01/q99, equal to 5 d.p.
- **Different (recorded):**
  - GPU: A100-SXM4-40GB (sm_80) vs GH200 120GB (sm_90);
  - CPU architecture: x86-64 vs aarch64 (Grace);
  - **Python 3.12.13 vs 3.12.14**;
  - run date: 30 Aug vs 1 Oct.
- **Not recorded on Colab, so possibly different:**
  - NVIDIA driver (GH200: 565.57.01);
  - numpy (Isambard 2.5.2), Pillow (12.3.0), torchvision (0.26.0+cu128), cuBLAS (12.8.4.1) and cuDNN (9.19.0.56);
  - the Hugging Face model revision, which is logged on neither machine.
  - Pillow and torchvision perform the image resize, so they are candidate causes alongside the GPU kernels. The logs cannot separate them.
- **Code path:** NB05's `model.predict_action_dist` vs `scripts/run_gpu.py:predict`. The decoding is the same; `run_gpu.py` also sets `allow_tf32 = False`, which is PyTorch's default.

**Verdict.** No verdict changes.
- Every p stays on the same side of .05: contrast .55 vs .55; splits .17 vs .36; signed .48 vs .24.
- Every rate moves by at most 1.8 points (both-left same-sign 83.9 → 85.7%).

**Source.**
- `btp_evidence_addenda.py::a1_gpu` → `addenda.json: A1_gpu_only`.
- Table 1 row `gpu` in `one_at_a_time.{csv,md}`, where the `cmp_*` columns hold the GH200 values.

**Status.**
- 8.1%, r = .968 and 99.3%: **VERIFIED**.
- "Library versions identical": **NEEDS REWORDING.** It is true for the logged model stack. The Python patch version differs, and numpy, Pillow, torchvision and the CUDA libraries are unknown on Colab.

**Recommended wording.** "Replaying the same 4-bit model on a different GPU (A100 → GH200; identical model-stack libraries; same readout) changes 8.1% of 2,720 lateral action tokens (r = .97) but no verdict: 5.1% vs 6.1% of opposite scenes have both instructions correct, and every test keeps its significance."

### 11.2 Physical units

**Numbers.**
- **dy's `bridge_orig` statistics from the model:** q01 = −0.04170349963009357 and q99 = +0.040855254605412394.
  - Source: `vla.get_action_stats("bridge_orig")` via `model.describe_action_space`, at full precision in `outputs/runs/logs/sgvla-smoke_6987042.out` (GH200).
  - The same values appear to 5 d.p. in the printed outputs of NB01 cell 14 and NB05 cell 10 (Colab).
  - The Hugging Face `config.json` `norm_stats` are not in the repo.
- **One bin of dy:** the grid step (q99 − q01)/255 is 3.2376 × 10⁻⁴ action units; the pipeline width /254 is 3.2503 × 10⁻⁴. If the units are metres, one bin is 0.324 or 0.325 mm per step, which is 1.6 mm/s at 5 Hz.
- **Control rate: 5 Hz.**
  - BridgeData V2: "the control frequency is 5 Hz" (`walke2023bridgedata`; `lit_substrate.md` A4.4).
  - OpenVLA: "the 5Hz non-blocking controller used in the BridgeData V2 tasks" (`kim2024openvla`; A1.8).
  - **Supported.**

**Evidence that the units are metres (indirect).**
- **What the action is:** in OpenVLA's `bridge_orig` transform, the action is the difference of consecutive proprioceptive end-effector states (`relabel_bridge_actions`, quoted in `lit_substrate.md` A2.10). It therefore has the unit of the state's position.
- **No stated unit:** none of our sources states that unit. BridgeData V2 says "continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose" (A4.7), and the robot code says "xyz deltas" (A4.9).
- **The magnitudes fit metres and rule out centimetres or millimetres:**
  - Per step, q01–q99 runs from −4.2 to +4.1 cm, at most 20 cm/s at 5 Hz.
  - In their first five steps, the demonstrations move a median 6.4 cm laterally (p90 15 cm, p99 24 cm; n = 17,035). This is the OXE "bridge" release's commanded `world_vector`, summed (`data.extract_episode`).
  - Episodes last a median 34 steps (IQR 25–39), which is 6.8 s at 5 Hz.
  - In centimetres, the whole per-step q01–q99 range would be ±0.4 mm, which is implausible for teleoperated tabletop pick-and-place.
- **Strength:** circumstantial but strong. The unit is inferred, not documented.

**Status.** q01/q99 and 5 Hz: **VERIFIED**. Metres: **supported indirectly; hedge it.**

**Recommended wording.** "One bin of the lateral action is 3.2 × 10⁻⁴ in Bridge's action units. That is about 0.3 mm per 0.2 s control step (5 Hz) if, as the magnitudes indicate, the units are metres."

### 11.3 Typical first-action magnitude

**Numbers.** Reference run: bf16 replay, folded readout `cf1`, baseline condition, n = 680 predictions.
- **Median |dy| = 4.29 bins (IQR 1.61–12.05).** This reproduces the coordinator's figure exactly.
  - 84.1% of the 680 predictions are decided (≥1 bin).
  - About 1.4 mm if the units are metres.
  - In /255 steps the median is 4.31. The 4-bit logs (`c1`) give 4.32 (1.68–12.17).
- **Effects as fractions of that median** (absolute values):

  | Quantity (bf16, cf1) | Bins | Fraction of median \|dy\| |
  |---|---|---|
  | Signed word effect, original wording (Table 1 `ref`) | −0.15 | 3.5% |
  | Signed word effect, prenominal (`w_prenominal`) | +0.35 | 8.1% |
  | Signed word effect, table_side (`w_table_side`) | −0.48 | 11.2% |
  | Positive control: within-frame paste displacement, median (claim 1, D) | +1.36 | 31.6% |

- **For scale (a different quantity; hedge it):** the demonstrations' early lateral motion is a median 6.4 cm over five steps, about 1.3 cm per step. The model's typical first lateral action (about 1.4 mm) is therefore roughly a tenth of a typical demonstrated step. The definitions differ: commanded vs achieved motion, and the mean of five steps vs the first step.

**Source.** `btp_evidence_addenda.py::a3_magnitude` → `addenda.json: A3_magnitude`.

**Status.** **VERIFIED.**

**Recommended wording.** "The typical first lateral action is 4.3 bins (median |dy|, IQR 1.6–12.0). The word's signed effect is 0.15–0.48 bins (4–11% of that), whereas moving only the pasted twin shifts the action by 1.4 bins (32%)."

### 11.4 Holm correction across Table 1

**Family.**
- 16 rows: ref, dx, sign, zero, tok, w254, mir_naive, argmax, thr0, thr2, nf4, mir, w_prenominal, w_object, w_table_side and sub_det.
- 3 p-values per row: the McNemar contrast, the binomial split test and the Wilcoxon signed test. That gives 48 tests.
- Holm step-down at α = .05: decisions from `reanalysis.holm`, adjusted p from `btp_evidence.holm_adjusted`.

| Family | Tests | Survivors (Holm-adjusted p) | First non-survivor |
|---|---|---|---|
| 16 rows | 48 | **7:** table_side signed (.0006); table_side splits (.013); prenominal signed (.034); mir signed and mir_naive signed (.034 each; corrected from .035 on 2 Oct, adjusted p = 0.03447); mir splits and mir_naive splits (.039 each) | sub_det signed, raw p = .0021 (adjusted .087) |
| without tok and w254 | 42 | the same 7 (adjusted .0005–.034) | sub_det signed (adjusted .074) |
| with gpu (A100 p = .55, .17, .48; the GH200 comparator's .55, .36, .24 would not survive either) | 51 | the same 7 (adjusted .0006–.042) | sub_det signed (adjusted .093) |
| distinct analyses only (drop sign, tok, w254, mir_naive) | 36 | 5: table_side signed and splits, prenominal signed, mir signed and splits | sub_det signed (adjusted .066) |

**Notes.**
- **The family contains near-duplicates.** sign and w254 reproduce ref's three p-values exactly, and tok differs only in the signed test (.010 vs .011). mir_naive reproduces mir's p-values exactly, because relabelling leaves two-sided p-values unchanged.
- The 7 survivors are therefore **5 distinct results**:
  - "… left side of the table" reversed, on both the split and the signed test;
  - "the left {noun}" forward, on the signed test;
  - the mirrored replication reversed, on both tests.
- **No sign-agreement contrast survives in any family.** The smallest are thr2 (.0078), zero (.012) and w_prenominal (.013).

**Source.** `btp_evidence_addenda.py::a4_holm` → `addenda.json: A4_holm_table1`.

**Status.** **VERIFIED.** The coordinator's 7 survivors are reproduced.

**Recommended wording.** "Across the 48 tests in Table 1 (16 rows × 3 statistics), Holm correction keeps 7, which are 5 distinct results:
- the reversed effect of '… on the left side of the table', on both split direction and signed effect;
- the forward signed effect of 'the left {noun}';
- the reversed split direction and signed effect on mirrored images.

No sign-agreement contrast survives."

### 11.5 Decomposition: opposite both-correct = split share × right-way share

**Identity.**
- In an opposite scene where both actions are decided, both instructions can be correct only if the two actions have opposite signs.
- So both-correct = right-way splits / n = (splits / n) × (right-way splits / splits) = x · y.
- This holds exactly in every row (`identity_holds`).

| Row | n | Splits | x = split share [Wilson] | Right way | y = right-way share [Wilson] | Both correct = x·y [Wilson] | [cluster bootstrap] |
|---|---|---|---|---|---|---|---|
| ref | 112 | 21 | 18.8 [12.6, 27.0] | 7 | 33.3 [17.2, 54.6] | 6.2 [3.1, 12.3] | [1.8, 10.8] |
| dx | 142 | 12 | 8.5 [4.9, 14.2] | 6 | 50.0 [25.4, 74.6] | 4.2 [2.0, 8.9] | [1.4, 7.7] |
| sign | 112 | 21 | 18.8 [12.6, 27.0] | 14 | 66.7 [45.4, 82.8] | 12.5 [7.6, 19.9] | [7.0, 18.9] |
| zero | 112 | 21 | 18.8 [12.6, 27.0] | 7 | 33.3 [17.2, 54.6] | 6.2 [3.1, 12.3] | [1.8, 10.8] |
| tok | 112 | 21 | 18.8 [12.6, 27.0] | 7 | 33.3 [17.2, 54.6] | 6.2 [3.1, 12.3] | [1.8, 10.8] |
| w254 | 112 | 21 | 18.8 [12.6, 27.0] | 7 | 33.3 [17.2, 54.6] | 6.2 [3.1, 12.3] | [1.8, 10.8] |
| mir_naive | 115 | 25 | 21.7 [15.2, 30.1] | 21 | 84.0 [65.3, 93.6] | 18.3 [12.3, 26.3] | [11.3, 25.9] |
| argmax | 108 | 20 | 18.5 [12.3, 26.9] | 9 | 45.0 [25.8, 65.8] | 8.3 [4.4, 15.1] | [3.6, 13.9] |
| thr0 | 151 | 35 | 23.2 [17.2, 30.5] | 16 | 45.7 [30.5, 61.8] | 10.6 [6.6, 16.5] | [5.9, 16.0] |
| thr2 | 90 | 17 | 18.9 [12.1, 28.2] | 6 | 35.3 [17.3, 58.7] | 6.7 [3.1, 13.8] | [2.2, 12.2] |
| nf4 | 116 | 19 | 16.4 [10.7, 24.2] | 7 | 36.8 [19.1, 59.0] | 6.0 [3.0, 11.9] | [1.8, 10.4] |
| mir | 115 | 25 | 21.7 [15.2, 30.1] | 4 | 16.0 [6.4, 34.7] | 3.5 [1.4, 8.6] | [0.9, 7.0] |
| w_prenominal | 107 | 26 | 24.3 [17.2, 33.2] | 20 | 76.9 [57.9, 89.0] | 18.7 [12.4, 27.1] | [12.0, 26.4] |
| w_object | 113 | 31 | 27.4 [20.1, 36.3] | 23 | 74.2 [56.8, 86.3] | 20.4 [14.0, 28.7] | [12.6, 27.7] |
| w_table_side | 113 | 24 | 21.2 [14.7, 29.7] | 3 | 12.5 [4.3, 31.0] | 2.7 [0.9, 7.5] | [0.0, 6.1] |
| sub_det | 66 | 11 | 16.7 [9.6, 27.4] | 3 | 27.3 [9.7, 56.6] | 4.5 [1.6, 12.5] | [0.0, 10.6] |
| gpu (A100, NF4, c1) | 117 | 19 | 16.2 [10.6, 24.0] | 6 | 31.6 [15.4, 54.0] | 5.1 [2.4, 10.7] | [1.7, 9.3] |
| gpu comparator (GH200, NF4, c1) | 115 | 19 | 16.5 [10.8, 24.4] | 7 | 36.8 [19.1, 59.0] | 6.1 [3.0, 12.0] | [1.8, 11.0] |

**Clustering.**
- The 151 opposite scenes come from 148 base frames: 145 frames contribute one scene and 3 contribute two.
- In the reference run, the 112 decided opposite scenes come from 110 frames (2 frames with two scenes), and the **21 splits come from 21 different frames**.
- The split-level counts (y) are therefore one per frame, and their binomial and Wilson intervals need no base-frame clustering. For x and both-correct, two frames contribute two scenes each, which is a negligible design effect.
- The Wilson and percentile-bootstrap intervals for both-correct differ by 1–2 points at the upper end. Wilson is better behaved for counts this small, and the bootstrap reaches 0 for table_side and sub_det.

**Reading.**
- x stays between 8.5% (dx) and 27.4% (object): splits are rare in every row.
- y ranges from 12.5% (table_side) to 84% (naive mirror scoring).
- Both-correct is therefore low mainly because splits are rare, and its variation across rows comes mostly from y, the direction of the splits.

**Source.** `btp_evidence_addenda.py::a5_decomposition` → `addenda.json: A5_decomposition`.

**Status.** **VERIFIED.**

### 11.6 Label-swap invariance of the sign-agreement contrast

**Statement.**
- Let (a_s, b_s) be scene s's lateral actions under its "left" and "right" instructions.
- Relabelling every "left" instruction as "right" and vice versa maps (a_s, b_s) to (b_s, a_s). The same-sign indicator 1[a_s · b_s > 0] and each action's decided status are symmetric in the pair. So every same-sign indicator, every per-layout same-sign rate, and hence the paired contrast and its McNemar p, are unchanged.
- The same swap negates every signed difference a_s − b_s, so the signed median changes sign; the two-sided p is unchanged.
- Each scene's targets are fixed by its geometry, so every right-way split becomes a wrong-way split (k → n_splits − k), and opposite both-correct becomes both-wrong.

**Numeric check (reference run; original → all scenes swapped).**
- Contrast: +10.7 → +10.7 points (p = .039 both; 75 pairs and 12 discordant both).
- Same-sign rates: 89.8 / 81.2 / 98.9%, unchanged. Image-right: 44 / 71 / 99%, unchanged.
- Splits: 7/21 → 14/21 (p = .19 both). Opposite both-correct: 6.2% → 12.5%.
- Signed median: −0.151 → +0.151 bins (p = .011 both).
- Swapping a random half of the scenes (seeded) also leaves the contrast at +10.7 (p = .039). The splits become 8/21 and the signed median −0.006 (p = .75).
- So the direction statistics depend on which instruction is called "left"; the contrast does not. This agrees with check K in `reviews/r2_rigour_checks.json`.

**Source.** `btp_evidence_addenda.py::a6_label_swap` → `addenda.json: A6_label_swap` (all checks true).

**Status.** **VERIFIED.**

### 11.7 Wording × run grid (App. D gaps)

**Conventions.**
- `cf1` is primary. Where `c1` differs, its value is given in brackets.
- "4-bit" is the GH200 NF4 replay and ladder, as in claim 9.
- BC = opposite both-correct = right-way splits / decided opposite scenes (§11.5).
- Splits: right-way / splits (binomial p).

**Splits and opposite both-correct**

| Wording | 4-bit | 4-bit mirror | bf16 | bf16 mirror |
|---|---|---|---|---|
| baseline | 7/19 (0.359); BC 7/116 = 6.0% [c1: 7/19; BC 7/115 = 6.1%] | 7/22 (0.134); BC 7/113 = 6.2% | 7/21 (0.189); BC 7/112 = 6.2% | 4/25 (0.00091); BC 4/115 = 3.5% |
| prenominal | 22/28 (0.004); BC 22/113 = 19.5% [c1: 22/28; BC 22/112 = 19.6%] | 23/34 (0.058); BC 23/130 = 17.7% | 20/26 (0.009); BC 20/107 = 18.7% | 20/26 (0.009); BC 20/126 = 15.9% [c1: 19/25; BC 19/126 = 15.1%] |
| object | 28/35 (0.00051); BC 28/115 = 24.3% [c1: 26/33; BC 26/114 = 22.8%] | 18/30 (0.362); BC 18/125 = 14.4% | 23/31 (0.011); BC 23/113 = 20.4% [c1: 21/28; BC 21/112 = 18.8%] | 16/24 (0.152); BC 16/122 = 13.1% [c1: 17/25; BC 17/123 = 13.8%] |
| table_side | 7/28 (0.013); BC 7/112 = 6.2% | 7/24 (0.064); BC 7/115 = 6.1% | 3/24 (0.00028); BC 3/113 = 2.7% | 9/28 (0.087); BC 9/117 = 7.7% |
| absent_noun | 10/21 (1.000); BC 10/104 = 9.6% | 1/18 (0.00014); BC 1/111 = 0.9% | 8/21 (0.383); BC 8/101 = 7.9% | 6/18 (0.238); BC 6/106 = 5.7% [c1: 6/18; BC 6/105 = 5.7%] |
| move | 14/28 (1.000); BC 14/84 = 16.7% [c1: 14/29; BC 14/85 = 16.5%] | 29/43 (0.032); BC 29/99 = 29.3% | 17/29 (0.458); BC 17/93 = 18.3% [c1: 17/31; BC 17/93 = 18.3%] | 21/42 (1.000); BC 21/94 = 22.3% |

**Sign-agreement contrast**, in points (McNemar p; pairs / discordant)

| Wording | 4-bit | 4-bit mirror | bf16 | bf16 mirror |
|---|---|---|---|---|
| baseline | +4.1 (0.549; 74/11) [c1: +4.1 (0.549)] | +2.7 (0.815; 74/18) | +10.7 (0.039; 75/12) | +1.3 (1.000; 79/23) |
| prenominal | +18.2 (0.00012; 77/14) [c1: +17.1 (0.00024)] | +17.1 (0.004; 82/22) | +15.7 (0.013; 70/17) | +6.5 (0.327; 93/26) [c1: +5.4 (0.424)] |
| object | +12.7 (0.031; 79/18) [c1: +14.3 (0.013)] | +8.3 (0.169; 96/26) | +3.7 (0.607; 81/15) [c1: +2.4 (0.804)] | -2.2 (0.824; 89/20) [c1: +0.0 (1.000)] |
| table_side | +14.8 (0.004; 81/16) | -3.9 (0.629; 76/17) | +9.6 (0.039; 83/12) | +2.3 (0.839; 86/24) |
| absent_noun | -3.1 (0.791; 65/14) | -11.3 (0.189; 62/21) | +6.8 (0.481; 59/18) | -9.6 (0.230; 73/25) [c1: -9.7 (0.230)] |
| move | +5.4 (0.648; 56/19) [c1: +7.0 (0.503)] | +6.0 (0.481; 67/18) | -3.2 (0.832; 62/22) [c1: +1.6 (1.000)] | +0.0 (1.000; 62/24) [c1: -1.6 (1.000)] |

**Notes.**
- The 4-bit baseline contrast is flagged because `c1` has 73 pairs, not 74 (+4.11 vs +4.05 points); the p is the same.
- The signed medians for every cell are in claim 9.
- 5 `absent_noun` scenes equal the `object` wording (G5).

**Source.** `btp_evidence_addenda.py::a7_grid, grid_markdown` → `addenda.json: A7_grid, A7_markdown`.

**Status.** **VERIFIED.**


### 11.8 Base-frame structure

**Numbers** (`addenda.json: A8_frames`).
- The 340 analysed scenes come from **191 base frames**: **149 frames hold two scenes** and 42 hold one.
- **Opposite scenes per frame:** 0 in 43 frames, 1 in 145, and **2 in 3** (frames 439, 4051 and 5192).
  - In each of those three frames, one scene was recorded as same-side (both-left twice, both-right once) and relabelled "opposite" by hand.
  - By recorded arrangement no frame holds two opposite scenes: 166 frames have one, 25 have none.
- **Same-side scenes per frame:** 0 in 20, 1 in 153 and 2 in 18. Three frames hold both a both-left and a both-right scene.
- **128 frames hold exactly one opposite and one same-side scene.** These are the pairing units of the McNemar contrast, which uses those where all four actions are decided (75 in the reference run).

**Source.** `btp_evidence_addenda.py::a8_frames`.

**Status.** **VERIFIED.** The coordinator's 149 two-scene frames and 3 two-opposite frames are reproduced.

### 11.9 Frame-level inference for the signed effect

**Definition.**
- **d_s:** scene s's signed left − right difference in bins of 0.000325 (+ = "left" more image-left; folded readout; mirrored images scored correctly). All 340 differences are nonzero.
- **Sign-flip test:** T = Σ_s sign(d_s) · rank(|d_s|). The signs of all scenes in a base frame are flipped together, 20,000 times (seed 20261002).
  - p = (1 + #{|T*| ≥ |T|}) / (1 + 20,000), so the smallest attainable p is 5 × 10⁻⁵ (the "floor").
  - Monte Carlo SE is about √(p(1 − p)/20,000), e.g. ±.0009 at p = .015.
- **Frame bootstrap:** resample the 191 frames with replacement (4,000 draws) and take the median of the pooled scenes; percentile 95% CI.
- **Holm** within run (six wordings) and over all 24 cells, applied to the frame-level p.
- Source: `btp_evidence_addenda.py::a9_frame_level` → `addenda.json: A9_frame_level`.

| Wording | Run | Median (bins) [frame-bootstrap 95% CI] | Wilcoxon p (scene) | Sign-flip p (frame) | Holm within run (frame) | Holm over 24 (frame) |
|---|---|---|---|---|---|---|
| baseline | 4-bit | -0.03 [-0.16, +0.01] | 0.215 | 0.243 | ✗ (0.486) | ✗ (1.000) |
|  | 4-bit mirror | -0.13 [-0.83, +0.00] | 0.026 | 0.025 | ✗ (0.074) | ✗ (0.197) |
|  | bf16 | -0.15 [-0.53, -0.00] | 0.011 | 0.015 | ✗ (0.062) | ✗ (0.154) |
|  | bf16 mirror | -0.48 [-1.31, -0.03] | 0.00077 | 0.002 | ✓ (0.008) | ✓ (0.025) |
| prenominal | 4-bit | +0.74 [+0.28, +1.47] | 1.1×10⁻⁵ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
|  | 4-bit mirror | +0.02 [-0.11, +0.36] | 0.583 | 0.578 | ✗ (0.578) | ✗ (1.000) |
|  | bf16 | +0.35 [+0.07, +0.89] | 0.00075 | 0.002 | ✓ (0.008) | ✓ (0.025) |
|  | bf16 mirror | +0.20 [-0.03, +1.04] | 0.006 | 0.014 | ✓ (0.028) | ✗ (0.153) |
| object | 4-bit | +0.80 [+0.07, +1.61] | 0.00025 | 0.002 | ✓ (0.005) | ✓ (0.025) |
|  | 4-bit mirror | +0.21 [-0.02, +0.91] | 0.050 | 0.074 | ✗ (0.149) | ✗ (0.447) |
|  | bf16 | +0.25 [+0.02, +1.03] | 0.010 | 0.021 | ✗ (0.062) | ✗ (0.185) |
|  | bf16 mirror | +0.29 [-0.01, +1.00] | 0.001 | 0.002 | ✓ (0.008) | ✓ (0.025) |
| table_side | 4-bit | -0.53 [-1.11, -0.14] | 4.0×10⁻⁷ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
|  | 4-bit mirror | -0.81 [-1.51, -0.20] | 5.9×10⁻⁷ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
|  | bf16 | -0.48 [-0.88, -0.15] | 1.1×10⁻⁵ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
|  | bf16 mirror | -0.80 [-1.50, -0.42] | 1.0×10⁻⁶ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
| absent_noun | 4-bit | -0.37 [-0.87, -0.03] | 0.00069 | 0.00095 | ✓ (0.004) | ✓ (0.017) |
|  | 4-bit mirror | -0.49 [-1.32, -0.13] | 5.0×10⁻⁶ | < 10⁻⁴ (floor) | ✓ (0.0003) | ✓ (0.001) |
|  | bf16 | -0.09 [-0.51, +0.01] | 0.041 | 0.040 | ✗ (0.081) | ✗ (0.283) |
|  | bf16 mirror | -0.32 [-0.91, -0.01] | 0.00049 | 0.002 | ✓ (0.008) | ✓ (0.025) |
| move | 4-bit | -0.08 [-0.74, +0.55] | 0.898 | 0.914 | ✗ (0.914) | ✗ (1.000) |
|  | 4-bit mirror | +0.77 [+0.06, +1.62] | 0.00012 | 0.00095 | ✓ (0.004) | ✓ (0.017) |
|  | bf16 | +0.17 [-0.11, +0.98] | 0.151 | 0.205 | ✗ (0.205) | ✗ (1.000) |
|  | bf16 mirror | +0.15 [-0.27, +1.00] | 0.184 | 0.227 | ✗ (0.227) | ✗ (1.000) |

**Counts over the four runs** (scene = Wilcoxon over scenes; frame = sign-flip by base frame):

| Wording | Raw p < .05 (scene / frame) | Holm within run, scene / frame | Holm over 24, scene / frame |
|---|---|---|---|
| baseline | 3 / 3 | 2 / 1 | 1 / 1 |
| prenominal | 3 / 3 | 3 / 3 | 2 / 2 |
| object | 3 / 3 | 3 / 2 | 2 / 2 |
| table_side | 4 / 4 | 4 / 4 | 4 / 4 |
| absent_noun | 4 / 4 | 3 / 3 | 3 / 3 |
| move | 1 / 1 | 1 / 1 | 1 / 1 |

**Checks against the review** (`reviews/r2_rigour_checks.py`, independent code and seed):
- Frame-level p: bf16 original .015 (review .016), object .021 (.021), prenominal .0016 (.0014), table side at the floor (5 × 10⁻⁵).
- Within-run Holm counts: table side 4/4, prenominal 3/4, object 2/4, original 1/4, matching the review.
- The differences are within Monte Carlo error.

**Consequence.** At the frame level, two of the scene-level within-run counts drop by one: object 3/4 → 2/4 and the original wording 2/4 → 1/4. Table side (4/4) and prenominal (3/4) are unchanged.
- bf16 original: scene-level Holm-adjusted p = .041, frame-level .062.
- bf16 object: frame-level p = .021, Holm-adjusted .062.

**Status.** **VERIFIED.** Because 149 of 191 frames hold two scenes, the frame-level counts are the defensible ones.

**Recommended wording.** "With p values from a sign-flip test that respects base frames and Holm correction within each run, the signed effect is significant in 4 of 4 runs for '… on the left side of the table' (reversed), 3 of 4 for 'the left {noun}', 2 of 4 for 'the object on the left' (both toward the named twin) and 1 of 4 for the original wording (reversed)."

### 11.10 Placebo for the direction-blind contrast

**Definition.** The ladder's paraphrase pair names the same (left) twin twice: "grab the {noun} on the left" (role a) vs "take the {noun} on the left" (role b). Its same-side minus opposite sign-agreement contrast is computed exactly as for left/right (`evaluate` → `analysis.same_side_test`, paired McNemar, folded readout). Split and both-correct scores are meaningless for this pair and are not used.

**Numbers** (identical with one bin = /254 or /255):

| Run | Placebo contrast, pts (McNemar p; pairs, discordant) | Same sign, both-left / opposite / both-right (%) | Left/right contrast in the same run |
|---|---|---|---|
| bf16, original images | **+11.7 (p = 0.012; 77, 11)** | 93.8 / 83.8 / 98.9 | +10.7 (p = .039; 75, 12) |
| bf16, mirrored images | +7.8 (p = 0.039; 90, 9) | 94.0 / 89.6 / 95.9 | +1.3 (p = 1.0) |
| 4-bit (GH200), original images | +2.4 (p = 0.774; 85, 12) | 90.2 / 88.9 / 97.8 | +4.1 (p = .55) |
| 4-bit (GH200), mirrored images | +5.4 (p = 0.227; 92, 11) | 89.7 / 87.8 / 94.9 | +2.7 (p = .82) |

**Paraphrase floor with the folded readout** (median |Δ| over 340 pairs, bins of 0.000325):
- **bf16:** left↔right 2.42 vs grab↔take **1.38**; left↔right larger in 59.1%; Wilcoxon p = 0.001.
- **4-bit:** 2.57 vs 1.72; 58.5%; p = 0.00031.
- With `c1` (claim 10 and the draft's App.) these were 2.46 vs 1.37 (p = .002) and 2.57 vs 1.72 (p = .00035).

**Source.** `btp_evidence_addenda.py::a10_placebo` → `addenda.json: A10_placebo`.

**Status.** **VERIFIED.** The review's +11.7 (p = .012; 77, 11), +7.8 (p = .039), +2.4 (p = .77), 1.38 and 2.42 are all reproduced.

**Recommended wording.** "As a placebo, two wordings that name the same twin ('grab' vs 'take the {noun} on the left') give a sign-agreement contrast of +11.7 points (McNemar p = .012) in bf16, as large as left vs right (+10.7, p = .039). The contrast does not isolate the spatial word."

### 11.11 One bin = the /255 grid step (Table 1 regenerated)

**Change.**
- Table 1's reference "one bin" is now the de-normalised grid step (q99 − q01)/255 = 0.0003238 (`btp_evidence.TABLE1_BIN`).
- thr2 = 2 × 0.0003238 = 0.000648; dx's floor is its own /255 step, 0.000224.
- Row w255 is renamed **`w254`**: "one bin taken as /254 = 0.000325 instead of the grid step".
- `btp_evidence.build_table1(ref_bin="254")` rebuilds the previous table, and `btp_evidence_addenda.py::a11_ref255` compares every cell (`addenda.json: A11_ref255`).

**Result: not every row is unchanged.**
- New `ref` equals the old `w255`, and new `w254` equals the old `ref`, so the reference's statistics are identical either way. So are those of 21 of the 22 computed rows, apart from the unit of the signed median.
- **Changed cells at display precision:**
  - **`zero` row:** opposite n 107 → 112; both-correct 6.5% [2.7, 11.4] → 6.2% [1.8, 10.8]; same-sign 81.3 → 81.2%; opposite image-right 56 → 57%; contrast +12.3 → +12.0 points (p = .012 both); splits 7/20 = 35% (p = .26) → 7/21 = 33% (p = .19).
    - Cause: 10 bf16 baseline predictions are near-deltas on bins 128 or 126 (cf1 = −0.000100 or −0.000748). Measured from normalised zero they lie 1.000–1.0025 grid steps away: decided under /255, not under /254.
    - It is a knife-edge of the zero-point row, not of the reference.
  - **`mir`, `mir_naive`, `mir_none`:** signed median ∓0.48 → ∓0.49 (−0.483 → −0.485 bins). This is the unit change crossing a rounding boundary.
  - Labels only: `ref`, `dx`, `thr2`, `w254`.
- **Every signed median scales by 0.000325/0.0003238 = 1.0038.** The argmax row's signed p moves in the fifth decimal (.03800 → .03796).
- No p crosses .05 in any row, and the §11.4 Holm results and the §11.5 decomposition are unchanged apart from the `zero` row.
- **§§1–10 and §11.1–§11.3 keep the project's 0.000325 bins** (`reanalysis.ONE_BIN`).
  - To express their bin-valued numbers in /255 bins, multiply by 1.0038: for example 4.29 → 4.31, 1.36 → 1.36, −0.48 → −0.48/−0.49, 5.86 → 5.88, 7.59 → 7.62.
  - Fractions and p-values are unaffected.
  - No logged 4-bit prediction changes decided status (0 of 2,720); 2 of 2,720 bf16 replay predictions do (claim 2).

**Status.** VERIFIED, with one correction: the `zero` row changes (no verdict changes).

### 11.12 Argmax decisions counted from the zero-motion bin

**Definition.**
- Under argmax, physical zero falls in bin 128, which decodes to −0.31 bins. Bin 127 (normalised zero) decodes to −1.31 bins and bin 129 to +0.69.
- **"Physical value"** (Table 1 row `argmax`): the decoded action is compared with ± k grid steps.
- **"Grid steps from bin 128":** decided iff |b − 128| ≥ k, with the sign from b − 128. "Any nonzero" then means b ≠ 128, which for integer steps is the same as k = 1.
- Source: `btp_evidence_addenda.py::a12_argmax_bin128` → `addenda.json: A12_argmax_bin128`.

**Numbers** (bf16 replay, baseline, n = 680).
- **59 predictions (8.7%) sit on bin 128**; 58 (8.5%) on bin 127; 61 (9.0%) on bin 129.

| Convention | Decided if | Opp. n | Opp. both correct | Opp. same sign | Image-right SSL / opp / SSR | Contrast, pts (p) | Splits right way (p) |
|---|---|---|---|---|---|---|---|
| physical value (a1) | any nonzero value | 151 | 10.6% | 78.8% | 48 / 66 / 96 | +0.8 (1.000) | 16/32 (1.000) |
| grid steps from bin 128 | b ≠ 128 | 127 | 8.7% | 81.1% | 41 / 63 / 97 | +0.0 (1.000) | 11/24 (0.839) |
| physical value (a1) | |value| ≥ 1 step (Table 1 `argmax`) | 108 | 8.3% | 81.5% | 45 / 72 / 98 | +6.9 (0.227) | 9/20 (0.824) |
| grid steps from bin 128 | |b − 128| ≥ 1 (same as b ≠ 128) | 127 | 8.7% | 81.1% | 41 / 63 / 97 | +0.0 (1.000) | 11/24 (0.839) |
| physical value (a1) | |value| ≥ 2 steps | 83 | 8.4% | 83.1% | 46 / 76 / 99 | +11.5 (0.070) | 7/14 (1.000) |
| grid steps from bin 128 | |b − 128| ≥ 2 | 92 | 7.6% | 82.6% | 36 / 70 / 99 | +11.7 (0.039) | 7/16 (0.804) |

- The signed median is 0.00 in every variant (30% ties; r = −0.16; p = .038 physical vs .036 from bin 128).
  - Differences do not depend on the zero point; the p differs only through floating-point rounding of tied differences.
- **Reading:**
  - Counting from the zero-motion bin removes the asymmetry, under which one step right of zero is a decision but one step left is not.
  - At ≥1 step the opposite-scene image-right share falls from 72% to 63% (both-left 45% → 41%), and the contrast from +6.9 (p = .23) to 0.0 (p = 1.0).
  - At ≥2 steps the contrast is +11.7 (p = .039) from bin 128, against +11.5 (p = .070) by physical value. That is one more case where the direction-blind contrast's significance turns on a convention.

**Status.** **VERIFIED.** The review's 8.7% is reproduced.

### 11.13 Multiplicity summary

**Families, as the paper defines them** (folded readout; Holm at α = .05; `addenda.json: A13_multiplicity`).
- **(a) Within each run, over its six wordings.** Rejected:
  - 4-bit: signed, scene: table_side, prenominal, object, absent_noun; signed, frame: prenominal, table_side, absent_noun, object; splits: object, prenominal.
  - 4-bit mirror: signed, scene: table_side, absent_noun, move; signed, frame: table_side, absent_noun, move; splits: absent_noun.
  - bf16: signed, scene: table_side, prenominal, object, baseline; signed, frame: table_side, prenominal; splits: table_side, prenominal, object.
  - bf16 mirror: signed, scene: table_side, absent_noun, baseline, object, prenominal; signed, frame: table_side, object, absent_noun, baseline, prenominal; splits: baseline, prenominal.
- **(b) The signed effect over all 24 cells.** 13 cells are rejected with scene-level p and the same 13 with frame-level p:
  - table side in all four runs;
  - prenominal 4-bit and bf16;
  - object 4-bit and bf16 mirror;
  - absent noun 4-bit, 4-bit mirror and bf16 mirror;
  - move 4-bit mirror;
  - the original wording bf16 mirror.
- **(c) The sign-agreement contrast, within each run over the five alternative wordings.** Rejected:
  - 4-bit: prenominal, table_side (six wordings: prenominal, table_side).
  - 4-bit mirror: prenominal (six wordings: prenominal).
  - bf16: none (six wordings: none).
  - bf16 mirror: none (six wordings: none).
- **(d) The 48 Table 1 tests (descriptive only).** 7 survive, which are 5 distinct results (§11.4).

**The paper's highlighted claims** (✓ survives, with its Holm-adjusted p; n/a = the statistic is not in that family):

| Highlighted claim (folded readout unless argmax) | Raw p (scene; frame) | (a) Holm within run, 6 wordings (scene; frame) | (b) Holm over 24 cells, signed (scene; frame) | (c) Contrast, Holm within run, 5 alternatives | (d) Table 1, 48 tests (descriptive) |
|---|---|---|---|---|---|
| prenominal, bf16: 20/26 splits the right way | 0.009 | ✓ (0.047) | n/a | n/a | ✗ (0.365) |
| table side, bf16: 3/24 splits the right way | 0.00028 | ✓ (0.002) | n/a | n/a | ✓ (0.013) |
| mirrored, bf16 (original wording): 4/25 splits | 0.00091 | ✓ (0.005) | n/a | n/a | ✓ (0.039) |
| original wording, bf16: signed −0.15 | 0.011; 0.015 | ✓ (0.041); ✗ (0.062) | ✗ (0.103); ✗ (0.154) | n/a | ✗ (0.401) |
| prenominal, bf16: signed +0.35 | 0.00075; 0.002 | ✓ (0.004); ✓ (0.008) | ✓ (0.010); ✓ (0.025) | n/a | ✓ (0.034) |
| table side, bf16: signed −0.48 | 1.1×10⁻⁵; 5.0×10⁻⁵ | ✓ (6.9×10⁻⁵); ✓ (0.0003) | ✓ (0.00022); ✓ (0.001) | n/a | ✓ (0.00055) |
| mirrored, bf16 (original wording): signed −0.48 | 0.00077; 0.002 | ✓ (0.003); ✓ (0.008) | ✓ (0.010); ✓ (0.025) | n/a | ✓ (0.034) |
| original wording, bf16: contrast +10.7 (direction-blind) | 0.039 | n/a | n/a | not in family (original wording); six wordings ✗ (0.193) | ✗ (1.000) |
| table side, bf16: contrast +9.6 (direction-blind) | 0.039 | n/a | n/a | ✗ (0.154); six wordings ✗ (0.193) | ✗ (1.000) |
| table side, 4-bit: contrast +14.8 (direction-blind) | 0.004 | n/a | n/a | ✓ (0.017); six wordings ✓ (0.021) | n/a |
| argmax readout, bf16: signed 0.00 (r = −0.16) | 0.038 | n/a | n/a | n/a | ✗ (1.000) |

**What this means for the draft.**
- Survives every family it belongs to, at both scene and frame level:
  - table side, bf16 (signed and splits);
  - prenominal, bf16 (signed);
  - the mirrored replication (signed and splits);
  - and, for the contrast, 4-bit table side in family (c).
- **prenominal bf16 splits (20/26)** survives (a) only narrowly (adjusted .047) and not (d).
- **The original wording's reversed signed effect in bf16 (−0.15, p = .011)** survives only scene-level within-run Holm (.041). It fails at the frame level (.062) and globally (.10 / .15). Present it as raw, or as "nominally significant".
- **Neither bf16 direction-blind contrast highlighted in the draft** (original +10.7 and table side +9.6, both p = .039) survives any family. The original wording is not even in family (c), whose five alternatives exclude it. These are best presented as raw p values that illustrate the statistic's blindness, which is how the draft uses them.
- **The draft's within-run counts sentence (4/3/3/2)** uses scene-level p. The frame-level counts are 4/3/2/1 (§11.9).

---

## 12. Results of the 3 Oct experiments (X1, X2, B0, B1 stage 1, B4.1)

**Regenerate**
- `.venv/bin/python docs/btp_paper/evidence/btp_runs_a100.py` writes `evidence/a100.json` (B4.1, X2).
- `.venv/bin/python docs/btp_paper/evidence/btp_spec_curve.py` writes `outputs/btp/spec_curve.{csv,json}` and `figures/fig_spec_curve.*` (B1).
- `run_gpu_analysis.py --runs outputs/btp/x2_{raw,lower}` gives the natural-frame sections for X2.

**Inputs**
- The A100 runs are in `outputs/btp/a100/`; they were pulled from termitech on 3 Oct. Every A100 run uses batch size 1, greedy decoding and the folded readout, and logs full distributions as sidecars on termitech.
- **Data integrity.** termitech's `v2/probe_predictions.csv` had been truncated (1,864 of 2,720 rows). It was replaced, md5-verified, before the runs that are reported here. The partial first launch was deleted.

### 12.1 X1: readout convention applied in the pipeline
- `reanalysis.load_run(readout="folded")` is now the default. `run_gpu_analysis.py` reports with the folded readout, and the replay-vs-log agreement still uses the 255-token slice, which is the only readout the log carries.
- The changes match §0 G1. For example, the 4-bit `object` contrast moves from p = .013 to p = .031 and Holm no longer rejects it.
- This ledger's own scripts request `readout="255"` explicitly, so §0–§11 regenerate byte-identically. That was checked on 3 Oct.

### 12.2 X2: lower-casing the natural-frame instructions (A100, same GPUs for both)
- **Per prediction.** Lower-casing changes the text of 2,276 of 3,103 natural-frame prompts (73.3%). Among those, the lateral argmax token changes in **11.2%** in bf16 (expected-value r = .950) and 13.0% in NF4 (r = .939).
- **Conclusions** (bf16, raw → lower):
  - **Swap effect:** "left" is more image-left in 45.1% → 44.7% of frames.
  - **Agreement with the demonstrated first motion:**
    - own instruction 83.0% → 82.1%;
    - term removed 83.1% → 82.5%;
    - antonym 76.0% → 75.8%.
  - **Destination words** (n = 988): 44.2% (p = 4.7×10⁻⁷) → 43.9% (p = 6.7×10⁻⁷).
  - **Starting-place words** (n = 23): 60.9% → 56.5%, not significant either way.
  - NF4 shows the same pattern: destination 46.7% → 46.8%, p = .002 → .003.
- **Recommended wording:** "Lower-casing the instructions, as OpenVLA's training does, changes the lateral argmax token in 11% of the affected natural-frame prompts (bf16, r = .95) but no natural-frame conclusion."

### 12.3 B4.1: GPU only, at bf16 (A100 vs GH200, same folded readout, batch size 1)
- **Replay** (2,720): 6.3% of lateral argmax tokens change; r = .980; sign agreement among decided actions 99.2%; median |Δ| 0.07 grid steps.
- **Ladder** (8,160): 7.5% of tokens change; r = .976.
- **Verdicts.** No wording changes direction.

  | Wording | GH200 | A100 |
  |---|---|---|
  | object, splits | 23/31 | 22/30 |
  | table side, splits | 3/24 | 3/25 |
  | prenominal, splits | 20/26 | 20/25 |
  | table side, signed effect | −0.48, p = 1.1×10⁻⁵ | −0.37, p = 9×10⁻⁵ |

- **Exception.** The original wording's fragile reversed effect loses significance: signed −0.15 (p = .011) on the GH200 vs −0.06 (p = .070) on the A100. Its splits are 7/21 vs 8/20, and opposite both-correct is 6.3% vs 7.1%.
- **Recommended wording:** "Moving bf16 inference from a GH200 to an A100 changes 6–7% of lateral argmax tokens (r = .98) and no wording's direction, but it moves the original wording's signed effect from p = .011 to p = .07."

### 12.4 O4: placebo-corrected contrast (B0 outcome; A100 bf16; original images)

**Definition**
- The contrast is the same-side minus opposite same-sign rate, unpaired, over decided scenes (≥1 grid step), in points.
- O4 is the left/right contrast minus the placebo contrast. The placebo pair is "grab …"/"take …" naming the same twin, in the same wording.
- The CI is a base-frame bootstrap with 4,000 draws; p comes from a base-frame label-swap permutation with 10,000 draws.

| Wording | Left/right | Placebo | O4 [95% CI] | p |
|---|---|---|---|---|
| original "… on the left" | +12.9 | +12.2 | **+0.7** [−8.6, +9.6] | .89 |
| "the left {noun}" | +16.4 | −1.1 | **+17.5** [+7.7, +27.5] | .001 |
| "the object on the left" | +6.8 | −6.3 | **+13.2** [+0.3, +26.0] | .049 |
| "… left side of the table" | +8.2 | −0.3 | **+8.5** [−2.1, +18.7] | .12 |
| original, GH200 (bf16) | +14.0 | +13.0 | **+0.9** [−8.0, +10.3] | .84 |

**Reading**
- The original wording's direction-blind contrast is entirely matched by a pair that names the same twin, so it says nothing about the word.
- "The left {noun}" separates the two instructions beyond the placebo. With its positive O1, that rules out a word→direction shortcut for this wording, which would give O4 < 0. It still falls far short of selection (O3 upper bound below 50%; §12.5).
- **Recommended wording:** "A placebo pair that names the same twin reproduces the original wording's contrast (+12.2 vs +12.9 points; difference +0.7 [−8.6, +9.6]) but not that of 'the left {noun}' (−1.1 vs +16.4; +17.5 [+7.7, +27.5], p = .001)."

### 12.5 B1 stage 1: specification curve (pre-declared design, `analysis_plan.md`, SHA-256 4b4cbedc…)

**Universe**
- 16 GH200 facets: 4 wordings × 2 readouts (expected value; argmax counted from bin 128) × 2 precisions (bf16; NF4).
- Within a facet, 4 subsets give the O1 specifications (64 in all), and 4 subsets × 3 thresholds give the O2/O3 specifications (192 in all).
- The joint test uses 10,000 base-frame label flips, Stouffer's Z, two-sided, with Holm across the 16 facets.

**No reliable selection.** The upper base-frame-bootstrap 95% bound of opposite both-correct is at most **39.2%** in all 192 specifications.

**Direction is set by the wording.**

| Wording | Facet Z (EV bf16 / EV NF4 / argmax bf16 / argmax NF4) | Holm p | Direction |
|---|---|---|---|
| "the left {noun}" | +6.18 / +6.39 / +5.57 / +5.03 | ≤ .046, all four | toward |
| "the object on the left" | +4.68 / +5.38 / +3.82 / +4.23 | .07 / .034 / .16 / .11 | toward (1 of 4 survives Holm) |
| "… left side of the table" | −6.65 / −7.98 / −3.67 / −7.07 | .006 / .002 / .16 / .003 | away (3 of 4 survive) |
| original "… on the left" | −5.55 / −2.11 / −5.10 / −0.12 | .022 / .52 / .038 / .95 | away in bf16, null in NF4 |

- **Argmax facets.** Most pairs tie, so the median O1 is exactly 0. Their direction is reported as the Stouffer sign (post hoc note, `analysis_plan.md`).
- **Variance of the 64 O1 estimates** (Shapley shares of OLS R²): wording **0.42**, readout 0.04, precision 0.03, scene subset 0.002; total R² 0.49.

**Recommended wording:** "Across 16 facets and 256 specifications, no specification shows reliable selection (upper 95% bound of opposite-scene both-correct at most 39%). The wording sets the direction: every facet of 'the left {noun}' points toward the named twin (Holm p ≤ .05), three of four table-side facets point away, and the original wording points away only in bf16. Wording explains 42% of the variance of the estimated word effect across specifications; readout, precision and scene subset explain 4%, 3% and under 1%."

### 12.6 B2: autoregressive prefix control (bf16, A100; `evidence/prefix_bf16.json`)

**Regenerate:** `.venv/bin/python docs/btp_paper/evidence/btp_prefix_analysis.py --precision bf16`.

**Method**
- A hand two-step decode passes its gate: dy after the model's own greedy dx reproduces `generate()` to 4×10⁻⁶ bins on 10 units, and to 9×10⁻⁵ grid steps against the full A100 ladder and replay runs.
- Every forced step runs alone at batch size 1. A batched step changed dy by up to 3.6 bins in bf16, so batching was dropped.
- Effects are oriented left−right dy differences in grid steps, positive when "left" moves further image-left.
  - **Total:** each instruction after its own dx token.
  - **CDE(t):** both instructions after the same dx token t.
  - **Direct:** the mean of CDE(dx_left) and CDE(dx_right).
  - **Indirect:** total − direct.
  - **Marginal direct:** dx weighted by p(dx | term-free prompt), tokens covering 99% of its mass.
- Values are **means** over the 340 scenes with base-frame bootstrap 95% CIs (4,000 draws). Medians do not add, so the decomposition uses means.

| Wording (original images) | Total | Direct | Indirect | Marginal direct | dx differs | Splits right way: own / dx fixed at the term-free token |
|---|---|---|---|---|---|---|
| original "… on the left" | −0.85 [−3.37, +1.43] | **−2.85** [−4.43, −1.44] | +2.00 | −2.27 | 60% | 8/20 / 5/12 |
| "the left {noun}" | **+5.08** [+1.90, +8.22] | **+4.69** [+2.28, +7.22] | +0.40 | +3.89 | 62% | 20/25 / 16/21 |
| "the object on the left" | **+4.70** [+1.17, +8.37] | **+5.04** [+2.30, +7.93] | −0.34 | +5.57 | 62% | 22/30 / 15/20 |
| "… left side of the table" | **−3.39** [−5.99, −0.89] | **−6.11** [−8.05, −4.16] | +2.72 | −4.42 | 62% | 3/25 / 5/21 |

- **Mirrored images (Type U), direct effects:** original −6.10, prenominal +3.09, object +4.52, table side −9.90. Each has the sign of its total, all with CIs excluding 0.
- The word × dx interaction, CDE(dx_left) − CDE(dx_right), is between −0.6 and +1.2 grid steps in every cell.

**Reading**
- The word acts on dy mainly **directly**: holding dx fixed keeps the sign of every wording's effect, and the direct effect is at least as large as the total.
- For the original and table-side wordings, the path through the earlier-decoded dx token runs the other way (+2.0 and +2.7) and partly cancels the direct effect.
- As a result, the original wording's **total** effect is null, while its **direct** effect is reversed.
- The prefix's role differs from what the association in §5 suggested ("pairs with different dx carry most of |Δdy|"). Magnitude moves with dx, but direction does not come from it.

**Recommended wording:** "Holding the earlier-decoded dx token fixed keeps the sign of every wording's effect and does not shrink it ('the left {noun}': direct +4.7 vs total +5.1 grid steps; table side −6.1 vs −3.4). For the original wording the path through dx runs the other way, so its total effect is null (−0.9 [−3.4, +1.4]) while its direct effect is reversed (−2.9 [−4.4, −1.4]): the estimand decides the verdict."

**Table I rows for the prefix** (`evidence/prefix_table1_rows.json`)
- Setting: original wording, original images, A100 bf16, folded readout, ≥1 grid step. Compare the two rows with each other, not with the GH200 reference.
- Columns: opp both-correct (n); image-right SSL/opp/SSR; contrast in points (p); splits right way (p); signed median in grid steps (p).

| Row | Opp. both-correct (n) | Image-right SSL/opp/SSR | Contrast, pts (p) | Splits right way (p) | Signed median (p) |
|---|---|---|---|---|---|
| A100, own greedy dx (total effect) | 7.1% (113) | 42/71/99 | +10.5 (.008) | 8/20 (.50) | −0.06 (.070) |
| A100, dx fixed at the term-free token (direct effect) | 4.2% (118) | 42/74/98 | +4.7 (.29) | 5/12 (.77) | −0.02 (.010) |

**Reading:** fixing dx makes the original wording's signed effect significant and reversed (p = .010). It also more than halves the direction-blind contrast (+10.5 → +4.7, p = .008 → .29), so much of that contrast rides on dx-token changes.

### 12.7 B7-lite: reporting audit (PRELIMINARY: one AI coder; `lit/b7_lite.md`)

**Status.** These counts must not go into the paper as findings until two humans (S, L) have double-coded at least 25% of the papers (plan B7). Until then the paper may only say "a preliminary audit", qualitatively.

**Sample.** The ten works in `lit/lit_vla.md` §(b), frozen before coding: STAGE, LIBERO-CF, BeTTER, LangGap, InstructMove, RefGuard, RoboIRGBench, Act2Answer, Kirouane et al., and Chen et al. (2026).

**Coding.** The 12 card items, each coded R (reported), P (partly), N (not) or N/A. Items 1–4 are N/A for papers that read no action values.

| Item | R / P / N | n coded |
|---|---|---|
| 1 axis and sign | 0 / 2 / 0 | 2 |
| 2 token map | 0 / 0 / 1 | 1 |
| 3 readout rule | 0 / 0 / 2 | 2 |
| 4 decoding order | 0 / 0 / 1 | 1 |
| 5 horizon and rate | 1 / 5 / 4 | 10 |
| 6 precision | 0 / 5 / 5 | 10 |
| 7 observations | 0 / 8 / 2 | 10 |
| 8 instructions | 0 / 10 / 0 | 10 |
| 9 training-data word use | 4 / 5 / 1 | 10 |
| 10 screening | 2 / 8 / 0 | 10 |
| 11 decisive statistic | 2 / 8 / 0 | 10 |
| 12 sensitivity | 4 / 0 / 6 | 10 |

**Patterns**
- 8 of 10 papers read no action values. Neither of the two that do states its readout rule.
- No paper states its inference precision; every partial credit is for a training GPU or a training dtype.
- No paper states how instruction text is normalised.
- Six of ten never test an alternative analysis choice.

**A real-world pipeline pitfall** (verified 3 Oct from the primary sources)
- **Finding.** Chen et al. (arXiv 2609.39971, App. B): "For its Semantic and Task cells, LIBERO-PRO writes the perturbed instruction into the (:language ...) block of each regenerated BDDL file, but its released evaluation code reads the instruction from the file name, so the policy receives the original instruction while the success predicate scores the perturbed task."
- **Issue.** The GitHub issue Zxy-MLlab/LIBERO-PRO#14 exists. It was opened on 30 Dec 2025 and is still open, and it describes the same symptom.
- **Acknowledgement.** Chen et al. say the maintainers acknowledged it, but no maintainer reply was visible in the fetched issue page. Do not claim an acknowledgement.
- **Recommended wording:** "In one released language-perturbation benchmark, the policy received the original instruction in the paraphrase and new-goal tests while the perturbed task was scored [chen2026instructions]."

**12.6b NF4 replication of the prefix control** (A100; `evidence/prefix_nf4.json`; same method; means in grid steps with base-frame 95% CIs)

| Wording (original images) | Total | Direct | Indirect |
|---|---|---|---|
| original | −1.85 [−4.57, +0.80] | **−3.46** [−5.13, −1.95] | +1.61 |
| "the left {noun}" | **+5.03** [+1.77, +8.33] | **+4.33** [+2.11, +6.70] | +0.70 |
| "the object on the left" | **+4.91** [+1.48, +8.57] | **+5.80** [+3.11, +8.72] | −0.89 |
| "… left side of the table" | **−5.32** [−7.71, −3.06] | **−5.37** [−7.10, −3.68] | +0.05 |

**Mirrored images (Type U), total → direct**
- original: −1.76 [−4.77, +1.09] → **−4.52** [−6.67, −2.54];
- "the left {noun}": +0.69 [−2.19, +3.58] → **+2.55** [+0.59, +4.80];
- object: +2.16 [−0.55, +5.00] → **+2.31** [+0.47, +4.31];
- table side: −7.07 → −7.15.

**Reading**
- The bf16 pattern replicates in NF4. The direct effect always has the sign of the total and is significant in every cell.
- In 1 of 4 original-image cells (the original wording) and 3 of 4 mirrored cells, the total effect's CI includes 0 while the direct effect's does not.
- So reading dy after the model's own dx token can hide a word effect in either direction.

**Recommended wording:** "Across both precisions and both image sets, holding dx fixed never changes the sign of a wording's effect, but in five of 16 cells (2 precisions × 2 image sets × 4 wordings) it turns a null total effect into a significant direct one." The five cells are: bf16 original/original; NF4 original/original; NF4 mirrored original, prenominal and object.

---

## 13. Round 2 (3 Oct): pre-processing (B5), screened-out scenes (B6) and the compute ladder (B4)

**Source.** `evidence/round2.json`, regenerated by `.venv/bin/python docs/btp_paper/evidence/btp_runs_round2.py`.

**Comparator.** Every row is compared with B4.1 (bf16, eager, A100, same stimuli), never with the GH200 runs. All runs use batch size 1 and the folded readout. The replay subset is 2,040 predictions (baseline, neutral, mirror and mirror-neutral); the ladder is 4,080 (6 wordings × 340 × 2, original images).

### 13.1 B5: observation pre-processing
- **official:** OpenVLA's Bridge evaluation code (`bridgev2_utils.resize_image`) applied to our 640×480 frames: JPEG encode/decode, then tf.image.resize to 224 with lanczos3 and antialias. TensorFlow 2.17 runs in a separate environment; the images are saved as PNG and fed to the processor.
- **bridgeorig:** an *approximation* of the training images: area resize to 256×256 (the bridge_orig resolution; the builder's method is an assumption), then the official path.

| | Lateral tokens changed vs B4.1 (replay / ladder) | r (replay / ladder) | Sign agreement, decided |
|---|---|---|---|
| official | **59.8%** / 59.1% | .588 / .506 | 89.9% |
| bridgeorig | **68.4%** / 69.0% | .605 / .549 | 86.5% |

Verdicts (original images; splits right way (p); signed median in grid steps (p); opposite both-correct):

| Wording | Ours (B4.1) | official | bridgeorig |
|---|---|---|---|
| original | 8/20 (.50); −0.06 (.070); 7.1% | 3/9 (.51); −0.04 (.54); 3.0% | 8/29 (.024); **−0.82** (.0003); 6.3% |
| "the left {noun}" | 20/25 (.004); +0.39 (.0002); 18.3% | 15/18 (.008); +0.40 (.0001); 15.8% | 35/43 (<10⁻⁴); **+1.84** (5×10⁻⁶); **28.0%** |
| "the object on the left" | 22/30 (.016); +0.20 (.010); 19.5% | 14/17 (.013); +0.47 (2×10⁻⁶); 14.4% | 23/28 (.0009); +1.07 (2×10⁻⁵); 18.4% |
| "… left side of the table" | 3/25 (.0002); −0.37 (9×10⁻⁵); 2.7% | 5/15 (.30); −0.09 (.016); 4.9% | 9/32 (.020); **−1.61** (9×10⁻¹⁰); 6.9% |

**Headline for the original wording**

| | Same-sign | Image-right SSL/opp/SSR | Contrast |
|---|---|---|---|
| ours | 82.3% | 42/71/99 | +10.5 (.008) |
| official | 91.1% | 53/78/96 | −1.5 (1.0) |
| bridgeorig | 77.2% | 38/70/94 | +7.9 (.15) |

**Reading**
- Pre-processing changes most individual lateral actions: 60–68%, more than NF4 vs bf16.
- It changes no wording's direction. The original wording is null under two paths and away under the third, which is already fragile everywhere.
- It does not produce reliable selection: opposite both-correct is at most 28.0%.
- The path closest to the training images **amplifies every word effect**: 4–5× in grid steps ("the left {noun}" +1.84 vs +0.39; table side −1.61 vs −0.37).
- The magnitudes in both papers therefore depend on the observation pipeline. The directions do not.

**Recommended wording:** "OpenVLA's own evaluation pre-processing changes the lateral action in 60% of predictions, and an approximation of its training images in 68%, yet no wording changes direction; the latter enlarges every word effect about fourfold ('the left {noun}': +1.8 vs +0.4 grid steps), with opposite-scene both-correct still at most 28%."

### 13.2 B6: screened-out composites (A100 bf16; original wording)
- Approved (820) and rejected (824) composites are both scored by their *recorded* arrangement, so hand relabelling (53 of 340 frozen layouts) is not involved.
- The approved set includes the 400 frozen scenes.

| | Opposite both-correct | Opposite same-sign | Image-right SSL/opp/SSR | Splits right way | Signed median (p) |
|---|---|---|---|---|---|
| approved | 5.4% | 82.9% | 47/72/88 | 17/54 | −0.10 (.0005) |
| rejected | 8.1% | 80.1% | 40/63/83 | 22/54 | −0.14 (.001) |

- **Reading:** the screened-out composites give the same verdict: no selection, a layout gradient, and the original wording weakly reversed. Screening narrows the scope; it does not drive the verdict.
- **Recommended wording:** "Scored the same way, the 824 composites that screening rejected give the same verdict as the 820 it approved (opposite both-correct 8.1% vs 5.4%; the original wording reversed in both, −0.14 vs −0.10 grid steps)."

### 13.3 B4: compute ladder (A100; each setting compared with B4.1 bf16 eager on the same stimuli)

| Setting | Lateral tokens changed (replay / ladder) | r (replay) | Sign agreement, decided | Directions: original / prenominal / object / table side |
|---|---|---|---|---|
| OpenVLA's own 4-bit default (`load_in_4bit=True`: FP4, no double quantisation) | **67.0%** / 67.3% | .642 | 86.9% | away / toward / toward / away |
| int8 (LLM.int8) | **31.9%** / 31.9% | .845 | 95.3% | away / toward / toward / away |
| fp16 | 7.4% / 7.1% | .967 | 99.4% | away / toward / toward / away |
| fp32 | 7.4% / 6.9% | .969 | 99.4% | away / toward / toward / away |
| bf16, sdpa attention | 6.7% / 5.6% | .978 | 99.4% | away / toward / toward / away |
| bf16 repeat (same GPUs) | **0.0%** | 1.000 | 100% | (replay only) |

- **How directions are called:** the sign of the signed median, where its Wilcoxon p < .05; otherwise null. The reference itself (B4.1) calls the original wording null (−0.06, p = .070).

**Reading**
- **Determinism.** Runs are deterministic at batch size 1.
- **Size of the change.** fp16, fp32 and an attention-kernel change move about 7% of lateral tokens, the same as a GPU change (§12.3). int8 moves 32%. 4-bit moves 58% (NF4 on the GH200) to 67% (OpenVLA's own FP4 default).
- **"4-bit" is not one setting.** FP4 and NF4 differ, and OpenVLA's paper does not name its type.
- **No wording changes direction** in any setting.
- **The original wording stays the fragile case.** It is reversed in all five non-reference settings but null in the reference. Under FP4 it is strongly reversed: −0.40 (p = 2×10⁻⁵), splits 3/19.
- **Selection stays unreliable.** Opposite both-correct is at most 21.2% (object, fp32).

**Recommended wording:** "Repeated runs are bit-identical. Changing the dtype (fp16, fp32) or the attention kernel moves about 7% of lateral actions, int8 32%, and OpenVLA's own 4-bit default 67% (NF4: 58%); no wording changes direction in any setting, and only the original wording's verdict moves (null in bf16, reversed in the other five)."

**B6, further columns used in Table I's `screen` rows** (from `evidence/round2.json`, the same run):

| | Contrast, pts (McNemar p) | Splits right way (binomial p) | Opposite both-correct CI |
|---|---|---|---|
| approved | +6.4 (0.064) | 17/54 (0.009) | [3.2, 7.9] |
| rejected | +2.0 (0.711) | 22/54 (0.220) | [5.2, 11.4] |
