# Specification curve: pre-declared analysis plan (B0)

**Frozen:** Sat 3 Oct 2026, before any specification curve was computed. Its SHA-256 is recorded in `docs/btp_experiments_plan.md` §B0.
**Status:** this is a pre-declared analysis, not a pre-registration, because the data have been seen (Table I, the ledger).
**Follows:** Simonsohn, Simmons & Nelson (2020) and Del Giudice & Gangestad (2021), as summarised in `lit/lit_methods.md` §(c), with the rigour review's corrections (`reviews/r2_rigour.md`, R-P1–R-P6, R-P13).
**Changes:** any change after the curve is computed is listed at the end of this file, marked post hoc.

## 1. Question and unit

- **Hypothesis H.** "left"/"right" moves OpenVLA-7B's first lateral action toward the named twin by selecting it, rather than by a word→direction shortcut.
- **Unit.** A left/right pair of instructions on one scene.
- **Clusters.** The 340 scenes come from 191 base frames. Every p-value and CI respects them:
  - sign flips and label flips are made jointly per base frame;
  - CIs come from a base-frame bootstrap with 2,000 draws.

## 2. Outcomes (all directional)

- **O1. Signed word effect toward the named twin's side.**
  - Definition: mean and median over all pairs of the oriented left−right difference, in grid steps of (q99 − q01)/255. Positive means "left" moves further image-left than "right" does.
  - Test: two-sided base-frame sign-flip test of the Wilcoxon statistic, with 10,000 flips.
  - Caveat: a word→direction shortcut also makes O1 positive.
- **O2. Share of decided opposite-scene splits going to the named twins.**
  - Test: two-sided binomial test against ½. Opposite scenes come from distinct base frames in all but three.
  - Caveat: a shortcut also makes O2 > ½.
- **O4. Placebo-corrected contrast.**
  - Definition: the same-side − opposite sign-agreement contrast for the left/right pair, minus the same contrast for a placebo pair naming the same twin ("grab …"/"take …"), paired by base frame.
  - Reading: O4 > 0 under grounding, O4 < 0 under a shortcut, O4 ≈ 0 under layout-only behaviour.
  - Test: base-frame bootstrap CI, plus a base-frame permutation test that swaps the pair labels.
  - Availability: the original wording in every run (the `paraphrase` pair); all four wordings in the A100 bf16 run (B4.1 placebo pairs).
- **O3. Opposite-scene both-correct**, descriptive only.
  - "No reliable selection" means: the upper base-frame-bootstrap 95% limit of O3 is below 50% in every specification.
- **No direction-blind statistic is used as a verdict.** The sign-agreement contrast appears only inside O4 and in the pitfall audit.

## 3. Pitfalls: fixed at their correct values, never varied

- dy (component 1); image-right negative; zero = physical zero.
- Under argmax, a decision counts grid steps from bin 128 (decided iff |b − 128| ≥ k), takes its sign from b − 128, and "any nonzero" means b ≠ 128.
- Expected value over all 256 action tokens with token 31744 folded (`cf1`).
- One bin = the /255 grid step.
- Mirrored images are scored by swapping and negating the targets.

## 4. The universe

**Type N facets: different estimands, so separate curves and joint tests**
- Wording: original "… on the left"; prenominal "the left {noun}"; object "the object on the left"; table_side "… on the left side of the table".
- Readout: expected value (the expected action) or argmax (the executed action under greedy decoding).
- Precision: bf16 or NF4 (4-bit NF4, bf16 compute), both on the same GH200 (the existing ladder and replay runs).
- Prefix (from B2, A100): own greedy dx (the total effect) or a forced common dx token (the controlled direct effect). The prefix facets are reported against B2's own-prefix rows on the same GPU, never against GH200 rows.
- That gives 4 × 2 × 2 = **16 GH200 facets**, plus the B2 prefix facets: 4 wordings × 2 prefixes × the precisions B2 covers.

**Type E specifications within a facet**
- Decided-threshold of 0, 1 or 2 grid steps (O2 and O3 only). Justification: measurement resolution; the same hypothesis at different detection thresholds.
- Scene subset: all 340; detected gripper (199); not relabelled (287); both. Justification: label quality; the same hypothesis on cleaner subsets.
- That gives 12 specifications per facet for O2 and O3, and 4 for O1 and O4.

**Type U: exploratory panels only, never pooled with the curves**
- Mirrored images, correctly scored. They are out of distribution for OpenVLA, and their effect differs (−0.49 vs −0.15 grid steps in bf16).
- Screening: the screened-out composites (B6) and approval reweighting.
- GPU: the same facets on the A100 bf16 run (B4.1).

## 5. Verdicts and joint inference

- **Per specification.** toward (O1 > 0, frame-level p < .05), away (O1 < 0, p < .05), or null.
- **Facet direction.** The sign of the facet's median O1.
- **Selection.** A facet is said to *select* the named twin only if all of these hold:
  - O1 > 0 and O4 > 0, both significant;
  - the O3 "no reliable selection" rule fails.
- **Primary joint test per facet.** Stouffer's Z over the facet's O1 specifications, signs kept, two-sided.
  - The null is built from 10,000 datasets in which the left/right labels are flipped jointly within each base frame.
  - Holm correction across facets.
- **Descriptive per facet.** The median O1; and the shares of specifications significant toward and away, each with a permutation p (ties count half).
- **Across facets.** Shapley shares of the variance of the O1 estimates over the indicators wording, readout, precision and prefix.

## 6. Display (Fig. 2 or the appendix)

- Four panels, one per wording (bf16, expected value, own prefix). Each shows its O1 and O2 specifications sorted by estimate with 95% CIs, and a dashboard of the Type E choices underneath.
- A facet-summary strip: one point per facet with its median O1 and Holm-adjusted joint p.
- Type N facets are never merged into one sorted curve.

## 7. Decision rules for the paper

- **No facet selects the named twin, and the wording facets differ in direction:** §V states this with the primary statistic. This is the expected outcome, from Table I.
- **A facet contradicts Table I:** the curve wins and the text changes.
- **O4 is unavailable for a facet:** report O1 and O2 for it, and say plainly that they cannot separate grounding from a shortcut.

## Post hoc changes

1. **3 Oct, after the stage-1 curve was computed.** Argmax facets can have a median O1 of exactly 0 because most pairs tie. For those facets the direction is also reported as the sign of the facet's Stouffer Z. The pre-declared median rule still gives "none (median 0)", and both are shown.
