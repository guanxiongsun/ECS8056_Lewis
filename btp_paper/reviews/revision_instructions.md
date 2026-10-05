# Revision instructions for draft v2 (orchestrator, 2 Oct 2026, evening)

Read first: `reviews/r1_workshop.md` (workshop reviewer; weak accept → accept with edits 1–5), `reviews/r2_rigour.md`
(rigour audit; 4 blockers), and the ledger's new §11 addenda in `evidence/claims_ledger.md` (the only source for any
new number; the regenerated `evidence/one_at_a_time.{csv,md}` uses the /255 reference and adds rows `gpu` and `w254`).
The updated plan is `docs/btp_experiments_plan.md` (v2); its IDs (B0–B11, B4.1) are the ones `\pending{}` should cite.

Where R1 and R2 conflict, R2 wins on facts and statistics, R1 on story and presentation. Everything below is decided;
`open_questions.md` should only list what is still open after this round.

## A. Must fix (blockers and decisions)
1. **Facts about the stimuli and pipeline** (R2 R-B1, R-M10; R1 §3).
   - Frames are **640×480** RGB from the Open X-Embodiment release of BridgeData V2
     (`gs://gresearch/robotics/bridge/0.1.0/`), passed straight to the processor (resize to 224×224). OpenVLA trained
     on `bridge_orig` (the official release, 256×256), and OpenVLA's README calls the OXE version "out of date (as of
     12/20/2023)". Use R2's replacement text at l. 487, l. 252 and l. 266; card item 7 becomes "source release and
     resolution; pre-processing; edits; transforms and re-scoring".
   - The second twin is the object's **own pixels cut out, mirrored and pasted**: the twins are mirror images, not
     identical objects. Say so where twins are introduced (l. 55, l. 120, l. 266).
   - Objects and the gripper are located with **OWLv2** (`\note{cite OWLv2}` — not in refs.bib), with an image-centre
     fallback for the gripper.
   - The runner disables TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`): add to App. A "Runs" and card item 6.
2. **The hook and the conclusion** (R1 edit 1; R2 R-B2, R-M1, R-M6).
   - Lead with reversals on **identical outputs**: mirror scoring (correct 4/25 vs naive 21/25 splits the right way on the
     same mirrored outputs); the direction-blind contrast (exactly invariant to swapping the two instructions' labels;
     the placebo pair "grab"/"take the {noun} on the left", which names the *same* twin, scores +11.7 points, McNemar
     p = .012, as much as left/right — ledger §11); the axis and sign conventions (dx erases the layout effect and the
     stimulus check; a flipped sign reverses the layout gradient). Wording and precision come second, explicitly as
     forks that change the question.
   - Keep **selection** (does the model pick the named twin? robustly no: opposite both-correct ≤ 20.4% in every row)
     apart from **direction** (which way does the residual word effect push? fragile, set by wording). Do not write
     "finds correct grounding" (l. 57) or "grounds them backwards" for a direction result; use "the residual word effect
     points toward/away from the named twin".
   - The bf16 original-wording contrast (+10.7, p = .039) does **not** survive the paper's own correction (Holm .19):
     either tag it "uncorrected" or, better, use the robust example (4-bit, table side: +14.8 points, p = .004, Holm
     ≈ .02, while its signed effect is reversed) — numbers from the ledger.
   - Abstract (R-M1): "Two conclusions survive every one-at-a-time variation: the model does not reliably select the
     named twin, and its two instructions mostly move the same way; a third, that the action follows the layout, fails
     only under two pitfalls. The direction of the residual word effect, which the wording sets, does not survive, and
     neither does the significance of the direction-blind contrast." (adapt; keep ≤ 160 words).
   - Title: R1's first choice, "Same Outputs, Opposite Verdicts: Pitfalls and Forks Beneath a VLA Language
     Evaluation" (drop "Language" only if it forces a third title line). List the current title as the alternative.
3. **"Positive is grounded" is wrong** (R2 R-B3). The signed left−right effect is positive under grounding *and* under a
   word→direction shortcut; only same-side scenes separate them. Use "positive = toward the named twin's side" in §II,
   Table I/IV/VII captions and App. E, and add O4 (placebo-corrected contrast) to App. E as in plan B0.
4. **Pending work is not claimed** (R1 edit 4; R2 R-B4). Unless the curve exists, "the first specification-curve
   analysis" becomes R2's fallback: "the first audit of the measurement layer beneath a VLA language verdict, with a
   pre-declared specification-curve design (App. E) for which we found no precedent in robot learning"; contribution
   (ii) becomes "a fork analysis and a pre-declared specification-curve design faceted by estimand". No "multiverse"
   claim. Triage main-text `\pending{}`: delete the B3 slot (replace with the invariance + placebo argument), keep B2
   (prefix) and B6 (screened-out) as one short pending clause each, and turn the B7 slot into the qualitative statement
   from lit-A ("none of the closest evaluations states its readout rule") plus a pending count.
5. **Statistics** (R2 R-M2, R-M3, R-M12; R1 edit 3).
   - Signed-effect p-values: base-frame sign-flip tests (ledger §11; 191 frames). Splits: binomial (opposite scenes come
     from distinct frames in all but three). CIs: base-frame bootstrap. Update card item 11 accordingly.
   - Define the Holm families once (App. B), as R2 R-M3 words them; tag p-values that do not survive as "uncorrected";
     add the Table I note: rows are sensitivity analyses, uncorrected; across its 48 tests, Holm keeps 7 (both sides of
     the mirror reversal, prenominal signed, table-side signed and splits) — ledger §11.
   - Wording significance counts with frame-level p (ledger §11): table side 4/4, prenominal 3/4, object 2/4,
     original 1/4 (within-run Holm). Use the ledger's final numbers.
   - In §V "Fragile", say that the contrast's significance turns on a handful of base frames (discordant pairs, R-M12)
     and define "strength" as a change in size or significance without a change of sign.
6. **Fork definition** (R1 edit 5; R1 W6). Forks are legitimate choices without a single right answer. Some change the
   estimand (Type N: readout, prefix, wording, precision): report each estimand. Others are equivalent tests of the same
   claim (Type E: threshold, subset): report their specification curve. Uncertain ones (Type U: mirrored images,
   screening): exploratory. Mirrored images are Type U (out of distribution for OpenVLA; R1 §2.2, R2 R-P4). Keep
   Type E/N/U out of the introduction if space is short (§IV can carry it).
7. **The /255 reference** (R2 R-M9). One bin = the grid step (q99 − q01)/255; rename row `w255` → `w254`
   ("one bin taken as /254"); thr2 = 0.000648. Use the regenerated one-at-a-time table.
8. **Units and scale** (R1 edit 9; R2 R-M4). One bin ≈ 0.32 mm of lateral end-effector motion per 0.2-s step (hedge as
   the ledger does on "metres"); the median first lateral command is 4.3 bins; the two instructions differ by a median of
   2.4 bins (grab/take 1.4); the residual word effects are fractions of a bin. One or two sentences in §II; say that
   the "decided" threshold is a deadband.

## B. Should (as space allows, in R1's order)
- **Effect figure** (R1 edit 2): replace the Fig. 2 placeholder in the main text with a real figure from existing
  numbers: for each Table I row (plus `gpu`), x = share of decided opposite scenes that split, y = share of splits that
  go to the named twin (Wilson CIs from the ledger), with iso-lines of both-correct = x·y, marked pitfall/fork. Every
  row sits at 8–27% splits while y spans 12–84%: selection fails because splits are rare; direction among splits is
  fragile. Write `figures/make_fig_split_direction.py` (use `/Users/s3057498/code/ECS8056_Lewis/.venv/bin/python`,
  matplotlib; read `evidence/one_at_a_time.csv` and the ledger's §11 table) and include the PDF at column width. The
  specification curve then appears in §V as two sentences plus App. E (design), with one `\pending{B1}`.
- **What transfers** (R1 edit 6): one short paragraph. Structural findings hold for any model by construction (sign
  and frame conventions, naive mirror scoring, the label-swap invariance and placebo failure of sign agreement, the
  edge-token fold for binned tokenisers); empirical ones are model-specific in size (precision, readout, threshold,
  wording, prefix). Flow-matching/diffusion heads remove the token map, bins and prefix but keep frames, signs, zero,
  mirror scoring, statistics and wording, and turn the readout into a sampling choice. Closed-loop twin tests inherit
  the statistic pitfalls. Reporting cost: logging the full 7×256 action distribution costs 3.5 kB per prediction in
  float16.
- **Table I readable at a glance** (R1 edit 7): add a Verdict column (toward / away / null for the direction
  statistics), add the `gpu` row, mark A100 rows; keep it to one `table*`.
- **Reporting card checkable** (R1 edit 8): add a "how to check" column or clause per item; map the action frame to
  *kinematics*, horizon and 5 Hz to *controller gains and rates*, the evaluated scene set to *reset distributions*;
  only the decisive statistic needs a new "evaluation protocol" category. To pay for it, delete the `\report{}` lines in
  §III–IV (Table II carries the same content).
- R1 edits 10–15 and R2 MINOR items as space allows (mirror paragraph: give the plain rule and why naive scoring is
  natural, restore the out-of-distribution clause; dx paragraph: why the error is natural — image x is horizontal,
  robot base x is forward, Bridge documents neither, Open X-Embodiment does not align frames; define terms before use;
  Fig. 1 shows the stimulus logic; hardware row as a headline negative result if the `gpu` row shows no verdict change).
- Positioning (R1 W8): one sentence on how we differ from vla-eval / IndustrialVLA-Bench (a language verdict within one
  model vs closed-loop success across models; the pitfall/fork split).
- App. D gaps: fill from ledger §11 or drop the absent-noun/move rows; no "--" cells.
- Prefix wording (R2 R-M7): "account for 82% of the summed |Δdy|", not "carry". Wording paragraph (R-M11): R2's
  non-causal replacement. 4-bit replication wording (R-M5): R2's replacement.

## C. Process
- Every new number must be in the ledger (§11) — if something you need is missing, write `\note{ledger: ...}` rather
  than computing it yourself.
- Rebuild with `./build.sh`; the main text must still end on page 4; no undefined references; anonymity grep clean.
- Update `claims_trace.md` (new numbers → ledger §11 items) and rewrite `open_questions.md` to list only what remains
  open (expected: B1/B2 timing, whether to keep the bf16 reference given that its signed effect is fragile, title).

## D. Corrections from the ledger's §11 (evidence agent, after these instructions were written)
- GPU-only comparison (§11.1): the model stack (torch, transformers, bitsandbytes) is identical, but Python differs
  (3.12.13 vs 3.12.14) and numpy/Pillow/torchvision/CUDA library versions on the A100 run are unknown. Say "GPU and
  minor, partly unlogged software differences; identical model stack". No verdict changes (5.1% vs 6.1%; +4.1, p = .55
  both; splits 6/19 vs 7/19; 8.1% of lateral tokens; r = .968).
- /255 reference (§11.11): the `zero` row changes (10 predictions sit exactly at the new threshold) and the mirrored
  signed median becomes ∓0.49 (was ∓0.48); no verdict changes. Use the regenerated table.
- Units (§11.2): "metres" is inferred from magnitudes, not documented — hedge ("assuming the dataset's metres").
  Word effects are 3.5%, 8.1% and 11.2% of the median first command (4.29 bins); the positive control 31.6%.
- Multiplicity (§11.13): the bf16 original-wording signed effect survives only the scene-level within-run correction
  (.041), not the frame-level one; neither highlighted bf16 contrast survives any family. The reference run's reversed
  direction for the original wording is therefore fragile — say so, and lead with table side and mirrored images.
- Argmax bin 128 (§11.12): 59/680 (8.7%) baseline argmax predictions sit on the physical-zero bin; counting decisions
  from it lowers the opposite-scene image-right share from 72% to 63%. Use in App. B (zero point under argmax).

---

# Round 3 (Sat 3 Oct, morning): fold in the new experiments

Source of every new number: ledger §12 (`evidence/claims_ledger.md` §12.1–12.6, with its recommended wordings).

**Main text**
1. **§V "The curve"** (B1 stage 1, §12.5). Replace the pending sentence with the result, in 3–4 lines:
   - 16 facets and 256 specifications;
   - no specification shows reliable selection: the upper 95% bound of opposite both-correct is at most 39%;
   - wording explains 42% of the variance of the estimated effect; readout, precision and subset explain 4%, 3% and under 1%;
   - every "the left {noun}" facet points toward the named twin, three of four table-side facets point away, and the original wording points away only in bf16.

   Fold or trim the "Fragile" paragraph if the curve now says the same thing.
2. **Introduction.** The novelty sentence becomes "the first audit of the measurement layer beneath a VLA language verdict and, to our knowledge, the first specification-curve analysis of a robot-learning evaluation (pre-declared; Sec. V)". Contribution (ii) becomes "a fork analysis and a pre-declared specification curve faceted by estimand".
3. **Abstract.** One sentence on the curve, e.g. "A pre-declared specification curve over 256 analyses agrees: wording explains 42% of the variance in the estimated word effect, and no analysis shows reliable selection." Stay at or under ~170 words; cut elsewhere if needed, for example the "release" clause or "Only two pitfalls break the layout effect".
4. **§III, direction-blind statistic** (O4, §12.4). Add one sentence. A placebo pair that names the same twin reproduces the original wording's contrast (+12.2 vs +12.9 points; difference +0.7 [−8.6, +9.6]) but not that of "the left {noun}" (−1.1 vs +16.4; +17.5 [+7.7, +27.5], p = .001). The latter's small effect is therefore separation toward the named twin, not a word→direction shortcut. Make clear this is on the A100 (bf16) runs.
5. **§IV, prefix paragraph** (B2, §12.6). Replace the association-only text and `\pending{B2}` with the result, using the ledger's recommended wording. Holding the earlier-decoded dx token fixed keeps every wording's sign and does not shrink the effect. For the original wording the dx path runs the other way, so its total effect is null while its direct effect is reversed: the estimand decides the verdict. Retag the paragraph as "(action space; direction)".

   Fill Table I's `prefix` row, or give the two §12.6 Table-I rows (A100) as a small inline comparison if the Table I columns do not fit. Its comparator is the A100 own-prefix row, not the GH200 reference; say so in a table note.
6. **§IV, precision and GPU paragraph** (§12.3). Merge the GPU sentences: changing only the GPU (A100 vs GH200) changes 6–8% of lateral tokens (bf16 and 4-bit; r ≥ .97) and no wording's direction, but moves the original wording's signed effect from p = .011 to .07 in bf16.
7. **Table II.**
   - Item 4: the B2 result in a few words (forced-prefix decode; the sign is kept).
   - Item 6: the bf16 GPU result.
   - Item 8: "lower-casing, as OpenVLA's training does, changes 11% of lateral tokens in affected prompts and no natural-frame conclusion" (§12.2).
   - Item 12: done (B1).
8. **§VII Limitations.** The prefix is now measured. Still not measured:
   - pre-processing (B5);
   - screened-out scenes (B6);
   - annotator κ (two annotators, in progress);
   - other precisions and kernels (B4).

**Appendix**
- **App. E.** Replace the pending banner with the stage-1 results:
  - the universe as run;
  - the 16-facet table (Z, Holm p, direction; for argmax facets with a median of 0, the Stouffer sign, plus the post hoc note);
  - the Shapley shares and the O3 bound;
  - `figures/fig_spec_curve.pdf` at full width, with a caption explaining the top panel (facet Stouffer Z, filled = Holm p < .05, circles = expected value, squares = argmax) and the bottom panels (O2 per specification for bf16 expected value; dashboard = threshold 0/1/2 and subset a(ll)/d(etected)/n(ot relabelled)/b(oth)).
  - Say stage 2 (O4 facets, the A100 facet, prefix facets, NF4 prefix) is pending.
- **O4 table** (§12.4), placed in App. E or App. C.
- **Prefix decomposition table** (§12.6, original and mirrored), with the gate facts.
- **App. A runs:** the A100 runs (bf16, batch 1, TF32 off, folded readout, `--save-dist`), and the termitech data-integrity note in one sentence (a truncated input file was replaced before the reported runs; no reported number used it).

**Process:** as before (build; page check; anonymity grep; `main.log`). Update `claims_trace.md` for every new number, from ledger §12. The main text must still end on page 4; pay for additions with cuts.

---

# Round 4 (Sat 3 Oct, afternoon): fold in B4, B5 and B6

**Sources.** Ledger §13.1 (B5, pre-processing), §13.2 (B6, screened-out composites) and §13.3 (B4, compute ladder), with their recommended wordings. All are A100 runs compared with B4.1 (A100 bf16); say so wherever a number appears. Also apply the orchestrator edits after v3 that are already in `main.tex` (claims_trace §G).

**Main text**
1. **§IV, precision and GPU.** Make the point that "4-bit" is not one setting. OpenVLA's own 4-bit default (FP4) changes 67% of lateral tokens and NF4 58%; int8 changes 32%; fp16, fp32 and an attention-kernel change about 7%, the same as a GPU change. Repeats are bit-identical. No wording changes direction in any setting, and only the original wording's verdict moves. Keep it within the current paragraph length if possible.
2. **Observations.** Add one or two sentences to §IV, or to the screening paragraph if more natural, with ledger §13.1's recommended wording:
   - OpenVLA's own evaluation pre-processing changes 60% of actions, and the training-like approximation 68%;
   - no wording changes direction;
   - the training-like path enlarges every word effect about fourfold, while opposite both-correct stays at or below 28%.

   Tag this choice "observations; strength".
3. **Screening.** Replace `\pending{B6}` with §13.2's result. Fill Table I's `screen` row: approved vs rejected composites scored by recorded arrangement on the A100, with that comparator stated in the table note.
4. **§VII Limitations.** Pre-processing, screened-out scenes and other precisions are now measured; remove them. Still pending:
   - annotator agreement (two annotators, in progress);
   - the double-coded reporting audit.
5. **Abstract.** Optional, only if it fits at about 170 words: one clause saying that the observation and compute pipelines change most individual actions without changing any wording's direction.
6. **Table II.**
   - Item 6: the compute ladder.
   - Item 7: B5.
   - Item 10: B6, with κ still pending.

**Appendix.** One compact table each, or one combined table, for B4 (§13.3), B5 (§13.1) and B6 (§13.2), plus a sentence on how B5's two paths were computed. TensorFlow ran in a separate environment; the bridgeorig path's 256×256 resize method is an assumption.

**Process.** As before. The main text must still end on page 4. Update `claims_trace.md`. Final message: at most 200 words.
