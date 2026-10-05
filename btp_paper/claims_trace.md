# Claims trace for `main.tex`, draft v3.2 (3 Oct 2026, with the round-2 results)

Every number and factual claim in the draft, in order of appearance, with its source. Re-checked against the ledger after §11 was added; every number added in v3 comes from ledger §12 and is marked **(v3)**. Section F lists where v3 departs from the Round 3 instructions, and why; §G is the orchestrator's edits after v3; **§H traces every v3.2 change (ledger §13) and lists v3.2's departures.** A script check (v3.2) matched all 759 numbers in `main.tex` (main text and appendix; layout parameters excluded) to the ledger, `one_at_a_time.{md,csv}`, `brief.md`, `analysis_plan.md`, the lit reports, the plan or `refs.bib`, verbatim or as a rounding of a source value. Unmatched: 2.43 (paraphrase floor in grid steps, the orchestrator's V-m2 fix, §E); .711 (Table I `screen` row, from `round2.json`, §H); 3.36, 3.53 and 5.76 (NF4 marginal direct effects in the prefix table, from `prefix_nf4.json`, added by the orchestrator, §G).

**Source codes**
- **L§n:** `evidence/claims_ledger.md`, section n.
  - **L§11.k:** the revision-round addenda (data in `evidence/addenda.json`, key `Ak_…`).
  - **L§12.k:** the 3 Oct results (X1, X2, B0, B1 stage 1, B4.1, B2). Data: `evidence/a100.json` (§12.2–12.4), `outputs/btp/spec_curve.json` (§12.5), `evidence/prefix_bf16.json` and `evidence/prefix_table1_rows.json` (§12.6).
  - **L-G1 … L-G5:** the ledger's "Read first" findings.
  - **L-conv:** its Conventions block. **L-data:** its Data block. **L-inv:** its inventory section.
- **T1:row:** `evidence/one_at_a_time.md`, regenerated with one bin = the /255 grid step, so the old `w255` row is now `w254` and there is a new `gpu` row.
  - **CSV:row:** the extra columns of `one_at_a_time.csv`: McNemar pairs and discordant pairs, same-side both-correct, and the `cmp_*` columns of row `gpu`.
- **lit-A / lit-B / lit-C:** `lit/lit_vla.md`, `lit/lit_methods.md` and `lit/lit_substrate.md`, with the bib key. **E#** is a verbatim-quote tag in lit-B §(b2).
- **brief §8 corr.:** the orchestrator's 18:45 correction in `brief.md` §8 (frames, composites, OWLv2).
- **R1 / R2:** `reviews/r1_workshop.md` and `reviews/r2_rigour.md`; R1's code checks are in its Part 2 §3.
- **plan:** `docs/btp_experiments_plan.md` (v2). **instr.:** `reviews/revision_instructions.md`; **instr. R3** is its "Round 3" section.
- **design:** `analysis_plan.md` (frozen copy `analysis_plan_frozen_2026-10-03.md`), including its "Post hoc changes".
- **CSV meta:** the per-prediction metadata columns of the A100 run files `outputs/btp/a100/*.csv` (`gpu_name`, `torch`, `transformers`, `bitsandbytes`). Not in the ledger; flagged in §D.
- **code:** `scripts/run_gpu.py` and the §12 evidence scripts. Not in the ledger; flagged in §D.

**Run names**
- **Reference:** the bf16 replay on a GH200, folded expected value (`cf1`), one bin = the grid step. This is T1:ref.
- **4-bit logs:** NF4 logged on an A100, 255-token readout (`c1`). This is T1:nf4log.
- **4-bit (GH200):** NF4 replay or ladder on a GH200, `cf1`. This is T1:nf4 and the 4-bit columns of L§9 and L§11.7.
- **Mirrored run:** bf16, mirrored images. These are T1:mir and T1:mir_naive.
- **A100 bf16 runs (v3):** the 3 Oct runs on A100 80GB PCIe GPUs: replay, ladder with a placebo pair per wording, prefix control, and natural frames in two text cases (bf16 and NF4). L§12.

---

## A. Main text

### Title, footnote, abstract

| Claim | Source |
|---|---|
| Title "Same Outputs, Opposite Verdicts: Pitfalls and Forks Beneath a VLA Language Evaluation" (v3; two lines without the manual break, checked in the build) | S's choice (coordinator, 3 Oct). Opposite verdicts on identical outputs: T1:mir vs T1:mir_naive; L§11.6 (label swap); L§11.10 (placebo). |
| Footnote, companion paper | brief §5 (verbatim). |
| 340 twin scenes | L§7. |
| Identical outputs: naive scoring 21 of 25 splits toward, correct 4 of 25 | T1:mir_naive, T1:mir; L§6. |
| A direction-blind contrast scores two same-twin instructions as high as left vs right | L§11.10 (+11.7, p = .012 vs +10.7, p = .039). |
| Reading the wrong axis makes the stimuli seem to fail | L§1: bf16 replay +1.0 pt, p = 1.0; 4-bit logs 86.1% vs 85.7%, p = 1.0. |
| Two conclusions survive every one-at-a-time variation: no reliable selection; mostly the same way | L "Which choices move which conclusion" (≤ 20.4%; 69–92%); T1 all rows. |
| Fragile: direction of the residual word effect; the contrast's significance ("set by the wording" cut in v3: the next sentence carries the wording) | L§9; L "Sign-agreement contrast … most fragile"; L§11.4, L§11.13. |
| **(v3)** A pre-declared specification curve over 256 analyses agrees: wording explains 42% of the variance in the estimated word effect; no analysis shows reliable selection | L§12.5: 64 O1 + 192 O2/O3 specifications = 256; Shapley share of wording 0.42; O3 upper bound ≤ 39.2% < 50% in every specification. Wording: instr. R3 item 3. Abstract is 170 words. |
| Card mapped to the checklist's announced categories (the "release" clause and "Only two pitfalls break the layout effect" were cut for the curve sentence, as instr. R3 item 3 allows; the release stays in contribution (iv)) | lit-B E75 `park2026everything`. |

### §I Introduction

| # | Claim | Source |
|---|---|---|
| 1 | Results rest on choices beneath the policy | lit-B E75 `park2026everything`. |
| 2 | OpenVLA-7B; an object and its mirrored copy (twins); real BridgeData V2 frames | `kim2024openvla`; brief §8 corr. (mirrored cutout); `walke2023bridgedata`. |
| 3 | For token-based policies the readout is part of the action space … | brief §2 "Lesson"; R1 edit 12 (moved to ¶1). |
| 4 | "bf16 unless marked" | brief §5. Every hook number is bf16 except the 4-bit statement, which is marked. |
| 5 | Definitions of opposite scenes and splits | L-conv; R2 R-m17. |
| 6 | Mirrored images: correct scoring 4 of 25 toward (p = .0009); naive gives 21 of 25 on the same outputs | T1:mir (p = 0.00091), T1:mir_naive; L§6. |
| 7 | "as mirror augmentation negates actions" | lit-C A6 `zhuang2025mirrorduo` (mirrors "images, proprioception, and actions"). |
| 8 | Direction-blind contrast gloss; placebo "grab"/"take" +11.7 points (p = .012), as high as left vs right | L§11.10; T1:ref (+10.7). |
| 9 | Wrong axis: stimuli seem not to work (p = 1.0) | L§1, bf16 replay row (+1.0 pt, Fisher p = 1.0). |
| 10 | Wording sets the direction; 4-bit changes 58% of lateral tokens but no headline verdict | L§9; L§4 (58.3% of 2,720; no headline change). |
| 11 | Pitfall and fork definitions; "can silently reverse or erase" | brief §2; instr. A6; R1 edit 5; R2 R-m39. |
| 12 | Estimand | lit-B E24 `lundberg2021what`. |
| 13 | Specification curve as a factorial ablation in sorted order | lit-B E4–E8 `simonsohn2020specification`; R1 edit 5 gloss. |
| 14 | Direction, strength or scope; selection vs direction | brief §1 (the call); instr. A2; R1 edit 1. |
| 15 | LLM evaluations: prompt format, first-token readouts, quantisation flips | lit-B E76 `sclar2024quantifying`, E84 `wang2024answer`, E89 `dutta2024accuracy`. |
| 16 | Undocumented pipeline conventions move VLA closed-loop success | lit-A C4: `choi2026vlaeval` (up to 55 points) and `islam2026deployment` (INT8 export was FP32); lit-C A7: `xu2026vlaquantbench` (omitted hooks silently change quantisation). IndustrialVLA-Bench removed from this list (R2 R-m4). |
| 17 | Closed-loop instruments can certify absent language following | lit-A C2 `kirouane2026measuring`. |
| 18 | "Unlike these audits, we study a language verdict within one model" | R1 W8; lit-A (e) #4. |
| 19 | STAGE reads OpenVLA's single-step action; direction-blind and direction-aware scores differ; what we add | lit-A (b) and (e) #1 `jin2026stage`. |
| 20 | **(v3)** Novelty: "the first audit of the measurement layer beneath a VLA language verdict and the first specification-curve analysis of a robot-learning evaluation (pre-declared; Sec. V)" | instr. R3 item 2. The sentence keeps its leading "To our knowledge, as of early October 2026", which covers both claims, so the second "to our knowledge" of the instruction is not repeated. The curve exists: L§12.5. No precedent: lit-A (d); lit-B §(d) G1 (App. G: "We found no application to robot learning"). |
| 21 | **(v3)** Contribution (ii) "a fork analysis and a pre-declared specification curve faceted by estimand"; release at an anonymous URL | instr. R3 item 2; brief §5. |

### Fig. 1

| Element | Source |
|---|---|
| Pipeline stages; dx decoded before dy | brief §7; L§5. |
| Stimulus pictograms: opposite (twin, gripper, twin) and same-side (gripper, twin, twin) | Design: workshop_plan_v2 §2; R1 edit 13. |
| Pitfall labels: tok, w254, dx/sign/zero, mir_naive, direction-blind contrast | T1 rows; L§11.6. |
| Fork labels: w\_\*, images, scenes, nf4, gpu, prefix, argmax, thr0/thr2 (v3: the prefix asterisk and "pending" removed; `prefix` is now a row of Tables I and IV) | T1 rows; L§12.6. |
| Layer band | brief §3. |

### §II The probe

| # | Claim | Source |
|---|---|---|
| 1 | One token per dimension, autoregressive, dx before dy | lit-C A2.7 (`kim2025finetuning` quote); L§5. |
| 2 | Greedy decoding is the default in the released code | lit-C "Read this first" #5, A2.6; R2 R-m14. |
| 3 | One bin, the grid step, is 3.2×10⁻⁴ action units; about 0.3 mm per 0.2-s step if metres | L§11.2 (recommended wording; 3.2376×10⁻⁴; units hedged). |
| 4 | 5 Hz control | L§11.2; lit-C A4.4 `walke2023bridgedata`. |
| 5 | Median first lateral command 4.3 bins | L§11.3 (4.29 in 0.000325 bins; 4.31 in grid steps). |
| 6 | Two instructions differ by a median of 2.4 bins; paraphrases 1.4 | L§11.10 (folded bf16: 2.42 vs 1.38 in /254 bins; 2.43 vs 1.38 in grid steps, as printed). |
| 7 | 640×480 frames from the Open X-Embodiment release | brief §8 corr.; instr. A1; `oneill2024open`. |
| 8 | An object's own pixels are cut out, mirrored and pasted | brief §8 corr.; R1 code check (`compose_scenes.py`, `mirror=True`). |
| 9 | Logic of opposite and same-side scenes | workshop_plan_v2 §2. |
| 10 | 340 scenes from 191 base frames; 93/151/96 | L§11.8 (191 frames); L§7 (93/151/96). |
| 11 | Reference values | L Table 1 reference definition (regenerated, /255). |
| 12 | The expected value has no ties; argmax ties in 30% of pairs | L§2 (0 ties); T1:argmax (30% ties). |
| 13 | "Decided" is a deadband | instr. A8. |
| 14 | Headline rates defined | R2 R-m17. |
| 15 | The signed median is positive toward the named twin's side under grounding and under a word-to-direction shortcut | R2 R-B3 (`reanalysis.py` l. 203–205 comment); instr. A3. |
| 16 | p-values in the text come from sign-flip tests by base frame | L§11.9. |
| 17 | The 4-bit A100 logs replicate the headline rates and carry only a 255-token readout | L-data; L-G1; R2 R-M5 wording. |
| 18 | Same sign 81.2%; both correct 6.2% (7/112); image-right 44/71/99 | T1:ref (0.0625 × 112 = 7). |
| 19 | Positive control: median +1.4 bins, 73.2% of 149 frames, p = 5.6×10⁻⁷ | L§1, bf16 replay row. n = 149 from L§1 D, confirmed for bf16 in `evidence.json`. |
| 20 | 32% of the median command | L§11.3 (31.6%). |

### Table I

**Cells.** Every cell is copied from the regenerated `one_at_a_time.md`:
- rows ref, dx, sign, zero, mir_naive, argmax, thr0, thr2, nf4, mir, w_prenominal, w_object, w_table_side and sub_det;
- the `zero` row's new values (24/57/99, +12.0 (.012), 7/21) per L§11.11;
- mir and mir_naive signed ∓0.49, per L§11.11;
- the `gpu` row: A100 values from T1:gpu, and the GH200 comparator from the CSV `cmp_*` columns (also L§11.1).

**Columns.** Rows tok and w254 and the same-sign column moved to Table IV (R1 edit 7). The pending row screen moved to Table IV (instr. A4).

**(v3) Prefix rows.** L§12.6 "Table I rows for the prefix" (`prefix_table1_rows.json`), original wording, original images, A100 bf16, folded readout, ≥ 1 grid step:
- `prefix` (dx fixed at the term-free token): 4.2 (n = 118), 42/74/98, +4.7 (.29), 5/12 (.77), −0.02 (.010); verdict away.
- comparator sub-row "vs reference settings on an A100" (own greedy dx): 7.1 (n = 113), 42/71/99, +10.5 (.008), 8/20 (.50), −0.06 (.070); verdict null. These equal the A100 column of L§12.3 (the reference analysis on an A100).
- The p-values .50 and .77 are printed to two decimals, as the ledger gives them.
- Changes tag "strength": no statistic changes sign (signed −0.06 → −0.02; splits 40% → 42% right way; contrast +10.5 → +4.7); the signed effect's significance changes (.070 → .010). See §F on the instructed "direction" tag.
- Caption note §: "A100, bf16: compare with the row below it (own dx), not with the reference" (instr. R3 item 5; L§12.6 "Compare the two rows with each other, not with the GH200 reference").
- The main text cites these rows for the bf16 GPU result too: the comparator row is the reference analysis on an A100 (L§12.3).

**Changes tags**
- Definitions: R2 R-m38.
- dx is tagged strength: R1 §2.4, R-m38.
- sign, mir_naive, w_prenominal and w_object are tagged direction; zero, argmax, thr0, thr2, nf4, mir, w_table_side and sub_det are tagged strength: L "Which choices move which conclusion".
- gpu is tagged none: L§11.1 ("No verdict changes").

**Verdict column**
- Each verdict is the sign of the signed-median column at p < .05; argmax is "away" by its rank-biserial (−0.16, p = .038). (v3) prefix: away (p = .010); its comparator: null (p = .070).
- ‡ marks a row with a test that survives Holm over 48 tests (L§11.4): mir_naive, mir, w_prenominal and w_table_side. The A100 prefix rows are outside that family (App. C says so).

**Caption values.** "1 bin ≈ 0.32 mm" is L§11.2 (0.324 mm if metres). † is T1:argmax.

### §III Pitfalls

| # | Claim | Source |
|---|---|---|
| 1 | A wrong branch is not a "reasonable specification" | lit-B E4 `simonsohn2020specification`; R2 R-m7. |
| 2 | Mirror rule; grounding codebases swap the words | lit-C A6 `kamath2021mdetr`, `deng2021transvg`. |
| 3 | Mirror augmentation negates the demonstrated lateral action | lit-C A6 `zhuang2025mirrorduo`. |
| 4 | Naive negation exchanges right-way and wrong-way splits | L§6 "Inverts, precisely". |
| 5 | 4/25 vs 21/25, both p = .0009, both surviving Holm (v3 rewording: "turns 4 of 25 right-way splits into 21 of 25") | T1:mir, T1:mir_naive; L§11.4 (adjusted .039 each). |
| 6 | Signed −0.49 → +0.49 bins | T1 (regenerated); L§11.11. |
| 7 | A significantly reversed result is scored as significant grounding | L§6 recommended wording. |
| 8 | Mirrored robot images are out of distribution; exploratory | lit-C A6 MirrorDuo; instr. A6 (Type U). |
| 9 | Sign agreement is unchanged under the label swap, which turns grounded into reverse behaviour | L§11.6. |
| 10 | Akin to a Type S error | lit-B E29 `gelman2014beyond`; R2 R-m8. |
| 11 | 4-bit GH200, table side: +14.8 (p = .004; Holm-adjusted .017); signed −0.53, p < 10⁻⁴ (v3: moved up to illustrate the Type S point) | L§8; L§11.13 (c) (.017, five alternatives); L§11.9 (frame p at the floor). |
| 12 | Placebo +11.7 (McNemar p = .012) in the reference run | L§11.10. |
| 13 | Left/right +10.7 (p = .039 uncorrected) | T1:ref; L§11.13 ("survives no family"). The v2 clause "whose signed effect is reversed (−0.15 bins, p = .015 uncorrected)" was cut in v3 for space; Table I keeps the signed effect. |
| 14 | **(v3)** On the A100 runs (bf16; unpaired rates): a placebo pair naming the same twin reproduces the original wording's contrast (+12.2 vs +12.9 points; difference +0.7 [−8.6, +9.6]) but not that of "the left {noun}" (−1.1 vs +16.4; +17.5 [+7.7, +27.5], p = .001) | L§12.4 table and recommended wording; "unpaired": L§12.4 Definition; A100 bf16: L§12.4 heading; instr. R3 item 4. |
| 14a | **(v3)** "whose effect therefore reflects separation toward the named twin, not a word-to-direction shortcut" | L§12.4 Reading (a shortcut would give O4 < 0; O1 is positive); instr. R3 item 4. "small" dropped: see §F. |
| 15 | Image x is horizontal; the model's dx reverses under a flip in 25.9% of decided frames, dy in 65.6% (4-bit logs) | L§1 axis identification (dx 25.9% of 85 pairs; dy 65.6% of 64 pairs); R2 R-m42 (4-bit logs); R1 edit 11. |
| 16 | No Bridge document defines frame or sign; OXE does not align frames | lit-C "Read this first" #4, A4.7, A3 (`walke2023bridgedata`, `oneill2024open`). |
| 17 | dx: 82/82/78; stimulus check +1.0 pt, p = 1.0 | T1:dx; L§1, bf16 replay row. |
| 18 | Sign flipped: 56/29/1 | T1:sign. |
| 19 | Normalised zero (−1.3 bins): opposite lean 71% → 57% | L-G4; T1:zero (regenerated, L§11.11). |
| 20 | 98 single-object frames: 96.9% along dy under our convention; dx at most 57% | L§1. |
| 21 | ECoT's code agrees | lit-C A3 `zawalski2024robotic`. |
| 22 | 256 action tokens decode to 255 centres between the 1st and 99th quantiles; 31744 folded | lit-C "Read this first" #1, A2.3–A2.4; lit-B E60b `kim2024openvla`; R2 R-m15. |
| 23 | Token map: 1.2% of 27,966; 0.995 → 0.30; 68%; over 99.7% | L§3 (recommended wording). |

### §IV Forks

| # | Claim | Source |
|---|---|---|
| 1 | Type N: readout, prefix, precision, wording; Type E: threshold, subset; Type U: mirrored images, screening | lit-B E19–E21 `delgiudice2021travelers`; instr. A6; plan B0. |
| 2 | Greedy decoding executes the argmax bin | brief §2; lit-C A2.6. |
| 3 | Argmax: both-correct 8.3%; contrast p .039 → .23; signed median 0, rank-biserial −0.16, p = .038 | T1:argmax; L§2. |
| 4 | **(v3)** Prefix paragraph tagged "(action space; strength)"; "The model decodes dy after its own greedy dx token" | L§5; lit-C "Read this first" #5. Tag: see §F (instr. R3 item 5 asked for "direction"). The v2 association (59.7%; 3.8 vs 0.95 bins; 82%) moved to App. C. |
| 5 | **(v3)** Holding the earlier-decoded dx token fixed (A100, bf16; means) keeps the sign of every wording's effect, at a similar or larger size ("the left {noun}": direct +4.7 vs total +5.1 bins; table side −6.1 vs −3.4) | L§12.6 table (+4.69 vs +5.08; −6.11 vs −3.39; object +5.04 vs +4.70; original −2.85 vs −0.85) and recommended wording. "at a similar or larger size" replaces the ledger's "does not shrink it": see §F. Bins = grid steps. |
| 5a | **(v3)** Original wording: total null (−0.9 [−3.4, +1.4]), direct reversed (−2.9 [−4.4, −1.4]); the dx path runs the other way; the estimand decides the verdict (row prefix) | L§12.6 table (−0.85 [−3.37, +1.43]; −2.85 [−4.43, −1.44]; indirect +2.00) and recommended wording. |
| 6 | 4-bit more than halves memory | lit-C A1.7 `kim2024openvla`. |
| 7 | OpenVLA's 4-bit result: different checkpoint, unnamed type, FP4 by default (our reading) | lit-C "Read this first" #2, A1.9, A2.12. |
| 8 | We ran NF4 with bf16 compute | L-data; L§4; lit-C `dettmers2023qlora`. |
| 9 | 58% of 2,720 tokens; no headline verdict changes; opposite rates ≤ 2.5 points (v3: the "no headline rate > 8.3" clause cut for space; App. C keeps it) | L§4. |
| 10 | Original wording's contrast p .55 → .039 (uncorrected) | L§4; T1:nf4, T1:ref. |
| 11 | **(v3)** Merged GPU sentence: changing the GPU (A100 vs GH200; same model stack and readout) changes 6–8% of lateral tokens (bf16 and 4-bit; r = .97–.98) and no wording's direction, but moves the original wording's signed effect from scene-level p = .011 to .07 in bf16 (rows gpu, prefix) | bf16: L§12.3 (replay 6.3%, r = .980; ladder 7.5%, r = .976; no wording changes direction; −0.15 (p = .011) vs −0.06 (p = .070)). 4-bit: L§11.1 (8.1%, r = .968). instr. R3 item 6. "r = .97–.98" instead of the instructed "r ≥ .97", because .968 < .97 (§F). "scene-level": `a100.json` `signed_p` for the GH200 is .01146, Table I's Wilcoxon p over scenes. "same model stack": L§11.1 (4-bit); CSV meta (bf16 A100). The 4-bit run's partly unlogged software differences stay in App. A. Paragraph tag "(hardware; strength)": the bf16 GPU change moves significance (was "strength, none"). |
| 12 | Wording directions consistent in four runs; 60–80%; 12.5–32% and 16–37%; positive or negative in 4 of 4 | L§9. |
| 13–14 | (v3: cut for space; the curve's facet-level Holm tests in §V replace them, and App. D keeps the frame-level counts and the bf16 contrast result) | L§11.9; L§11.13 (c). |
| 15 | Wording is a fork: left↔right changes the action more than a paraphrase; the words have several roles | R2 R-M11 text; L§11.10 (paraphrase floor); L§10. |
| 16 | Bridge annotation protocol quote | lit-C A4.1 `walke2023bridgedata`. |
| 17 | 3,574 instructions; 3,333 destination or direction; 183 starting place; roles predict opposite first motions | L§10 (counts; role statistics 39.3 vs 58.8, 81.8 vs 8.3 in App. F). |
| 18 | A claim made with one template has that template's scope | R1 edit 5. |
| 19 | Threshold 0: contrast p .039 → .86 | T1:thr0. |
| 20 | Screening 54.1% vs 45.8%, p = .0008 (v3: counts moved to App. A only) | L§7. |
| 21 | Fallback in 141/340 (41.5%), both-right 58%; detected gripper 4.5% (3/66) | L§7; T1:sub_det. |

### Fig. 2 (effect figure)

- **Data:** L§11.5 decomposition (x, y and their Wilson 95% CIs), read from `addenda.json: A5_decomposition` by `figures/make_fig_split_direction.py`.
  - The script asserts n, splits and right-way counts against `one_at_a_time.csv`.
  - It also asserts that x·y equals both-correct in every row.
- **Rows shown:**
  - Table I's rows plus the gpu comparator.
  - dx is omitted, because "toward" is undefined on dx (R1 edit 2).
  - zero, tok and w254 coincide with ref (L§11.5).
- **Arrows:** ref → sign and mir → mir_naive, which share model outputs (T1; L§6).
- **(v3) Caption:** "dx and the prefix pair omitted". The figure is unchanged; the A100 prefix pair is not plotted because the ledger gives no Wilson CIs for it, and computing them would add numbers outside the ledger.

### Table II (card)

- **Items:** brief §9, refined.
- **Check column and category mapping:** R1 edit 8.
  - Kinematics, controller gains and rates, and reset distributions are added.
  - Only the decisive statistic and the specification curve are "new: evaluation protocol".
- **Category names:** lit-B E75.
- **TF32:** R1 code check; L§11.1.
- **Item 7 wording:** instr. A1 (R2 R-B1).
- **(v3) Checks:** item 6 "agreement across precisions and GPUs" (L§12.3 adds the GPU check); item 8 "paraphrase floor; lower-casing" (L§12.2). This study's results for items 3, 4, 6, 8 and 12 went into the filled-in card (Table XIII, App. F), not into Table II: see §F.

### §V What survives

| # | Claim | Source |
|---|---|---|
| 0 | (v3) Heading "No selection, in any analysis" | Table I/IV rows (below) and the curve (L§12.5). |
| 1 | Opposite both-correct ≤ 20.4% in every row of Tables I and IV (reference 6.2%) | T1 (maximum w_object 20.4); L "Neither grounded". (v3) The A100 prefix rows: 4.2% and 7.1% (L§12.6). |
| 2 | In every row of Table I, splits occur in 8–27% of decided opposite scenes; the toward share ranges 12.5–84% | L§11.5 "Reading" (x 8.5–27.4%, y 12.5–84%). Scoped to Table I because w_move in Table IV has more splits. (v3) The A100 prefix rows fall inside both ranges: splits 12/118 = 10.2% and 20/113 = 17.7%; toward 5/12 = 41.7% and 8/20 = 40% (L§12.6 counts; arithmetic). |
| 3 | Same sign 69–92% in every row | T1 (min w_move 68.8, max dx 91.5); L "Not lexical". (v3) A100 prefix rows: 1 − splits/n = 89.8% and 82.3% (arithmetic from L§12.6; `a100.json` gives `opp_same_sign` .823 for the own-dx row). |
| 4 | Layout followed except dx and the flipped sign | L "Follows the layout". |
| 5 | (v3, now the last sentence of "The curve") The direction-blind contrast stays fragile: its significance switches with threshold, readout, prefix, precision, images, subset and wording | L, contrast list (thr0 .86, argmax .23, nf4 .55, mir 1.0, sub_det .38, object .61); (v3) prefix .008 → .29 (L§12.6). Zero and thr2 removed (R2 R-m23). |
| 6 | 12 discordant of 75 paired frames | CSV:ref (pairs 75, discordant 12); R2 R-M12. |
| 7 | (v3: the "Fragile" paragraph is folded into "The curve"; its Holm-over-48 sentence is cut from §V because the Table I caption (‡) and App. C carry it) | L§11.4. |
| 8 | **(v3)** The curve fixes every pitfall and crosses wording, readout and precision in 16 facets, with threshold and subset varied within each: 256 specifications | L§12.5 Universe (16 GH200 facets; 64 O1 + 192 O2/O3 specifications); design §3–4. |
| 9 | **(v3)** None shows reliable selection: the upper 95% bound of opposite both-correct is at most 39% | L§12.5 (39.2%, all 192 O2/O3 specifications). |
| 10 | **(v3)** Every facet of "the left {noun}" points toward the named twin and three of four table-side facets point away (Holm p < .05) | L§12.5 table (prenominal Holm p ≤ .046 in all four; table side .006, .002, .003 significant, .16 not). |
| 11 | **(v3)** "argmax facets by the joint test's sign, post hoc" | L§12.5 "Argmax facets" (median O1 exactly 0; Stouffer sign); design, Post hoc changes 1. Added because two of the four "the left {noun}" facets and one of the three table-side facets are argmax facets. |
| 12 | **(v3)** The original wording points away only in bf16 | L§12.5 (Holm .022 and .038 in bf16; .52 and .95 in NF4). |
| 13 | **(v3)** Wording explains 42% of the variance of the estimated word effect; readout, precision and subset 4%, 3% and under 1% | L§12.5 (Shapley shares of OLS R² of the 64 O1 estimates: 0.42, 0.04, 0.03, 0.002). |

### §VI Card

| Claim | Source |
|---|---|
| Mapped onto the announced categories, kinematics to reset distributions; only the evaluation protocol (items 11–12) needs a new category (v3: "Most items cost nothing to report", trimmed) | lit-B E75; R1 W7, edit 8; V-m fix. |
| Logging the 7×256 distribution costs 3.5 kB per prediction in float16 | L-inv (d) ("In float16 that is 3.5 KB per prediction"). |
| Controller gains' configuration "remains largely undocumented" | lit-C A8 `bronars2026tune` (verbatim; R2 R-m1). |
| STAGE, ConflictVLA-Bench and LIBERO-CF do not say whether actions are argmax, sampled or expected values | lit-A §(a) (`jin2026stage`, `hou2026conflictvla`, `fang2026liberocf`). |

### §VII What transfers, limitations, conclusion

| Claim | Source |
|---|---|
| Sign flip and naive mirror negation reverse every directional verdict; sign agreement ignores which instruction is which; closed-loop twin tests inherit all three | By construction: T1:sign, L§6, L§11.6; R1 edit 6; instr. B. |
| The edge-token fold applies to any binned tokeniser | lit-C A2–A3 (binned tokenisers), by construction. |
| Precision, readout, threshold, wording and prefix effects vary by model | instr. B (empirical findings). |
| Parallel continuous decoding and flow heads remove the token map, bins and prefix; flow heads make the readout a sampling choice | lit-B `kim2025finetuning` row; lit-C A3 (π0, π0.5 flow integration); R2 R-m16. |
| Offline single-step readouts forecast closed-loop success poorly | lit-A C4 `li2024evaluating`, `wang2026reali`; R2 R-m9. |
| OXE 640×480 vs the 256×256 release OpenVLA trained on; JPEG round trip and Lanczos resize skipped; not yet measured | brief §8 corr.; R2 R-B1 text. Stated as a limitation without `\pending` (R1 edit 4). |
| **(v3)** Not yet measured: pre-processing, the screened-out scenes, two-annotator agreement (in progress) and other precisions and kernels; the prefix is no longer listed | instr. R3 item 8 (B5, B6, κ, B4); open_questions "Resolved by S" (S is the second annotator); L§12.6 (prefix measured). |
| Conclusion: identical-output choices moved the verdict toward/away, significance/null, working/failing; (v3) "Pitfalls need reporting with their check" (trimmed) | T1:mir vs mir_naive; T1:thr0, T1:argmax vs ref; T1:dx with L§1. |

---

## B. Appendix

### App. A

| Claim | Source |
|---|---|
| 640×480 OXE frames; mirrored cutout; OWLv2 with image-centre fallback (`minderer2023scaling`, arXiv 2306.09683, verified by the orchestrator via the arXiv API) | brief §8 corr.; R1 code check (`detect_duplicates.py`). |
| Fig. 3: example stimuli chosen by rule | `figures/make_fig_examples.py`, from `google_drive/v2/constructed/{evaluation_set,constructed_manifest,constructed_review}.csv`. The hand-label counts (151/96/93; 60 unclear) and 182 fallbacks match L§7. |
| 191 base frames; 149 hold two scenes, 42 one | L§11.8. |
| 151 opposite scenes from 148 frames; the three two-scene frames were relabelled from same-side | L§11.5 (clustering); L§11.8. |
| Positive control on 149 base frames | L§1 D. |
| Screening (1,644 of 3,248; 1,604; 210/437, 437/808, 173/399 with CIs; 45.8% (383/836); χ² = 11.2, p = .0008, Fisher .0009; three-layout p = .0014; 824 = 781 + 26 + 17; 400/60/340; 53 relabelled) | L§7. |
| Fallback (182/400, 141/340; 31/93, 54/151, 56/96; p = .0004; 41 of 60; 199; 287) | L§7. |
| Instruction ladder; the paraphrase pair names the same twin | T1 w\_\* rows; L§11.10 (definition). |
| 4 scenes where the object wording equals the original | L-G5. |
| Prompt template and token 29871 | lit-C A2.8 and A2.4; R1 code check (`model.py:196`, `run_gpu.py`). |
| Twin prompts lower case | brief §8. |
| Model stack identical for the 4-bit logs and the GH200 runs (torch, transformers, bitsandbytes, tokenizers, timm, Hub client, accelerate versions); NF4 settings; eager; seed 42; greedy batch 1 without a mask; TF32 disabled (PyTorch default) | L§11.1; R1 code check. (v3: scoped to those two machines.) |
| GPU, CPU and Python 3.12.13 vs 3.12.14 differ; unlogged A100 versions; Pillow and torchvision perform the resize | L§11.1. |
| 20 duplicates bit-identical | L§4; L-inv (b). |
| **(v3)** Added A100 runs: NVIDIA A100 80GB PCIe; same torch, transformers and bitsandbytes versions, logged with every prediction | CSV meta: all 43,692 rows of `outputs/btp/a100/*.csv` read "NVIDIA A100 80GB PCIe", torch 2.11.0+cu128, transformers 4.40.1, bitsandbytes 0.50.2 (§D). |
| **(v3)** Greedy decoding at batch size 1; folded readout; every prediction's full 7×256 distribution logged (`--save-dist`) | L§12 Inputs ("batch size 1, greedy decoding and the folded readout, and logs full distributions as sidecars"). float16 and 7×256: code (`run_gpu.py` `--save-dist` help text; §D). |
| **(v3)** TF32 disabled on the A100 runs | code: `run_gpu.py` sets `torch.backends.cuda.matmul.allow_tf32 = False` (§D); instr. R3. |
| **(v3)** Data integrity: a truncated copy of one input file on that machine was replaced with a checksum-verified copy before the runs; no reported number used it | L§12 Inputs ("Data integrity": 1,864 of 2,720 rows; replaced, md5-verified; partial first launch deleted); instr. R3. The machine is not named. |
| **(v3)** Table III lower block: replay 2,720; ladder 8,160 + 3 placebo pairs; prefix control (forced dx; folded and argmax); natural frames, 2 text cases, NF4 and bf16, 3,103 each | L§12.3 (2,720; 8,160); L§12.4 (a placebo pair per wording); L§12.6 (prefix); L§12.2 (3,103; bf16 and NF4). Readout columns: CSV columns a*/c*/cf* and the prefix run's ct1*/at1* (§D). The ladder's total row count (12,240) is not printed. |
| **(v3)** "on the A100 it adds a placebo pair in each of the prenominal, object and table-side wordings" | L§12.4 Definition; `btp_runs_a100.py` `PLACEBO` map. |
| **(v3)** 27,966 predictions "across the six GH200 runs" (was "all six GPU runs"); natural frames also enter through the text-case check | L§3 table (replay, ladder and natural, each NF4 and bf16, all GH200); L§12.2. |
| Table III; "Ladder (5 wordings + pair)" = 340 × 6 prompt pairs × 2 × 2; natural frames (3,103) enter only the token-map count and Table VI; 27,966 | L-data; L§3; R2 R-m29, R-m30. |

### App. B

| Claim | Source |
|---|---|
| Token ids 31744–31999; 256 edges, 255 centres; index map; fold | lit-B E60c; lit-C A2.1–A2.4 [derived]. |
| De-normalisation formula; q01 = −0.04170, q99 = +0.04086 | lit-C A2.4; L§11.2; L-G4. |
| Grid step 3.2376×10⁻⁴; /254 width 3.2503×10⁻⁴; they differ by 0.4% | L§11.2; L§11.11 (×1.0038). |
| dx bin 0.000224 | L§11.11; T1:dx. |
| Resolution counts 44.4% (151/340), 32.6% (111/340), 11.8% (40/340) | L§2. |
| Units: action = difference of consecutive end-effector states; unit unstated; q01–q99 spans −4.2 to +4.1 cm; 0.324 mm per step, 1.6 mm/s at 5 Hz | L§11.2; lit-C A2.10. |
| Median \|dy\| 4.3 bins (IQR 1.6–12.0); 84.1% of 680 decided | L§11.3. |
| Effects as 3.5%, 8.1% and 11.2% of the median; positive control 31.6% | L§11.3. |
| Readout definitions; folded EV renormalised over the 256 tokens; mass ≥ 0.959 "in every GH200 run" (v3: scoped, because the ledger reports mass for the GH200 runs only) | L§3; L-data; R1 code check (`run_gpu.py:97–105`). |
| Greedy, unrestricted generation; dy read after dx | lit-C "Read this first" #5; L§5. |
| Zero: −0.000424 (−1.3 bins); bin 128 → −0.31; bin 127 vs 129 (+0.69); 7.2% on bin 127 | L-G4. |
| 59/680 (8.7%) on bin 128; from bin 128, image-right 72% → 63% and contrast +6.9 (.23) → 0.0 (1.0); at 2 steps +11.7 (.039) vs +11.5 (.070) | L§11.12. |
| All tests two-sided | R2 R-M3. |
| 21 reference splits from 21 different frames | L§11.5 (clustering). |
| Wilcoxon drops zero differences (none under EV; 30% under argmax) | L§11.9 ("All 340 differences are nonzero"); T1:argmax; R2 R-m19. |
| Sign-flip test: 20,000 flips, 191 frames | L§11.9. |
| McNemar over 75 paired frames | CSV:ref; L§11.8. |
| Bootstrap 2,000 draws; reference both-correct bootstrap [1.8, 10.8] vs Wilson [3.1, 12.3] | L-conv; L§11.5; R2 R-m41. |
| Holm families | R2 R-M3 wording; L§11.13. |
| Mirror-scoring formulas; dx-row convention | L§6; T1 notes. |

### App. C

| Claim | Source |
|---|---|
| Table IV, every row | T1 (regenerated) plus CSV pairs and discordant counts; gpu GH200 pairs 73 from L§11.1. |
| Pending rows screen, dist (v3: dist's pending text now says the A100 runs log full distributions) | T1:gpu_screen, gpu_dist; L§12 Inputs. |
| **(v3)** Row prefix in Table IV points to Table I (against the A100 own-dx row) and Table VIII | L§12.6. |
| **(v3)** "Row prefix and its comparator are A100 bf16 analyses and are compared with each other" | L§12.6 Table I rows ("Compare the two rows with each other, not with the GH200 reference"). |
| **(v3)** "The A100 prefix rows are not in the family" | L§11.4 (the family is the 16 listed rows). |
| mir_none: same-side both-correct 26.7% and 5.4% vs 48.3% and 77.0% | L§6 table; CSV. |
| Holm over 48 tests: the family; seven survivors with adjusted p (.0006, .013, .034, .034, .034, .039, .039; the mirrored pair is 0.03447); first non-survivor sub_det (raw .0021, adjusted .087); same seven at 42 and 51 tests; no contrast survives | L§11.4. |
| Table V (decomposition, Wilson CIs) | L§11.5 table. |
| Table VI (token agreement); folded replay r = .709 | L§4 table (including "cf1 .709"); R2 R-m27. |
| Table VII (headline by readout) | L§2 "Headline by readout". |
| 5.1% of 680 (4-bit logs); argmax medians (p = .70, 33% ties; p = .038, r = −0.16, 30% ties) | L§2; R2 R-m42. |
| Opposite rates move ≤ 2.5; both-right 90.6% → 98.9% (+8.3) | L§4. |
| dx token differs in 5.0%; dy token then changes in 84.7% (median 6.0 bins) vs 4.1% | L§4 mechanism; L§11.1. |
| **(v3) Prefix control paragraph:** GH200 bf16 replay: dx differs in 59.7% of 340 pairs; dy then differs by a median of 3.8 bins vs 0.95; 82% of the summed \|Δdy\| | L§5, bf16 row (moved here from §IV). |
| **(v3)** Gate: dy after the own greedy dx reproduces `generate()` to 4×10⁻⁶ bins on 10 scene–wording units and to 9×10⁻⁵ bins against the full A100 ladder and replay runs | L§12.6 Method. "scene–wording units": a unit is one scene × wording × image transform with its left, right and term-free prompts (code: `run_gpu.py` `prefix_unit`; §D). |
| **(v3)** Every forced step alone at batch size 1; a batched step changed dy by up to 3.6 bins in bf16 | L§12.6 Method. |
| **(v3)** Means because medians do not add; word × dx interaction between −0.6 and +1.2 bins in every cell | L§12.6. |
| **(v3)** Signs kept on original and mirrored images; on original images at a similar or larger size | L§12.6 table; mirrored: "Each has the sign of its total". The size claim is restricted to original images because the ledger gives no mirrored totals. |
| **(v3)** Original and table-side: the dx path runs against the direct effect (+2.0, +2.7) and partly cancels it; total null, direct reversed; magnitude moves with dx, direction does not come from it | L§12.6 Reading. |
| **(v3)** Fixing dx at the term-free token: signed p .070 → .010; contrast +10.5 → +4.7, p .008 → .29; much of the contrast rides on dx-token changes | L§12.6 Table I rows and their Reading. |
| **(v3) Table VIII (prefix decomposition):** every cell (totals, directs with CIs, indirect, marginal direct, dx differs 60/62/62/62%, splits own/fixed, mirrored directs −6.10/+3.09/+4.52/−9.90) | L§12.6 table and "Mirrored images" line. Caption definitions: L§12.6 Method; `btp_prefix_analysis.py` docstring. |

### App. D

| Claim | Source |
|---|---|
| Significance counts with frame-level p: raw 3, 3, 3, 4, 4, 1; within-run Holm 1, 3, 2, 4, 3, 1; global 1, 2, 2, 4, 3, 1 | L§11.9 counts table (frame columns). |
| Paraphrase floor, folded: bf16 2.43 vs 1.38 (grid steps; 2.42 in /254 bins), 59.1%, p = .001; 4-bit 2.57 vs 1.72, 58.5%, p = .00031 | L§11.10. |
| Table IX in v3, VIII in v2 (all 24 cells: splits with p, both-correct with n, contrast with pairs and discordant, medians, scene and frame p) | L§11.7 and L§11.9. Generated from `addenda.json` (A7_grid cf1, A9_frame_level) and cross-checked against the ledger tables. The caption and V-m2 fix put the medians in grid steps. |
| Table X in v3, IX in v2 (placebo in four runs) | L§11.10. |
| 255-token differences (19/25, 76.0%; 26/33); Holm flip on 4-bit object (.013 vs .031) | L-G1; L§9; L§11.7. |
| Holm on the contrast over the five alternatives: 4-bit prenominal and table side (plus object with c1); 4-bit mirrored prenominal (both readouts); bf16 none | L§11.13 (c); L§9; R2 R-m25. |
| Three cells with a significant positive contrast and backwards splits | L§8. |
| Label swap: contrast +10.7 unchanged (75, 12); splits 7/21 → 14/21; signed ∓0.15; random half: splits 8/21 | L§11.6. |

### App. E (specification curve: design and stage 1)

| Claim | Source |
|---|---|
| **(v3)** Title "Specification curve"; the design was written down and hashed before the curve was computed; a pre-declared analysis, not a pre-registration; one post hoc rule | design header ("Frozen: Sat 3 Oct 2026, before any specification curve was computed"; SHA-256 4b4cbedc… recorded in plan §B0, as L§12.5's heading says); design "Post hoc changes". |
| **(v3)** Stage 1 covers the 16 GH200 facets; stage 2 (O4 facets, the A100 facet, the prefix facets, an NF4 prefix run) pending | L§12.5; instr. R3 (App. E). |
| Claim tested, unit, clustering; O1, O2, O3, the O3 rule; pitfalls fixed (including the bin-128 rule); Type E specifications (12 and 4); /254 left out (0.4%); verdict rules; joint test; display | design §1–6 (plan v2 B0); L§11.12; L§11.11; L§7 (199, 287); R2 R-P1, R-P3, R-P6, R-m37, R-P13. Unchanged from v2. |
| **(v3)** O4 as run: unpaired rates over decided scenes; base-frame bootstrap CI and label-swap permutation test; GH200 original +14.0 and +13.0 vs the paired +10.7 and +11.7 | L§12.4 Definition and table; T1:ref; L§11.10. The v2 estimate "≈ −1 point" is replaced by the as-run value in Table XII. |
| **(v3)** Every wording has a placebo pair in the A100 bf16 run; on the GH200 only the original wording | L§12.4; design §2 (Availability). |
| **(v3)** Facets: stage 2 adds the prefix, reported against A100 own-prefix runs, never GH200 rows; median, mode and sampled readouts pending B9, from the distributions now logged | design §4; L§12 Inputs. |
| **(v3)** Type U adds the A100 facet (stage 2) | design §4 (Type U "GPU"). |
| **(v3)** Variance decomposition indicators: wording, readout, precision and subset (prefix in stage 2) | L§12.5 (Shapley over these four); design §5. |
| **(v3) Stage 1 as run:** 16 facets; 4 O1 specifications each (64) and 12 O2/O3 (192): 256; 10,000 base-frame label flips; Stouffer's Z over the four O1 specifications; two-sided; Holm across 16 | L§12.5 Universe. |
| **(v3) Stage 1 results:** O3 upper bound ≤ 39.2% in all 192 specifications, so no facet selects; facet directions; object toward in all four, one survives Holm; Shapley 0.42, 0.04, 0.03, 0.002; R² 0.49 | L§12.5. |
| **(v3) Post hoc:** every argmax facet has a median O1 of exactly 0; direction also reported as the sign of Z | L§12.5 "Argmax facets"; design, Post hoc changes 1. All eight argmax medians are 0.0 in `spec_curve.json` (§D). |
| **(v3) O4 paragraph:** the original wording's contrast is matched on both GPUs; "the left {noun}" separates beyond the placebo; with positive O1 this excludes a shortcut (O4 < 0); far short of selection | L§12.4 Reading; L§12.5 (O3 bound). |
| **(v3) Fig. 4** (`figures/fig_spec_curve.pdf`) | L§12 Regenerate (`btp_spec_curve.py`). Caption: instr. R3 (top: facet Stouffer Z, filled = Holm p < .05, circles = expected value, squares = argmax; bottom: O2 per specification for bf16 expected value; dashboard threshold 0/1/2 and subset a/d/n/b). "Wilson 95% CIs", "sorted" and the dotted one-half line: the script (`E.wilson`, `sort_values("o2_share")`; §D). |
| **(v3) Table XI (16 facets):** Z per facet; Holm p ("≤ .046" for the four "the left {noun}" facets, as the ledger gives it); direction = sign of Z, starred for argmax | L§12.5 table. "For expected-value facets also the sign of the median O1": checked in `spec_curve.json` (`o1_median_all` has the sign of Z in all eight; §D). |
| **(v3) Table XII (O4):** all cells | L§12.4 table. |

### App. F (Table XIII in v3; Table X in v2)

| Item | Source |
|---|---|
| 1 | L§1 (96.9%, 3.1%, 57%, 56%; 4-bit logs: 65.6% of 64, 25.9%, 16.8%; "from the left" 82% of 22 vs 8% of 24); L§11.2 (units); `zawalski2024robotic`. |
| 2 | lit-C A2.4; L-G4; L§11.2; L§3. |
| 3 | L§2 (32.6%); T1:argmax (30%). (v3) Full 7×256 distributions logged in float16 in the A100 runs, not in the GH200 runs: L§12 Inputs; code (`--save-dist`, float16; §D); v2 item 3 ("not logged"). |
| 4 | L§5; lit-C "Read this first" #5; L§3 (mass). (v3) Forced-prefix decode: L§12.6 (signs kept; marginal direct −2.27/+3.89/+5.57/−4.42 has the same signs; original total −0.85 [−3.37, +1.43] → "−0.9 [−3.4, +1.4]", direct −2.85 [−4.43, −1.44] → "−2.9 [−4.4, −1.4]"). |
| 5 | L§11.2 (5 Hz); `walke2023bridgedata`. |
| 6 | L§11.1 (stack, TF32, Python versions, unlogged versions; v3 attributes Python 3.12.13 and the unlogged versions to "the 4-bit A100 run"); (v3) "bf16 … on an A100 80GB PCIe (added runs)" and "identical model stack on all three": CSV meta (§D); L§4 (58.3%; folded r .709 → ".71"); L§11.1 (8.1%, r = .97, no verdict change); L§4 (20 duplicates). (v3) bf16 GPU comparison: L§12.3 (6.3% of 2,720 replay, 7.5% of 8,160 ladder, r .980/.976 → ".98"; no wording's direction changed; p .011 → .070, Wilcoxon over scenes per `a100.json` `signed_p`); "A100 (80GB PCIe)": CSV meta (§D). Pending B4. |
| 7 | brief §8 corr.; R2 R-B1 text (README "out of date"); L§6 (mirroring). Pending B5. |
| 8 | T1 w\_\*; L§11.10 (paraphrase floor, folded); brief §8 (lower case). (v3) A placebo pair per wording on the A100: L§12.4. Lower-casing: L§12.2 (2,276 of 3,103 prompts change; argmax token changes in 11.2% → "11%", r = .950 → ".95"; NF4 13.0% → "13%"; no natural-frame conclusion changes; A100) and its recommended wording. Pending B5 (appended token). |
| 9 | L§10 (3,574/17,035; 3,333; 183; 2; 56; role-statistic definition: single lateral word, first motion > 5 mm); lit-C A4.1; R2 R-m40. |
| 10 | L§7; L§11.8 (191 frames). Pending B6 (inter-annotator κ, two annotators, in progress: open_questions "Resolved by S"). |
| 11 | L§11.9; L§11.6 and L§11.10 (checks); L-conv; App. B Holm families; L§1 (positive control, bf16). (v3) O4 for every wording on the A100: L§12.4. |
| 12 | T1. (v3) Stage 1: L§12.5 (16 GH200 facets, 256 specifications, a joint test per facet). Stage 2 pending (B1 stage 2). |

### App. G

Each sentence is supported by the lit reports' "what it does" or quote columns, as in v1. Changes in v2:
- "to our knowledge" added to four negative claims (R2 R-m10).
- "nearly two million steering commands" (lit-C A4.13; R2 R-m11).
- QAIL moved to the action-level group (lit-C A7; R2 R-m12).
- CogACT listed among "continuous or flow-based action heads", because lit-C A3 records only that it argues against binning; R2 R-m13 calls it a diffusion module, which no lit report states.
- Kurtic (aggregate parity, E90) separated from Dutta (flips, E89) (R2 R-m3).
- Bronars quoted verbatim (R2 R-m1).
- Action-token probabilities: lit-A (d) #1 and lit-B G6 (`zollo2025confidence`, `jang2026verifier`, `karli2025insight`).

---

## C. Every `\pending{}` and the experiment that fills it (plan v2 IDs), v3.2

There are 9 uses: **1 in the main text** and 8 in the appendix. The 10th `\pending{` match in the file is the macro definition.

| # | Location | Pending text | Filled by |
|---|---|---|---|
| 1 | §VI | B7 count | **B7** (double-coded by two humans), or **B7-lite** reported as preliminary. |
| 2 | Table IV, row `dist` | B9: from the full distributions logged in the A100 runs | **B9**. |
| 3 | App. C, screening paragraph | B6 step 2 | **B6 step 2**: hand check of the recorded layouts of 100 rejected composites. |
| 4 | App. E, opening paragraph | B1 stage 2 | **B1 stage 2**: O4 facets, the A100 facet, the prefix facets, an NF4 prefix run. |
| 5 | App. E, facets | B9 | **B9**. |
| 6 | App. E, Type U | B6 step 3 | **B6 step 3**: reweighting by modelled approval within layout. |
| 7 | Table XVI, item 8 | B5 step 3 | **B5 step 3**: prompt with and without the appended empty token. |
| 8 | Table XVI, item 10 | B6: inter-annotator κ, two annotators, 100 frozen scenes; in progress | **B6 step 4**. |
| 9 | Table XVI, item 12 | B1 stage 2 | **B1 stage 2**. |

**Removed in v3.2, because the result is now in the paper:** §IV `\pending{B6}` (screened-out composites, L§13.2); Table IV row `screen` (B6); App. E Type U `\pending{B6}` (comparison done; the remaining reweighting is now `B6 step 3`); Table XVI item 6 `\pending{B4}` (L§13.3); item 7 `\pending{B5 pre-processing run}` (L§13.1).

**Removed in v3:** §IV B2; §V B1; Table IV `prefix`; App. E B0 banner; App. E O4 (B4.1); App. E facets (B2); card items 3 (B4.1) and 4 (B2).

**Plan items with no `\pending{}`:** X1–X4; B3; B8, B10, B11 (optional).

## D. Facts used that are not in the ledger (flagged)

- **brief §8 correction:**
  - 640×480 frames from the Open X-Embodiment release, resized by the processor to 224×224;
  - OpenVLA trained on the 256×256 `bridge_orig` release; README: OXE version "out of date";
  - mirrored cutouts; OWLv2 with image-centre fallback;
  - lower-cased prompts;
  - OpenVLA's JPEG and Lanczos evaluation path.
- **R1 code checks:**
  - prompt template;
  - token 29871;
  - renormalised folded readout;
  - TF32 disabled (also L§11.1).
- **lit-C, code-derived:** token ids, the 32,000-token vocabulary, the decode formula, and the FP4 default ("our reading of the code").
- **plan v2 B0, design parameters (not results):** 16/24 facets, 12/4 specifications and 10,000 flips.
- **Fig. 3:** stimulus images selected by the stated rule from the local manifest files; no numbers.
- **Not used, because the ledger lacks them** (listed in open_questions.md):
  - a pooled "word across phrasings" estimate (R1 edit 5);
  - a 1-mm (3-bin) deadband row (R1 edit 9);
  - bf16 dx image-right shares (R2 R-m21 gives 83.1% vs 82.1% from `evidence.json`; the main text uses the ledger's +1.0 point).

- **(v3) Facts from files, not from the ledger text:**
  - CSV meta: every A100 row (43,692) reads "NVIDIA A100 80GB PCIe", torch 2.11.0+cu128, transformers 4.40.1, bitsandbytes 0.50.2 (App. A; Table XIII item 6). The paper prints the GPU model and "same … versions", not the version numbers again.
  - `scripts/run_gpu.py`: TF32 disabled (`allow_tf32 = False`); `--save-dist` stores 7×256 float16 probabilities; a prefix "unit" is one scene × wording × transform with left, right and term-free prompts (App. A, App. C).
  - Readout columns of the A100 files (a*, c*, cf*; prefix ct1*, at1*) for Table III's "all three" and "folded, argmax".
  - `outputs/btp/spec_curve.json` checks behind wording, not printed numbers: all eight argmax facets have a median O1 of 0.0; in all eight expected-value facets the median O1 has the sign of Z (Table XI caption). The individual Holm p-values of the "the left {noun}" facets (.0096, .0065, .026, .046) are not printed; the paper uses the ledger's "≤ .046".
  - Fig. 4's Wilson CIs, sorting and dotted one-half line come from `btp_spec_curve.py`.
  - Same-sign rates of the two A100 prefix rows (89.8%, 82.3%) are derived as 1 − splits/n for the §V range claim; they are not printed. `a100.json` confirms .823.
  - "scene-level" for the GPU p-values: `a100.json` `signed_p` for the GH200 equals Table I's Wilcoxon p (.0115).
  - The A100 ladder has 12,240 rows (8,160 + three placebo pairs); the paper prints "8,160 + 3 pairs", not 12,240.

- **(v3.2) Facts from files, not from the ledger text:**
  - `evidence/round2.json` → `b6`: the contrasts (+2.0, p = .711; +6.4, p = .064) and split p-values (.220; .009) in Table I's `screen` pair. §13 names this file as its source. The decided opposite-scene counts behind the §V range checks (271 and 316) come from the same file: splits 54/271 = 19.9% and 54/316 = 17.1%; toward 40.7% and 31.5%. They are not printed.
  - `evidence/prefix_nf4.json`: the NF4 original-wording interaction (+1.85), used only to scope a sentence to bf16. It is not printed.

## E. Orchestrator fixes after the v2 verification (2 Oct 2026, evening)

Source: `reviews/r2_verification_v2.md`.
- **V-M1:** App. F item 9 now states the destination and starting-place role statistics (ledger §10: 39.3% vs 58.8%; 81.8% vs 8.3%).
- **V-M2:** "What transfers" is narrowed.
- **V-m1 to V-m8:** applied as specified.
  - Table VII is in grid steps: five cells changed by +0.01 in magnitude, and the paraphrase floor is 2.43.
  - The Holm value is .034.
- **V-m9:** figure PDFs carry no Creator, Producer or date metadata.
- **R-m2:** the LLM sentence is reworded.
- **"term-free":** defined at its first use.
- **OWLv2:** cited as `minderer2023scaling`.
- **Two cuts for space:** the companion-paper clause in §IV, and a tighter §VII limitations sentence.

## F. Where v3 departs from the Round 3 instructions, and why

1. **Prefix tag "strength", not "direction"** (instr. R3 item 5). Holding dx fixed keeps every wording's sign (L§12.6), and Table I defines "direction" as a sign change and "strength" as a change in size or significance without one. The original wording goes from null to away, which Table I already tags "strength" in the reverse case (ref → nf4, away → null). Tagging the paragraph "direction" would contradict the table's own definitions. Easy to switch back if the coordinator prefers.
2. **"at a similar or larger size" instead of "does not shrink it"** (L§12.6 recommended wording). For "the left {noun}" the direct effect (+4.69) is smaller than the total (+5.08), the very example the sentence cites. The ledger's "at least as large as the total" holds for the other three wordings only.
3. **"small" dropped from "The latter's small effect …"** (instr. R3 item 4). The prefix paragraph reports means of about +5 bins for the same wording, so "small" would read as contradictory; §V carries the no-selection result.
4. **"r = .97–.98" instead of "r ≥ .97"** (instr. R3 item 6). The 4-bit GPU correlation is .968 (L§11.1).
5. **Table II items 4, 6, 8, 12** (instr. R3 item 7). The main-text card keeps generic checks (item 6 now "agreement across precisions and GPUs", item 8 "paraphrase floor; lower-casing"); the study's results (forced prefix keeps the sign; the bf16 GPU result; the lower-casing sentence from L§12.2; the curve done) are in the filled-in card, Table XIII items 3, 4, 6, 8 and 12. Putting them in Table II's narrow check column would have cost about two lines that page 4 does not have, and would have mixed results into a card that is otherwise a list of checks.
6. **Novelty sentence:** one leading "To our knowledge, as of early October 2026" covers both claims, so the instructed second "to our knowledge" is not repeated.
7. **Cuts that paid for v3** (main text still ends at the foot of page 4): the wording paragraph's frame-level Holm counts (App. D keeps them); the Holm-over-48 sentence of "Fragile" (Table I caption and App. C keep it); the v2 prefix association (moved to App. C); the reference's "−0.15, p = .015" clause in §III; "no headline rate by more than 8.3" in §IV (App. C); the screening counts in §IV (App. A); the abstract's "Only two pitfalls break the layout effect" and release clause; small trims in §VI, the limitations and the conclusion.


## G. Orchestrator edits after v3 (3 Oct)

| Edit | Source |
|---|---|
| §IV prefix: "NF4 agrees, App. C" | Ledger §12.6b (NF4 replication) |
| App. C, Table V: NF4 rows and caption | Ledger §12.6b (`evidence/prefix_nf4.json`) |
| App. C text: NF4 replication; "five of 16 cells" | Ledger §12.6b (corrected count) |
| §VI: "one released language-perturbation benchmark even fed its policy the unperturbed instruction" | Ledger §12.7. Chen et al. App. B, verified verbatim; LIBERO-PRO issue #14 exists (opened 30 Dec 2025, open). No maintainer acknowledgement is claimed. |
| §V: the "fragile" sentence shortened ("switches with almost every fork; 12 of 75 paired frames") | Ledger §11.13 and Table I; no new number |
| §VII: transfer and limitations sentences shortened | No new numbers |

## H. v3.2 (3 Oct, afternoon): round-2 results (ledger §13), with every change and its source

Table numbers as built in v3.2: IX compute and pre-processing ladder, X pre-processing verdicts, XI screening, XII wording × run, XIII placebo, XIV facets, XV O4, XVI filled-in card. The orchestrator's edits after v3 (§G) are kept unchanged.

### Main text

| Claim | Source |
|---|---|
| Abstract: "Pre-processing and 4-bit inference change most actions but no wording's direction" (optional clause; abstract now 174 words) | L§13.1 (60% and 68% of lateral tokens; no wording changes direction); L§13.3 (FP4 67%; no wording changes direction in any setting); L§4 (NF4 58%); instr. R4 item 5. To make room: "reading" cut, "that name" → "naming", "the direction of the residual word effect" → "the residual word effect's direction", "of the variance in the estimated word effect" → "of the word effect's variance". |
| Intro: "4-bit inference changes 58–67% of lateral tokens but no wording's direction" (was "58% … but no headline verdict") | L§4 (58%, NF4 on the GH200); L§13.3 (67%, FP4 on the A100; no wording changes direction); L§9 (4-bit and bf16 directions agree). "No headline verdict" was dropped because the ledger does not state FP4's headline rates. |
| Intro: "and" before contribution (iv) dropped | Space. |
| §II: 4-bit logs sentence reworded; "within a frame" dropped from the positive control | Same facts (L-data; L§1). Space. |
| §IV intro: pre-processing listed as Type N | instr. R4 item 2; it changes which images the model is asked about. |
| **§IV precision paragraph:** "4-bit" is not one setting; OpenVLA's paper names no type; its code's default (FP4) changes 67% of lateral tokens against bf16 (A100); the NF4 we ran 58% (GH200) but no headline verdict; on the A100 int8 32%, fp16, fp32 or another attention kernel about 7%, like a GPU change (6–8%, r = .97–.98); repeats bit-identical; no wording changes direction in any setting; only the original wording's verdict moves (null in bf16 on the A100, reversed in its five other A100 settings and in bf16 on the GH200) | L§13.3 (67.0/67.3%; 31.9%; 7.4, 7.4, 6.7%; repeat 0.0%; directions; "reversed in all five non-reference settings but null in the reference"; "OpenVLA's paper does not name its type"); L§4 (58%; no headline verdict); L§12.3 and L§11.1 (6–8%; r .968–.980); T1:ref (GH200 bf16 reversed). instr. R4 item 1. Cut for space: "used a different checkpoint" and the NF4 → bf16 contrast p (.55 → .039), which Table I keeps. |
| **§IV pre-processing paragraph** "(observations; strength)": against our frames (A100, bf16), OpenVLA's evaluation pre-processing changes 60% of lateral actions and a training-image approximation 68%; no wording changes direction; the latter enlarges every word effect four- to fivefold ("the left {noun}" +1.8 vs +0.4 bins); opposite both-correct at most 28% | L§13.1 recommended wording and tables (59.8%; 68.4%; +1.84 vs +0.39; 28.0%); L§13 Comparator (B4.1, A100 bf16). "four- to fivefold" replaces "about fourfold": see departures below. instr. R4 item 2 (tag). |
| §IV wording paragraph: the Bridge annotation-protocol quote cut | Space; App. F item 9 keeps it (`walke2023bridgedata`). |
| **§IV screening:** both scored by recorded arrangement (A100, bf16), the 824 composites screening rejected give the same verdict as the 820 it approved (opposite both-correct 8.1% vs 5.4%; the original wording reversed in both; row screen) | L§13.2 recommended wording and table. "−0.14 vs −0.10 grid steps" moved to Table I and Table XI for space. instr. R4 item 3. |
| §IV screening: "(41.5%; both-right 58%)" → "(41.5%)"; "(opposite both-correct 4.5%, 3/66)" → "(row sub_det)" | Space; App. A and Table I keep the numbers (L§7; T1:sub_det). |
| **Table I, `screen` pair:** rejected 8.1, 40/63/83, +2.0 (.711), 22/54 (.220), −0.14 (.001), away; approved 5.4, 47/72/88, +6.4 (.064), 17/54 (.009), −0.10 (.0005), away | L§13.2 table (both-correct, image-right, splits, signed and p). Contrast and split p-values: `evidence/round2.json` → `b6.{rejected,approved}.{contrast_pts,contrast_p,splits_p}`, the data file §13 names as its source; they are not printed in the ledger text (§D). They come from the same `evaluate()` as every Table I row (`btp_runs_round2.py`). Changes "strength": the split test crosses .05 (.009 → .220) and no statistic changes sign. Note §: compared with the approved composites, both scored by recorded layout (instr. R4 item 3). |
| Table I caption: "1 bin ≈ 0.32 mm if the units are metres" removed | Space; §II states it (L§11.2). |
| Fig. 2 caption: "dx, prefix and screen omitted"; "filled: the reference's outputs"; "A direction-blind contrast never reads y" cut | The figure is unchanged (L§11.5); space. |
| Table II: caption shortened; item 6 check "compute ladder: dtype, kernel, GPU, repeat"; item 7 "flip swaps the named twin; own pre-processing"; item 10 "approval rate per condition; score the rejected"; item 11 "clustered tests, multiplicity, controls" | L§13.3, L§13.1, L§13.2; instr. R4 item 6. Results in Table XVI (departures below). |
| §VI: "from kinematics to reset distributions" cut | Space. |
| §VII Limitations: "Annotator agreement (two annotators) and a double-coded audit of reporting practice are in progress." Pre-processing, screened-out scenes and other precisions removed | instr. R4 item 4; plan B6 step 4 and B7 (two humans double-code ≥ 25% of papers). |

### Appendix

| Claim | Source |
|---|---|
| App. A Runs: a second round on the same A100s reran a replay subset of 2,040 predictions and a ladder of 4,080 under six compute settings and two pre-processing paths, and scored the screened-out composites | L§13 Comparator and §13.1–13.3. Table III rows: compute ladder (6 settings), pre-processing (2 paths), screening check (1,644 = 820 + 824 composites; L§13.2, L§7). |
| App. C prefix paragraph: interaction range scoped to "every bf16 cell" | L§12.6. In `prefix_nf4.json`, the NF4 original-wording interaction is +1.85, outside −0.6 to +1.2, so "every cell" was wrong once the orchestrator added the NF4 rows. |
| App. C "Compute": comparator, batch size 1, folded readout; 2,040 replay (original wording and term-free prompt, original and mirrored images) and 4,080 ladder (six prompt pairs, original images); repeat on the replay only; direction rule | L§13 Comparator; L§13.3 ("How directions are called"; "(replay only)"). |
| Compute numbers: bit-identical repeat; about 7%; 32%; 58% (NF4, GH200) to 67% (FP4, no double quantisation); paper does not name the type; original null in the reference, reversed in the five others, FP4 −0.40 (p = 2×10⁻⁵), splits 3/19; both-correct ≤ 21.2% | L§13.3 Reading. |
| Pre-processing method: official path (JPEG encode/decode, `tf.image.resize` lanczos3 with antialias to 224; TensorFlow 2.17 in a separate environment; PNG); training-like path (area resize to 256×256, method assumed, then the official path) | L§13.1; instr. R4 (Appendix). |
| Pre-processing results: no direction change; original null (ours, official) and reversed (training-like); four- to fivefold ("the left {noun}" +1.84 vs +0.39; table side −1.61 vs −0.37); both-correct ≤ 28.0%; magnitudes depend on the pipeline, directions do not | L§13.1 tables and Reading. |
| Screening: 824 rejected and 820 approved, both by recorded arrangement; the approved set includes the 400 frozen scenes; hand relabelling (53) not involved; same verdict; screening narrows the scope; the 1,604 never-screened not scored; recorded layouts of the rejected not yet checked by hand (`\pending{B6 step 2}`) | L§13.2; L§7 (1,604); plan B6 step 2. |
| Table IX (compute and pre-processing ladder): every cell | L§13.3 table (tokens, r replay, sign agreement, directions); L§13.1 agreement table (59.8/59.1, 68.4/69.0, r .588 and .605, 89.9, 86.5); directions for the pre-processing rows and the reference from the L§13.1 verdict table under the L§13.3 rule. |
| Table X (pre-processing verdicts and the original wording's headline): every cell | L§13.1 verdict and headline tables. |
| Table XI (screening): every cell | L§13.2 table. |
| Table IV `screen` row: pointer to Table I and Table XI | L§13.2. |
| App. E Type U: screened-out and approved composites give the same verdict; reweighting pending (`B6 step 3`) | L§13.2; plan B6 step 3. |
| Table XVI item 6: compute ladder (repeat bit-identical; sdpa, fp16, fp32 6.7/7.4/7.4%, r ≥ .967; int8 31.9%, r = .845; FP4 67.0%, r = .642; no direction change; original null → reversed) | L§13.3. |
| Table XVI item 7: official 59.8%, training-like 68.4% (approximation); no direction change; four- to fivefold; both-correct ≤ 28.0% | L§13.1. |
| Table XVI item 8: pending relabelled `B5 step 3` (appended token) | plan B5 step 3 (not run in round 2). |
| Table XVI item 10: 824 rejected vs 820 approved (8.1% vs 5.4%; −0.14 vs −0.10 bins); κ still pending | L§13.2; plan B6 step 4. |

### v3.2 departures from the Round 4 instructions

1. **"Four- to fivefold" instead of "about fourfold".** The ledger's own reading gives 4–5× (prenominal 4.7×, table side 4.4×, object 5.35×).
2. **Table II keeps checks, not results**, as in v3 (§F item 5). The B4, B5 and B6 results are in Table XVI items 6, 7 and 10.
3. **Table I `screen` uses two numbers that are only in `round2.json`**: the contrast and the split p-values. The ledger's B6 table omits them, but Table I's columns need them.
4. **The main-text B6 sentence drops "−0.14 vs −0.10 grid steps"** for space. Table I and Table XI show them.
5. **One new appendix pending, `B6 step 2`.** The rejected composites are scored by recorded layouts that nobody has checked by hand. The comparison with the approved composites is fair, because both sets use recorded layouts, but the check is still open.
6. **Abstract at 174 words** with the optional clause, about 170 as allowed.

