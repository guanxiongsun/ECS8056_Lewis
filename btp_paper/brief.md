# BtP paper: writing brief (v2, 2 Oct 2026)

For the writer and reviewer agents. Sources you may use, and nothing else:
- numbers: `evidence/claims_ledger.md` (use its recommended wordings), `evidence/one_at_a_time.md|csv`;
- citations: `lit/lit_vla.md|bib`, `lit/lit_methods.md|bib`, `lit/lit_substrate.md|bib` (claims about a paper only
  where its lit report quotes support);
- background: `docs/workshop_plan_v2.md` (§2 BtP, §5 outline) and `docs/done.md`. Where they disagree with the
  ledger, the ledger wins.

## 1. Venue facts
- CoRL 2026 workshop "Everything Beneath the Policy: How controllers, hardware, data infrastructure shape robot
  learning" (BtP). Deadline Fri 9 Oct 2026, 23:59 CT. Double-blind, OpenReview, 2–3 reviewers.
- **4 pages of main text**, including figures and tables. References and appendices are unlimited and come after.
- Template: IEEEtran conference mode (allowed; packs more per page than the CoRL template). Skeleton: `main.tex`.
- Call, quoted: "Because these choices can change the strength, scope, or even direction of an empirical claim,
  methodological contributions should also examine how sensitive their conclusions are to key substrate choices."
  In scope: controlled studies of design choices; factorial ablations; negative or surprising results; new
  reporting conventions; tools that expose hidden choices. Reviewed on "what they teach the community about the
  substrate beneath the policy". The organisers will release a "Design Choice Reporting Checklist v0.1" (hardware and
  embodiment, motors and actuation, kinematics, controller gains and rates, action spaces, observations,
  demonstrations, simulation, reward design, reset distributions). Their Methodology question names
  "factorial designs, scaling-style ablations, or model-organism tasks"; their Reporting question asks "What should
  papers report so results are reproducible, interpretable, and easier to compare?"

## 2. Story
**Hook.** Holding one model (OpenVLA-7B) and one stimulus set fixed, the same language evaluation concludes that the
model grounds "left" and "right", grounds them backwards, ignores them, or that the stimuli do not work, depending on
choices that papers rarely report. (Ledger numbers for each verdict: correct direction — "the left {noun}" wording,
splits 20/26, p = .009; reversed — "... left side of the table", 3/24, p = .0003, or correctly scored mirrored images,
4/25, p = .0009; significant "grounding" from a direction-blind contrast while the behaviour is reversed — bf16
original wording, +10.7 points, p = .039, signed effect −0.15 bins, p = .011; null — 4-bit original wording, +4.1,
p = .55; stimuli "fail" — reading dx, paste side unrelated to the action, 86.1% vs 85.7% image-right, p = 1.0.)

**Measurement substrate.** Between the policy's output and the verdict sit: the action interface (which action
component is the probed axis, its sign and its zero; the token↔bin map; the bin grid; the readout; the decoding
order), the compute stack (inference precision; GPU), the observation pipeline (image transforms and their
re-scoring; preprocessing), the language and data pipeline (instruction wording; how the training data uses the
probed words), and the evaluation protocol (stimulus screening and fallbacks; the decided-threshold; the decisive
statistic).

**Two kinds of choice** — this distinction is the paper's main conceptual contribution:
- **Pitfalls**: conventions with a right answer that, set wrongly, silently change a verdict. Ours: axis (dx vs dy),
  sign, zero point, token map, bin width vs grid, mirror scoring, direction-blind statistic. Remedy: report the
  convention and the check that verified it.
- **Forks**: legitimate choices that answer different questions (estimands). Readout: the executed action (argmax
  under greedy decoding) vs the expected action. Prefix: the total effect of the word on dy (through the model's own
  dx token) vs its direct effect (dx held fixed). Precision: the model as deployed on small GPUs (4-bit) vs native
  (bf16). Wording: the effect of one phrase vs of a word across phrasings. Screening and subsets: the population of
  scenes a claim covers. Threshold: what counts as a decision. Remedy: report the specification curve.

**Tag** each choice with what it changes, in the call's words: **direction**, **strength** or **scope** of a claim.

**What survives (one-at-a-time now; the full curve is pending).** "Neither grounded": opposite-scene both-correct
≤ 20.4% in every row (reference 6.2%). "Not lexical": opposite same-sign 69–92% in every row. "Follows the layout":
holds in every row except the pitfall rows (dx flattens it; the flipped sign reverses it). Fragile: the
direction-blind contrast (its significance switches with threshold, readout, precision, images, subset, zero point
and wording) and the direction of the residual word effect (set by wording, consistently across four runs).

**Lesson for the substrate community.** For token-based policies the readout from logits to numbers is part of the
action space, and an evaluation's conventions are part of the substrate; a reporting checklist should cover them.
Pitfalls need reporting so that readers can check them; forks need a specification curve.

## 3. Mapping to the workshop's substrate layers
| Workshop layer / checklist item | Our choices |
|---|---|
| Action spaces | probed axis, sign and zero (action frame ↔ image); tokeniser, token↔bin map, bin grid; readout; decoding order (autoregressive prefix); horizon (first action) |
| Hardware / compute | inference precision and quantisation (4-bit is the usual choice on small GPUs); GPU type (A100 vs GH200 at the same precision changes 8.1% of lateral tokens) |
| Observations | image preprocessing vs training; image edits (compositing); mirroring and its re-scoring |
| Demonstrations / data pipeline | the training data's language conventions (Bridge uses "left/right" mostly for destinations), which make wording a fork; text normalisation (OpenVLA lowercases instructions in training) |
| Evaluation protocol | stimulus screening, fallbacks, annotation; decided-threshold; decisive statistic; multiplicity |

## 4. Structure and page budget (4 pages, two columns = 8 columns)
1. **Introduction** (~1 col). Hook; the measurement substrate; the gap (§6); the model-organism design;
   contributions: (i) a pitfall audit, (ii) a fork analysis with a specification curve, (iii) a reporting card
   offered to the workshop checklist, (iv) released probe, stimuli and outputs (anonymous URL). Fig. 1 (schematic, see
   §7) at the top of page 1 or 2 spanning both columns or one column.
2. **The probe** (~1 col). Model; stimuli (twins; opposite vs same-side layouts; why same-side scenes separate
   grounding from a word→direction shortcut); conditions; the reference analysis (every choice at its reference
   value: dy, image-right negative, physical zero, expected value over all 256 tokens with the edge token folded,
   ≥1 bin, bf16 on one GPU, original wording, all 340 scenes, correct mirror scoring, direction-aware statistics);
   the behavioural verdict in 3–4 sentences (bf16 reference numbers, 4-bit logs as replication), citing the
   companion paper anonymously; the positive control (paste displacement on dy).
3. **Pitfalls** (~1.5 col) and 4. **Forks** (~2 col). Lead with Table 1: a compact version of
   `evidence/one_at_a_time.md` as a `table*` at `\footnotesize` or `\scriptsize`, adding a "Kind" column
   (pitfall/fork) and a "Changes" column (direction/strength/scope); keep the rows ref, dx, sign, zero, tok, w255,
   argmax, thr0, thr2, nf4, mir, mir_naive, w_prenominal, w_object, w_table_side, sub_det, and pending rows for the
   prefix and screened-out scenes; you may drop columns to fit. Prose only for the most instructive cases: mirror
   scoring (a significantly reversed result scored as significant grounding), the direction-blind statistic, the
   axis/sign/zero group, the token map (invisible on dy, obvious on the gripper), precision (58% of tokens, no
   headline change, secondary significance changes), the prefix (association only; control pending), wording.
   Each choice discussed in prose ends with a bold "Report:" line (`\report{}` macro).
5. **What survives** (~1 col): Fig. 2 specification curve, `\pending{}` placeholder box with a precise caption of
   what it will show; the universe (forks only, pitfalls held at their correct values), the joint inference test
   (from `lit/lit_methods.md`), and the conclusions robust in the one-at-a-time table.
6. **Reporting card** (~1 col): Table 2 grouped by the layers in §3, mapped to the workshop's checklist categories.
   The filled-in card for this study goes in the appendix.
7. **Limitations and conclusion** (~0.5 col).
Appendix: stimuli and screening; readout and statistics definitions; the full one-at-a-time table with CIs; the full
wording × run table; specification-curve universe and verdict rules (pending); the filled-in reporting card;
extended related work (the lit reports have plenty).

## 5. Rules
- **Numbers.** Only from the ledger and the one-at-a-time table, using the ledger's recommended wording,
  denominators and hedges. Name the run each number comes from (bf16 replay with the folded readout is the
  reference; 4-bit logs are the replication and carry only the 255-token readout). Anything not yet computed becomes
  `\pending{what will go here}`. Never round a number into a stronger claim. Scope every claim: OpenVLA-7B, first
  action, lateral axis, composited Bridge frames, this stimulus set.
- **Causal language.** The prefix result is an association. The precision result is "no headline change", not
  "no change". The Bridge statistics explain the wording effect in the companion paper; here, say only that the
  training data uses the words differently by role and give the numbers.
- **Pitfall phrasing.** Counterfactual and neutral ("scored on dx instead of dy, the action is unrelated to ..."),
  never as anyone's mistake. Do not mention any dissertation, thesis, course, student, notebook (NB06 etc.) or
  "original pipeline".
- **Citations.** Only keys from `lit/*.bib` (merge into `refs.bib`, de-duplicate keys). If you need one that is
  missing, write `\note{cite: ...}`.
- **Anonymity.** No names, institutions, course codes, student IDs, repository URLs, Hugging Face links,
  acknowledgements, or GPU-cluster names (say "an NVIDIA GH200" and "an A100"). Cite the companion paper as anonymous.
  Dual-submission footnote: "A companion paper on the behaviour itself is under review at another CoRL 2026
  workshop; this paper concerns its measurement." Code and stimuli: "released at an anonymous URL".
- **Style.** British English. Short declarative sentences; define each term once; no hype words. Use
  "image-left/right" for directions in the image and say so once. Prefer numbers with denominators.
- **Length.** Main text including figures and tables must end by the bottom of page 4. Check with `./build.sh`
  (prints the page count and the page where References starts; References must start on page 5 or later only if
  the main text overflows — it should start on page 4 or 5 with the main text ending on page 4).

## 6. Gap and positioning (from `lit/lit_vla.md` §(d)–(e))
**Gap, stated narrowly** (use "to our knowledge, as of early October 2026"): no prior evaluation of VLAs varies the
measurement choices beneath a language-use verdict and reports which verdicts survive; we found no multiverse or
specification-curve analysis of any robot-learning evaluation. Existing language evaluations fix one analysis path,
check robustness to a single choice (a detection threshold, a layout swap, rendering conditions), or audit
closed-loop success instruments and pipelines rather than readouts. We also know of no VLA language probe with
identical objects on the same side of the gripper in real robot frames.

**Concede and cite**:
- STAGE (`2609.13458`): the protocol neighbour (fixed observation, single-step OpenVLA readout under instruction
  swaps, a direction-aware metric next to a direction-blind one, threshold-free AUC). It already shows that
  direction-blind sensitivity and direction-aware consistency differ. We add twins and same-side controls on real
  frames for OpenVLA, token-distribution readouts, and a multiverse across choices; our conclusion concerns the
  measurement, not the model.
- Kirouane et al. (`2609.07470`): the closest thesis ("measurement rather than translation") with closed-loop
  instruments for Greek instructions. Converging evidence, not prior art for our method.
- vla-eval, IndustrialVLA-Bench, the SmolVLA ONNX study, VLAQuantBench: 2026 evidence that the evaluation and
  deployment substrate moves closed-loop success; none examines a language verdict. Our card adds readout-level items.
- Act2Answer: swap-averaged left/right scoring with an effective-N caveat; single-choice ablations.
- Caveat to state: offline single-step action readouts do not forecast closed-loop success (SIMPLER; REAL-I). We
  present results as properties of the measurement, not deployment predictions.
- Methodological lineage (from `lit/lit_methods.md`): analytic flexibility and specification curves; deep RL
  reproducibility; LLM evaluation sensitivity (prompt format, metric choice, first-token readouts, quantisation
  flips); robot-learning evaluation practice; reporting checklists.

## 7. Figures and tables
- **Fig. 1** (TikZ, compiles with tectonic): the measurement pipeline left to right — scene with twins (opposite /
  same-side) → OpenVLA → action-token distributions (dx decoded before dy) → readout (argmax / expected value) →
  action frame (axis, sign, zero) → scoring (mirror swap) → statistic (direction-aware or not) → verdict. Mark
  pitfalls in one colour and forks in another, each labelled with its Table 1 row. Keep it simple and legible at
  column width or full width; it may replace prose.
- **Table 1**: the one-at-a-time table (see §4).
- **Fig. 2**: specification curve, placeholder (`\fbox` with the `\pending{}` text) and a caption that states the
  universe, the outcome measures (signed word effect toward the named twin; opposite both-correct), the joint test,
  and the indicator panel.
- **Table 2**: the reporting card.

## 8. Facts checked by the orchestrator (2 Oct 2026)
- OpenVLA lowercases the instruction in training: `prismatic/vla/datasets/datasets.py` line 42,
  `lang = rlds_batch["task"]["language_instruction"].decode().lower()`; its evaluation helper does too
  (`experiments/robot/openvla_utils.py` line 163, `task_label.lower()`). In our Bridge manifest 45.4% of the 17,035
  instructions contain an uppercase letter and 27.8% end with a period. The natural-frame runs pass the raw text; the
  twin-scene prompts are generated in lower case, so the twin results are unaffected. Effect on the natural frames is
  not measured (`\pending{}` if mentioned; card item).
- OpenVLA's official Bridge evaluation pre-processes images "to make input images in distribution with respect to the
  inputs seen at training time": JPEG encode/decode, then `tf.image.resize(..., method="lanczos3", antialias=True)` to
  224×224 (`experiments/robot/bridge/bridgev2_utils.py` lines 101–115), and asserts that centre-cropping is disabled
  for Bridge. **Correction (18:45):** our frames are **640×480** RGB from the Open X-Embodiment release of Bridge
  (`gs://gresearch/robotics/bridge/0.1.0/`, `data.py` l. 31–53), passed straight to the Hugging Face processor, which
  resizes them to 224×224. OpenVLA trained on `bridge_orig` (the official BridgeData V2 release, 256×256 frames); its
  README says "the version in OXE is out of date (as of 12/20/2023)". So the observation path differs in source
  release and resolution (640×480 vs 256×256), resize kernel and JPEG round trip. Effect not yet measured
  (`\pending{}` if mentioned; card item). Composites: the second twin is the object's own pixels cut out, mirrored
  and pasted (`compose_scenes.py` docstring); objects and the gripper are located with an open-vocabulary detector
  (OWLv2), with an image-centre fallback for the gripper when detection fails.
- The GPU runner predicts one prompt at a time (batch size 1, no padding), greedy decoding; dy is read conditioned on
  the model's own greedy dx token. 20 duplicate stimuli on different GH200s gave bit-identical outputs.

## 9. Draft reporting card (refine; map each item to a workshop checklist category)
Action interface: (1) probed axis, sign and zero in image terms, and how verified (e.g. against demonstrations);
(2) tokeniser: bins and centres, token↔bin map including edge tokens, un-normalisation statistics; (3) readout
(argmax / expected value / sampled) and the share of ties; (4) decoding order: whether the probed dimension is
conditioned on earlier decoded ones (own prefix, forced, marginal); (5) horizon (first action / chunk / closed loop).
Compute: (6) precision, quantisation library and version, attention kernel, GPU; action-level agreement with native
precision. Observations: (7) preprocessing relative to training (resize kernel, JPEG round trip, crop); image edits;
transforms and their re-scoring rule. Language and data: (8) exact templates, text normalisation, paraphrase noise
floor, number of wordings; (9) how the probed words are used in the training data. Evaluation: (10) screening
criteria, approval rates per condition, fallbacks, annotators and agreement; (11) the decisive statistic (is it
direction-aware?), unit of analysis, thresholds, multiplicity; a positive control and the channel it was run on;
(12) sensitivity: the verdict across forks (specification curve).

## 10. From the methodology and substrate reviews (read `lit/lit_methods.md` §(c) and `lit/lit_substrate.md` "Read this first")
- **Bibliography.** Use `refs.bib` (229 merged, verified entries) and the canonical keys in `bib_keymap.md`
  (e.g. `li2024evaluating` for SIMPLER, `kim2025finetuning` for OpenVLA-OFT, `kamath2023whats`, `fang2026liberocf`,
  `brohan2023rt2`). Do not add entries that are not in `refs.bib`.
- **Specification-curve design** (Simonsohn et al. 2020 + Del Giudice & Gangestad 2021): only *equivalent* (Type E)
  alternatives share one curve; forks that change the estimand (Type N: readout argmax vs expected value; own vs
  fixed prefix; each wording template; possibly bf16 vs deployed 4-bit) get separate curves or facets, each with its
  own joint test; Type U choices (e.g. screening rules with condition-dependent approval) go to exploratory panels.
  Pitfalls are excluded from every curve and audited separately. Joint inference: re-run every specification on
  datasets where the left/right label is flipped at random within each scene (the design-based null; clusters =
  base frames), and report the median effect, the share of specifications significant **in the predicted direction**,
  and Stouffer's Z, with permutation p-values (ties count half). Never use Simonsohn's two-sided "dominant sign"
  display: it would score consistently backwards behaviour as an effect — our own decisive-statistic pitfall. This
  Type E/N/U grounding is the formal basis for the pitfall/fork distinction; Auspurg & Brüderl show that much
  many-analyst disagreement is about unclear estimands. Say so in one or two sentences.
- **Novelty, as the methods review words it:** to our knowledge, the first specification-curve analysis of a
  robot-learning evaluation, and the first aimed at the measurement layer of a VLA language probe. Do **not** claim
  to be first to show wording or precision sensitivity in VLAs (LIBERO-Para, LADEV, VLATest, STAR-Gen, VLAQuantBench
  do, at the success level), or first to use action-token probabilities (Zollo & Zemel, MG-Select, INSIGHT).
- **LLM parallels for the intro** (one sentence, two or three citations): prompt formatting moves LLM accuracy by up
  to 76 points (Sclar et al.); first-token vs generated answers disagree (Wang et al.); quantised models flip
  individual answers at equal accuracy (Dutta et al.) — OpenVLA's "4-bit matches bf16" is exactly that aggregate
  argument; metric choice can manufacture or erase effects (Schaeffer et al.).
- **Substrate facts we can state with citations** (see quotes in `lit/lit_substrate.md`):
  - OpenVLA's paper describes 256 bins / integers 0–255; its code builds 256 edges, i.e. **255 bin centres**, and
    decoding folds the top token into the last centre ("only (# bins - 1) bin intervals"). Our grid step
    (q99 − q01)/255 follows the code. Write "256 bins (255 centres)".
  - OpenVLA's "4-bit matches bf16" (71.9% vs 71.3% on 8 Bridge tasks, 80 rollouts each) used a different checkpoint
    (footnote 4) and does not name the 4-bit type; the default 4-bit path in its pinned transformers version is FP4
    (derived from code — hedge). We ran **NF4** with bf16 compute: say so.
  - BridgeData V2's instructions were written after collection by crowd workers asked to describe the task "with
    particular emphasis on the final location of any moved objects" — the data-pipeline root of why its "left/right"
    mostly name destinations. This is a demonstrations-layer fact for the paper.
  - No document defines Bridge's action frame or sign; OpenVLA re-derives Bridge actions from successive gripper
    states; ECoT's Bridge labelling code independently maps −y to "right" "assuming a fixed camera" (corroborates
    image-right = −dy). Open X-Embodiment does not align action frames across datasets.
  - Greedy decoding is the de facto default (`do_sample=False` in every official call); `predict_action` appends
    token 29871 to the prompt; generation is not restricted to action tokens.
  - Flipping images needs flipped language: referring-expression codebases swap "left"/"right" when flipping
    (MDETR, TransVG, Yang et al. 2019); others disable flips; MirrorDuo mirrors actions too. Neither OpenVLA nor Octo
    trains with flips. DIAL notes that with duplicate objects, baselines ignore "left/right".
  - Organisers' relevant work: Bronars et al. 2026 (RSS) on controller gains that "remain largely undocumented" in
    large datasets — cite once where we argue that undocumented conventions shape results. Aljalbout et al. 2024 on
    the action space.
