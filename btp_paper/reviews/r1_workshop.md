# R1: workshop programme-committee review of the BtP draft (2 Oct 2026)

Reviewer stance: a CoRL 2026 BtP programme-committee member, reading as a robot-learning and controls person, not a VLA-interpretability specialist. I read `main.tex`, the rendered pages 1–5, `brief.md`, `open_questions.md`, `claims_trace.md` and the gap and threat sections of the three literature reports. I also checked in the code the assumptions that `open_questions.md` §3 flags (§2.3 below). All line numbers refer to `main.tex` as of 2 Oct 2026, 18:25.

---

## Part 1. Review

### Summary

The paper uses one model organism, OpenVLA-7B reading real BridgeData V2 frames with a composited twin object, to show that the measurement layer between a policy's output and an evaluation's verdict can reverse, create or erase a language-use verdict. That layer covers the action axis, sign and zero; the token-to-bin map; the readout; precision and GPU; mirror scoring; wording; threshold; screening; and the decisive statistic. It separates *pitfalls* (conventions with a right answer, to be reported with a check) from *forks* (choices without one, to be reported as a specification curve). It tags each choice by whether it changes the direction, strength or scope of a claim. It reports a one-at-a-time sensitivity table (the specification curve is still pending) and proposes a reporting card mapped onto the workshop's announced checklist. The robust conclusion is that the model does not reliably pick the named twin. The direction of the small remaining word effect, and any direction-blind statistic, are fragile.

### Strengths

1. **A well-built model organism.** Twins on opposite sides and on the same side of the gripper, plus a paste-displacement positive control, make the measurement layer the only thing that varies. This is the kind of "model-organism task" the call's Methodology question asks for.
2. **A transferable distinction.** "Pitfall vs fork", with a different remedy for each (report-and-verify vs curve), is the most reusable idea in the paper and maps directly onto a checklist. Grounding it in Simonsohn et al. and Del Giudice & Gangestad is appropriate.
3. **Some reversals are genuinely striking because the model outputs are identical.** Correct vs naive mirror scoring gives 4/25 vs 21/25 splits, both at p = .0009. A direction-blind contrast reports "grounding" on outputs whose signed effect is reversed. Reading dx instead of dy makes the manipulation check fail.
4. **Substrate findings this audience will value:**
   - Bridge documents no action frame or sign, and the paper verifies one on demonstrations (96.9% vs ≤57%).
   - The edge-token fold is invisible on dy but obvious on the gripper (expected value 0.995 → 0.30).
   - NF4 vs bf16 changes 58% of lateral tokens with no headline change.
   - An A100 and a GH200 at the same precision disagree on 8.1% of lateral tokens, while duplicates on two GH200s are bit-identical.
   - Bridge's post-hoc labelling protocol ("emphasis on the final location") explains why "left/right" plays more than one role in the data.
5. **Unusually careful claims.** The prefix result is called an association, precision gets "no headline change" rather than "no change", pitfalls are phrased counterfactually, and every number is traced to its source. The software and hardware reporting in App. A is better than most papers'.
6. **A concrete, worked reporting card.** Table II plus the filled-in App. F answer the call's Reporting question directly.

### Weaknesses

**W1. The hook is built mostly from forks, and its vocabulary contradicts the conclusion.**
- *Forks.* Of the four verdicts in the abstract, three come from forks: wording ("grounds" vs "grounds backwards"), mirrored vs original images, and 4-bit precision ("ignores"). Only "stimuli fail" (dx) is a pitfall. A reader can reply "different questions, different answers", and the paper's own definition of a fork concedes that. The cleanest same-output reversal, naive mirror scoring (21/25), is not in the hook at all; only its correct half (4/25) is.
- *Vocabulary.* The hook says the model "grounds" (or grounds backwards) the words; the conclusion says it "does not reliably select the named twin". These are different quantities: how often the word splits the two actions, and which way the splits go. They need different names, or the paper reads as self-contradictory. The intro's "the evaluation finds correct grounding" (l. 57) for a wording whose both-correct rate is 18.7% is the sharpest case.
- *"Ignores them".* The paper's own paraphrase floor contradicts this verdict at 4-bit: swapping left/right moves the action more than swapping grab/take (p = .00035). The 4-bit null is a null on direction, not on sensitivity. Say so; it strengthens the point.

**W2. The main methodological novelty is pending, and the text already claims it.**
- l. 61 says "we add … a multiverse over measurement choices" and "this is the first specification-curve analysis of a robot-learning evaluation".
- Today the paper has a one-at-a-time audit, a design (App. E) and a placeholder Fig. 2.
- The main text has nine `\pending` markers. A reviewer will treat a placeholder figure as missing work.

**W3. The statistics invite an easy attack.**
- *Borderline and uncorrected.* Many narrative p-values sit at .01–.05 (.039, .038, .011, .049). Table I carries 48 uncorrected tests (16 rows × 3), and base-frame clustering enters only the CIs. One base frame yields several configurations (`construct_id` in `compose_scenes.py`), so the Wilcoxon over 340 pairs is mildly anticonservative.
- *"Fragile" proves little as stated.* The claim that the direction-blind contrast is fragile ("its significance switches with the threshold, readout, precision, …") is what any p ≈ .04 statistic does under perturbation. A reviewer will say it shows nothing about the statistic.
- *The real argument is missing.* What makes the contrast a pitfall is structural: it is unchanged when "left" and "right" are swapped, so it cannot separate grounding from its reverse. The paper does not say this, and leaves a `\pending{B3}` simulation in its place.
- *The good news is unused.* The reversals worth leading with survive a table-wide Holm correction (Appendix of this review, item B).

**W4. Scope, and the size of the effects.**
- *Narrow scope.* The paper studies one model, the first action, offline readouts and composited frames.
- *Tiny effects.* Assuming dy is in metres, one bin is ≈0.33 mm of commanded lateral motion per 5 Hz step. The signed word effects are 0.05–0.16 mm per step, and the positive control 0.46 mm. A controls reader will ask at once whether verdicts about sub-millimetre first commands matter. The paper never states the physical unit.
- *Too little on transfer.* That continuous, flow-matching or chunked policies, closed-loop evaluation or other datasets inherit the lessons is asserted in one sentence ("Continuous action heads remove the token map, bins and prefix").
- *What is missing.* Without these, the paper reads as an audit of a niche VLA-language probe rather than a lesson about the substrate.

**W5. Presentation.**
- *No figure shows an effect.* Fig. 1 is a map. Table I needs ten columns decoded before a reader sees verdicts flip, and has no verdict column.
- *No stimulus image anywhere.* Readers cannot judge how plausible the composites look.
- *Terms used before they are defined.* "Splits", "opposite-scene", "direction-blind contrast", "signed effect", "bins" and "term-free action" all appear in intro ¶2 before §II defines them. "Estimand", "specification curve" and "residual word effect" are never glossed in plain words.
- *Dense prose.* Many sentences carry three to six numbers.

**W6. The fork definition is inconsistent with the paper's own design.**
- The abstract, intro, Fig. 1 caption and §IV define a fork as a choice that defines a *different estimand*.
- Yet threshold, scene subset and images are treated as *equivalent* (Type E) choices that share one curve (l. 59, l. 199, App. E). Those are forks that do not change the question.
- A methods reviewer will spot this. A controls reviewer will simply be lost in "Type E/N/U".

**W7. The reporting card says what to report but not how to check it.**
- The paper's own remedy for a pitfall is to "report the convention *and the check that verified it*", but Table II lists no checks.
- The mapping to the organisers' categories misses natural homes:
  - the action frame is a *kinematics* convention;
  - the horizon and the 5 Hz rate belong under *controller gains and rates*;
  - the evaluated scene set (screening, subsets) is the evaluation's *reset distribution*.

  With those, only the statistic needs a new "evaluation protocol" category.
- There is nothing on the cost of reporting.

**W8. Positioning against the pipeline audits is too thin.**
- vla-eval and IndustrialVLA-Bench (undocumented pipeline parameters that move success by up to 55 points) are the closest in spirit to "pitfalls", but they appear only inside a citation list.
- One sentence should say how this paper differs: a language verdict within one model rather than closed-loop success across models, and the pitfall/fork split.
- The STAGE concession is good as written.

### Questions to the authors

1. How many base frames underlie the 340 scenes and the 151 opposite scenes? Do the binomial and Wilcoxon p-values in Table I account for that clustering? A base-frame sign-flip permutation, the same machinery as the B1 joint test, would settle it offline.
2. In what physical units is dy? If metres (the ledger's "motion > 5 mm" role statistics suggest so), please state effects in mm per step. Would a physically motivated deadband, for example 3 bins ≈ 1 mm, change any verdict?
3. Why is the reference readout the expected value rather than the executed (argmax) action? For a robot audience, the executed action is the natural default. If the reason is resolution (30% ties under argmax), say so in a clause.
4. What is the verdict for the second branch of the wording fork, "the word across phrasings" (the signed effect pooled over the four named-twin templates)? It is computable from existing outputs.
5. Can the GPU-only effect on the verdict be isolated without new runs? Compare the A100 NF4 logs with the GH200 NF4 replay, both read with the 255-token readout, which Table III says the replay logged.
6. Which of the pitfalls have you seen in *published* evaluations? If none can be cited, say that they are demonstrated counterfactually, and point to the closest live practice. Direction-blind "action sensitivity" and action-change metrics are examples.
7. Is component 0 of Bridge's action the forward axis? A check: dx against the object's vertical image position on the 98 unedited frames. If so, the dx pitfall has a natural explanation (edit 11).
8. Will the release include the full 256-way action-token distributions per prediction? That would make the paper a "tool that exposes hidden choices" in the call's sense.
9. Does the companion paper's headline analysis use this paper's reference analysis? If not, which Table I row does it correspond to? Readers of both papers will check.

### Scores (1–5)

| Criterion | Score | Reason |
|---|---|---|
| Relevance | 4 | Action space, compute, observation pipeline, data pipeline and evaluation protocol are all in the call. The VLA-language framing is narrower than it needs to be. |
| Novelty | 3 | The pitfall/fork split and a readout-level audit of a language verdict are new for VLAs. Individual phenomena are known (wording and quantisation sensitivity at the success level; STAGE's direction-blind vs direction-aware metrics). The specification curve, the main methodological novelty, is pending. Rises to 4 with B1. |
| Technical soundness | 3 | Careful and traceable, but there are borderline uncorrected p-values, clustering is unclear, the effects are sub-millimetre, and the scope is one model and first actions. |
| Clarity | 3 | Dense; jargon used before definition; no effect figure or stimulus image. The fork definition is inconsistent. |
| Significance | 3 | The card and the frame, mirror and statistic lessons are useful. Their reach beyond token-based VLA probes is asserted, not argued. |

### Recommendation

**Weak accept** for the draft as it stands. With top edits 1–5 made and every `\pending` either delivered or removed from the main text, I would move to **accept**. **Confidence: 4/5** (expert in evaluation methodology and multiverse analysis, familiar with VLA tokenisation; I did not re-run any analysis).

---

## Part 2. Responses to `open_questions.md`

### §1 Title

- **Keep the hook; replace the subtitle.** "The Measurement Substrate Beneath" is jargon-heavy and costs a line.
- **First choice:** *Same Outputs, Opposite Verdicts: Pitfalls and Forks Beneath a VLA Evaluation.* Use it if the hook is rebuilt around identical-output reversals (edit 1): "same outputs" is then literally true of the lead examples, "Pitfalls and Forks" names the contribution, and "Beneath" echoes the workshop.
- **Acceptable:** *Same Model, Same Scenes, Opposite Verdicts: Pitfalls and Forks Beneath a VLA Evaluation.*
- **Fit:** check that the subtitle fits on one line (the current line 2 holds 39 characters).
- **The alternatives.** I agree that alternative 1 ("decide") overstates. Alternative 3 claims a result about token-based policies in general from one model.

### §2 Decisions needed

1. **Novelty depends on B1.**
   - Agree. B1 needs no GPU time apart from the prefix facet: every other facet and the label-flip joint test can be computed from the logged ladder runs, which have both precisions, both image transforms and all three readouts.
   - Run B1 with 16 facets and drop the prefix facet if B2 does not land. Show a compact per-facet summary (median effect, share significant toward the named twin, Stouffer's Z, permutation p) in the appendix and one sentence in §V, not 16 curves in the main text.
   - If B1 slips, use the fallback novelty sentence in edit 4. It is lit-A's defensible gap claim and is true of the one-at-a-time audit alone.
2. **Mirrored images: Type U, not Type E.**
   - Mirrored robot images are out of distribution (MirrorDuo), and neither OpenVLA nor Octo trains with flips.
   - The data say they are not equivalent: the mirrored run gives a much stronger reversed effect (−0.48 vs −0.15 bins) and a different layout profile (39/79/86 vs 44/71/99).
   - Pooling them into a Type E curve mixes populations. Report them as an exploratory replication panel, and restore the out-of-distribution clause in §III (edit 10).
   - Side effect: the within-facet curves shrink (3 thresholds × 4 subsets for O2/O3). That is itself informative: most of the variation sits between facets, that is, most choices change the question.
3. **/254 vs /255.** Switch the reference to the /255 grid step. The numbers are identical, the inconsistency disappears, and the `w255` row can leave Table I. Keep "/254" as a counterfactual row in Table IV.
4. **"Changes" tags.**
   - **dx → strength (erases).** Reading dx does not narrow where a claim holds; it removes the layout effect and the stimulus check. "Scope" describes what happened incorrectly.
   - **mir → strength.** Keep, and say "out of distribution" in the text.
   - **w_table_side → strength.** Keep; same direction as the reference, stronger.
   - **sub_det → strength.** Keep; the contrast loses significance and the signed effect stays reversed.

   The deeper fix is a **Verdict** column (edit 7). The tags are relative to a reference whose own verdict is "weakly reversed", which confuses readers. An absolute toward/away/null verdict per row removes that dependence. Reserve "scope" for statements about a fork as a whole: a claim about "left" made with one template has the scope of that template (edit 5).
5. **GPU fork row.** Yes, and it needs no B4. Compute it from existing outputs (Question 5) and put it in Table I and in the effect figure. It is the most "beneath the policy" row in the paper, and "no verdict change" would be a clean negative result, which the call explicitly welcomes. Also move "the machines also differ in CPU architecture and driver" from App. A into the main text.
6. **Act2Answer.** Low priority. If space allows, add one clause to the positioning sentence in l. 61 ("Act2Answer averages left/right-swapped layouts [kachaev2026act2answer]"), not to the mirror paragraph. App. G is sufficient if space is short.
7. **B3 sentence.** Delete the slot and replace it with the label-swap invariance argument (edit 3b). It is a one-line proof, stronger than a simulation and immune to the p-value critique.
8. **B7 sentence.** Do a desk audit ("B7-lite") of the ten works in lit-A's comparison table against the 12 card items. It takes reading time only and turns "rarely reported" into a number, which answers the call's Reporting question directly. If it is not done, delete the `\pending` and point to lit-A's observation in App. G that none of the closest evaluations states its readout rule.

### §3 Guesses and assumptions (checked in the code)

| Assumption | Status | Evidence |
|---|---|---|
| Prompt template | **Correct** | `model.py:196`: `"In: What action should the robot take to {instruction}?\nOut:"` |
| Token 29871 appended | **Correct** | `scripts/run_gpu.py` `predict()` appends `EMPTY_TOKEN_ID` (`action_bins.py:34`, = 29871) when absent |
| Folded expected value renormalised over the 256 action tokens | **Correct** | `scripts/run_gpu.py:97–105`: `expected = (p / mass) @ centers[bins]` |
| Scene construction | **Needs rewording** | The copy is cut along its segmentation mask and **mirrored horizontally by default** before pasting (`compose_scenes.py:13`, `paste_duplicate(..., mirror=True)`, l. 523–553); the only call site (l. 748) keeps the default. The twins are therefore identical up to reflection, not identical. A reviewer who opens the released stimuli will notice. See edit 14. |
| "A gripper detector" | **Name it** | OWLv2 (`detect_duplicates.py:46`, `google/owlv2-base-patch16-ensemble`; SAM for masks, l. 50), queried with "robot gripper / robot arm / robotic arm"; score ≥ 0.15; boxes centred in the lower quarter of the frame are discarded; image-centre fallback (`compose_scenes.py:105–121, 578–608`). Neither model is in `refs.bib`; name it with a `\note{cite}`. |
| Frame size 256×256 | Not verified | I could not confirm this quickly from the runner; keep it out of the main text until checked. |
| B0 numbers (16/32 facets, 36/12 specifications, 1,000 draws) | Cannot verify | Update when `analysis_plan.md` is frozen. Note that edit 2.2 above changes the 36/12. |
| **New:** TF32 | **Unreported setting** | `scripts/run_gpu.py:69` sets `torch.backends.cuda.matmul.allow_tf32 = False`. This is a compute setting that matters on A100 and GH200; add it to App. A "Runs" and card item 6. |

### §4 Claims softened

I agree with every softening listed. Four more are needed:

1. "the evaluation finds correct grounding" (l. 57) → "the word moves the arm toward the named twin" (W1).
2. "Wording is a fork, not noise, because the training data use the probed words in more than one role" (l. 187). This reads as causal; replacement in edit 5.
3. "they are rarely written down" (l. 57). Support it with B7-lite or the lit-A observation.
4. "One conclusion survives every one-at-a-time variation" is right; keep the qualifier wherever the claim is repeated until B1 exists.

### §5 Weakest points

1. **Curve pending.** Agree it is the biggest risk; the triage is in edit 4. A one-at-a-time audit plus a pre-declared design is acceptable at a workshop *if the text does not claim the curve*.
2. **Hook mixes pitfalls and forks.** Agree, and it is worse than you say: three of the four abstract verdicts are forks, and naive mirror scoring is missing from the hook. Lead with mirror scoring, the direction-blind statistic and dx; demote wording and precision to "choices that change the question" (edit 1).
3. **p-values near .05.** Pre-empt in three moves (edit 3): table-wide Holm (seven tests survive, including both sides of the mirror-scoring reversal); the label-swap invariance as the structural argument against the direction-blind statistic; and an explicit statement that per-row p-values are descriptive.
4. **dx straw man.** It is not one if you say why it is natural. Image coordinates put x horizontal; robot base frames conventionally put x forward. Bridge documents neither, and Open X-Embodiment does not align frames. The 98-frame check is the remedy any evaluation can run (edit 11).
5. **Scope.** Turn it into a strength by separating structural findings from empirical ones (edit 6):
   - *structural*, true for any model by construction: sign, naive mirror scoring, label-swap invariance of the contrast, edge-token fold;
   - *empirical*, model-specific in size: precision, readout, threshold, wording, prefix.

   B10 (ECoT) is not needed for a workshop paper.
6. **App. D gaps.** Fill the missing counts from `evidence.json`, or drop the absent-noun and "move" rows from Table VII; they are not used in the main text. Cells marked "--" look unfinished.

### §6 Space

- I agree with the candidate cuts in your order. Two more are larger than any of them:
  1. delete the nine `\report{}` lines (about 13 column lines), since Table II lists the same items;
  2. shorten §V "The curve" to three lines, since App. E has the design (about 6 lines).
- Keep the layer band in Fig. 1: it is the paper's most explicit link to the workshop.
- The full space ledger for my edits is at the end of Part 3.

---

## Part 3. Top edits (ranked by impact)

Additions are paid for by the cuts named in each edit. The ledger at the end shows that the plan fits in 4 pages.

### 1. Lead with same-output reversals, and separate "selection" from "direction"

- **Location:**
  - Abstract (l. 48).
  - Intro ¶2 (l. 57, "Holding the model and the 340 scenes fixed … rarely written down.").
  - Conclusion (l. 254, "moved a language verdict between correct and reversed grounding").
- **Problem:** W1. The current hook invites "different questions, different answers", omits the strongest example (21/25 under naive scoring) and uses "grounds" in a sense the conclusion denies.
- **Change: abstract** (about 165 words, close to the current 158):

```latex
Holding fixed one model (OpenVLA-7B) and 340 scenes (real BridgeData~V2 frames with a composited twin object), we vary one evaluation choice at a time. On identical outputs, a plausible but wrong scoring of mirrored scenes finds the word moving the arm toward the named twin in 21 of the 25 scenes where ``left'' and ``right'' move it apart; the correct scoring finds 4 of 25 (both $p<.001$). On identical outputs, a direction-blind contrast reports significant grounding where the signed effect is significantly reversed, and reading the wrong action axis makes the stimuli seem to fail. We separate \emph{pitfalls}, conventions with a right answer that papers should report with the check that verified them, from \emph{forks}, choices without one, which call for a specification curve: 4-bit inference changes 58\% of lateral action tokens but no headline verdict, while the wording sets the direction of the small remaining word effect. One conclusion survives every one-at-a-time change: the model does not reliably select the named twin. We map a reporting card onto the workshop's checklist and release the probe, stimuli and outputs.
```

- **Change: intro ¶2** (replaces l. 57 up to "Fig.~\ref{fig:pipeline} places …"; about the same length):

```latex
Choices that leave every model output unchanged can reverse the verdict (bf16 inference unless marked). Mirroring images is a common way to cancel a layout bias. After a horizontal flip, ``the mug on the left'' still asks for image-left motion, but names the other twin. Scored against the twin each instruction names in the image the model sees, the word moves the arm toward that twin in only 4 of the 25 opposite scenes (one twin on each side of the gripper) in which the two instructions move it apart ($p=.0009$): the effect is reversed. Scored by negating the expected direction, as mirror augmentation negates demonstrated actions, the same outputs give 21 of 25 ($p=.0009$): significant grounding. On the reference outputs, a direction-blind contrast reports grounding ($+10.7$ points, $p=.039$) while the signed effect points the wrong way ($-0.15$ bins, $p=.011$), and read on the wrong action axis the stimuli seem not to work ($p=1.0$). Other choices change the question rather than the scoring: the wording sets the direction of the small remaining word effect (``pick up the left \{noun\}'': 20 of 26 toward the named twin; ``\dots\ on the left side of the table'': 3 of 24), and 4-bit inference changes 58\% of lateral action tokens but no headline verdict.
```

- **Change: conclusion** (l. 254):

```latex
\head{Conclusion.}With the model's outputs fixed, scoring and readout choices alone moved the verdict between significant movement toward and away from the named twin, between significance and null, and between working and failing stimuli. For token-based policies the readout from logits to commands is part of the action space, and an evaluation's conventions are part of the substrate. Pitfalls need reporting with the check that verified them; forks need a specification curve.
```

- **Vocabulary rule for the whole paper:**
  - *selection*: both instructions correct, the robust "no";
  - *direction*: which way the word pushes when it splits the actions, the fragile one;
  - reserve "grounding" for the hypothetical verdicts of other evaluations.
- **Space:** about +1 line. Pay with edit 5's intro cut.

### 2. Add one figure that shows the effect

- **Location:**
  - If B1 slips, replace the Fig. 2 placeholder (l. 202–208).
  - If B1 lands, insert a one-column figure in §V and slim Table I (edit 7).
- **Problem:** W5. No figure shows an effect, so a skimmer cannot see the verdicts flip.
- **The key fact.** The paper's argument has a two-dimensional structure that one plot shows. In every row, opposite both-correct equals *split rate × right-way share* exactly. For example, ref: 18.8% × 33.3% = 6.2%; object wording: 27.4% × 74.2% = 20.4%. Data are in item A of the appendix to this review.
- **Plot specification:**
  - One point per Table I row.
  - *x* = % of opposite scenes in which the two instructions move the arm apart (100 − same sign).
  - *y* = % of those splits that go toward the named twins, with Wilson 95% intervals.
  - A dashed line at y = 50.
  - Dotted iso-curves x·y = 10% and 20% (= opposite both-correct, i.e. selection).
  - A note "grounded model: (100, 100)" in the corner.
  - Colours: pitfalls vermillion, forks blue, reference black.
  - Filled markers for rows that share the reference's model outputs (dx, sign, zero, argmax, thr0, thr2, sub_det); open markers for rows with different inputs or weights (nf4, mir, mir_naive, wordings).
  - Arrows ref→sign ("sign only") and mir→mir_naive ("scoring only").
  - Labels on the two wording extremes.
  - The dx point can be omitted, since its "toward" is undefined; mention it in the caption.
- **Caption:**

```latex
Each point is one row of Table~\ref{tab:oat}. Under every choice the two instructions move the arm apart in 8--27\% of opposite scenes, so selection ($x\cdot y$, dotted; opposite both-correct) stays below 21\%. Which way those splits go ranges from 12\% to 84\%; arrows join analyses of identical outputs. A direction-blind contrast reads only the horizontal axis, relative to same-side scenes.
```

- **Space:** +15 column lines, or about +7 net if it replaces the placeholder. Pay with edit 8 (the `\report` lines, −13) and edit 7 (two rows of Table I).

### 3. Pre-empt the statistics attack

- **Location:**
  - §II "Reference analysis" (l. 122), after "… between same-side and opposite scenes.".
  - §III direction-blind paragraph (l. 173), "\pending{B3: which statistic separates which behaviour, in simulation.}".
  - §V "Fragile." (l. 197).
- **Problem:** W3. The reader cannot tell which verdicts survive correction, and the direction-blind contrast is attacked on the weakest ground (switching significance) instead of the strongest (invariance).
- **Change (a), §II addition:**

```latex
Per-row $p$-values in Table~\ref{tab:oat} are uncorrected and treat scenes as independent: they show what a single-path analysis would report. Across its 48 tests, Holm correction keeps seven: the split and signed tests of both mirror scorings and of the table-side wording, and the signed test of ``the left \{noun\}''. The reversals we lead with survive; the reference's signed effect and every direction-blind contrast do not.
```

- **Change (b), §III direction-blind** (replaces the `\pending{B3…}`):

```latex
The contrast is unchanged when every ``left'' label is swapped with ``right'', which turns grounded behaviour into its reverse, so it cannot separate the two at any sample size; it tests whether the word's effect depends on the layout, not its direction.
```

- **Change (c), §V "Fragile":**

```latex
\head{Fragile.}The direction-blind contrast, which is borderline in every row ($-1.8$ to $+15.7$ points; no $p$ below .008), so that almost any choice flips its significance; and the direction of the residual word effect, which the wording sets.
```

- **Optional, offline:** replace the Wilcoxon p for the signed median with a base-frame sign-flip permutation p (Question 1).
- **Space:** about +3 lines. Pay by cutting "Twin items should be scored as a set~\cite{thrush2022winoground}." (−1) and the bin-width sentence in §III, "Taking a bin as … (App.~\ref{app:defs})." (−2); App. B already has it.

### 4. Remove the overclaims tied to pending work, and triage the pending items

- **Location:**
  - l. 61: "and a multiverse over measurement choices" and "this is the first specification-curve analysis of a robot-learning evaluation, and the first aimed at the measurement layer of a VLA language probe".
  - l. 63, contribution (ii).
  - Table I `prefix` and `screen` rows (l. 155–156).
  - §V "The curve" (l. 199).
  - §VI B7 (l. 246); §VII B5/B6 (l. 252).
- **Problem:** W2.
- **Change, l. 61:** "…we add twins with same-side controls on real frames, readouts of the action-token distribution, and a one-at-a-time audit of twelve measurement choices." (Count your rows.) Then:
  - **If B1 lands:** keep "this is the first specification-curve analysis of a robot-learning evaluation (Fig.~\ref{fig:curve})".
  - **If not:** "…but to our knowledge, as of early October 2026, no robot-learning evaluation has varied the measurement choices beneath a verdict and reported which verdicts survive (App.~\ref{app:related})."
- **Change, positioning:** add one clause:

```latex
Unlike pipeline audits of closed-loop success~\cite{choi2026vlaeval,wang2026industrialvla}, we audit a language verdict within one model and separate conventions with a right answer from choices that change the question.
```

- **Change, contribution (ii) if B1 slips:** "a fork analysis with a pre-declared specification-curve design (App.~\ref{app:curve})".
- **Triage by the 6 Oct freeze:**
  - B1 without the prefix facet: offline, so do it.
  - B3: replaced by edit 3(b).
  - B7-lite: desk work, so do it.
  - B2, B5, B6: if they are not run, delete the `prefix` and `screen` rows from Table I (keep them in Table IV as "not run"), and state B5 and B6 as limitations without `\pending`.
  - Shorten §V "The curve" to: "The specification curve crosses the forks with every pitfall at its correct value, one facet per question, with a label-flip joint test per facet (App.~\ref{app:curve})."
- **Space:** −2 Table I rows (about −4 column lines) and about −6 lines in §V.

### 5. Make the fork definition consistent; trim Type E/N/U from the intro; finish the wording fork

- **Location:**
  - Abstract ("forks, legitimate choices that define different estimands").
  - Intro ¶3 (l. 59, from "\emph{Forks} are legitimate choices…" to "…unclear research questions~\cite{auspurg2021has}.").
  - Fig. 1 caption (l. 110).
  - §IV opener (l. 179).
  - §IV wording (l. 187, "Wording is a fork, not noise, because the training data use the probed words in more than one role.").
- **Problem:** W6. Also, the wording fork's second branch, "a word across phrasings", is named but never computed.
- **Change, intro ¶3** (replaces the sentences named above):

```latex
\emph{Forks} have no single right answer. Some change the question: the executed (argmax) and the expected action, or one phrasing and the word across phrasings, are different estimands, that is, different quantities a claim can be about~\cite{lundberg2021what}. Others, such as the decision threshold, are interchangeable ways to answer one question. Papers should report forks as a specification curve, a factorial ablation over analysis choices plotted in sorted order with the choices marked beneath~\cite{simonsohn2020specification}: one curve per question, with interchangeable choices varied within it~\cite{delgiudice2021travelers}.
```

- **Change, Fig. 1 caption:** "forks (F, dashed blue; choices without a single right answer)".
- **Change, §IV opener:** "A fork has no single right answer. Readout, prefix, precision and wording change the question; the threshold and the scene subset are interchangeable ways to answer it; screening is uncertain." Introduce the Type E/N/U labels here or in §V, once.
- **Change, wording:**

```latex
Wording is a fork rather than noise: each template's direction holds in all four runs and exceeds the paraphrase floor (App.~\ref{app:wording}), and the training data use the probed words in more than one role. A claim about ``left'' made with one template therefore has the scope of that template; pooled over the four templates, the signed effect is \pending{pooled estimate, existing outputs}.
```

- **Space:** about −3 lines in the intro; ±0 in §IV.

### 6. Say what transfers beyond this probe

- **Location:** §VII (l. 252). Replace "Continuous action heads remove the token map, bins and prefix~\cite{kim2025finetuning,black2025pi0}." and add a paragraph before "\head{Conclusion.}".
- **Problem:** W4. The lesson stops at OpenVLA. A controls or hardware reader needs to know what carries over to continuous and flow heads, chunked actions, closed-loop scoring and other datasets.
- **Change** (about 7 lines):

```latex
\head{What transfers.}Three findings hold for any model and horizon by construction: a flipped sign or a negated mirror target reverses every directional verdict, and a sign-agreement contrast is unchanged when ``left'' and ``right'' are swapped, so it cannot tell grounding from its reverse; all three apply to closed-loop scoring of which object the arm reaches. Undocumented action frames are a dataset property~\cite{oneill2024open}; checking that demonstrated motion points toward the manipulated object verifies one with a hundred frames. Continuous heads remove the token map, bins and prefix~\cite{kim2025finetuning,black2025pi0} but not the readout fork: a flow-matching head executes one sampled chunk, whereas a probe may average samples, so sample count, integration steps, seed and the chunk step read become card items. Precision and GPU type changed 58\% and 8.1\% of lateral tokens but no headline verdict: report action-level agreement, not only success. Logging the 256-way distributions costs about 7~kB per prediction and makes the readout and threshold forks re-analysable offline.
```

- **Space:** +7 lines. Pay with the moved sentence (−1.5), edit 5 (−3) and edit 4's §V cut.

### 7. Make Table I readable at a glance

- **Location:** Table I (l. 127–159; caption l. 131).
- **Problem:**
  - Verdicts are visible only after decoding three test columns.
  - Rows `tok` and `w255` duplicate `ref`.
  - Same sign is redundant with the new figure.
  - The dx row's "image-right" means negative dx, which only App. B says.
  - The tags are relative to a weakly reversed reference.
- **Change:**
  - Add a **Verdict** column for the direction-aware result:

    | Verdict | Rows |
    |---|---|
    | toward*, from the signed test | sign, w_object |
    | toward†, survives Holm | mir_naive, w_prenominal |
    | away* | ref, zero, argmax, thr0, thr2, sub_det |
    | away† | mir, w_table_side |
    | null | nf4 |
    | null, no layout effect | dx |

    Use "toward/away" rather than "grounded/reversed", keeping "grounded" for selection.
  - Drop rows `tok` and `w255`; §III states both in a sentence, and Table IV keeps them.
  - Drop the Same-sign column; it is the figure's x-axis and stays in Table IV.
  - Retag dx as "strength".
  - Make /255 the reference threshold.
  - Footnote the dx row: "negative dx".
  - Add the `gpu` row (edit 15).
  - Cut the caption to two sentences; the test names move to App. C.
  - Fallback if space runs out: move Table I to the appendix (Table IV already holds every row) and keep the figure plus a six-row excerpt.
- **Space:** −2 rows net.

### 8. Make the reporting card checkable and map it fully onto the checklist; pay with the `\report` lines

- **Location:** Table II (l. 211–240); §VI (l. 246); the nine `\report{…}` lines in §III–IV (l. 167, 169, 171, 173, 181, 183, 185, 187, 189).
- **Problem:** W7.
- **Change:**
  - Add a **Check** column, at most six words each:

    | Item | Check |
    |---|---|
    | 1 | demonstrated motion toward object (98 frames) |
    | 2 | dimension with known behaviour (gripper) |
    | 3 | readout agreement; share of ties |
    | 4 | forced-prefix decode |
    | 6 | token agreement with native precision |
    | 7 | same-side scores unchanged by the transform |
    | 8 | paraphrase floor |
    | 9 | role counts in training language |
    | 10 | approval rate per condition |
    | 11 | positive control on probed axis; label-swap test (a direction-aware statistic flips sign) |
    | 12 | joint test per facet |

  - **Re-map the categories:**

    | Item | Organisers' category |
    |---|---|
    | 1 | [kinematics; action spaces] |
    | 5 | [controller gains and rates] (control rate; which step is read) |
    | 7 | [observations; simulation] (composited stimuli) |
    | 10 | [reset distributions] (the evaluated scene set) |

    Only items 11–12 then need the proposed "evaluation protocol" category.
  - **Add details:** "effects in physical units" to item 1; "full distributions logged?" to item 3; "TF32" to item 6. Replace the "Here" entries "gripper" and "context" with "none on \dy" and "–".
  - **§VI:** add a cost sentence: "Most items are settings and cost nothing to report; items 1 and 6 need one extra pass over about a hundred frames or the stimuli; item 12 is offline once distributions are logged."
  - **Delete the nine `\report{}` lines;** end each paragraph with "(card item $n$)".
  - If the Check column makes the table too tall for one column, make Table II a `table*` at the top of p. 4.
- **Space:** about −13 lines (report lines) +8 (Check column, cost sentence) = −5 net.

### 9. State the physical unit; call the threshold a deadband; say the effects are small

- **Location:**
  - §II "Model." (l. 118), after "We read the lateral component \dy\ of the first action.".
  - §IV threshold paragraph (l. 189).
  - Table I header "Signed, bins".
- **Problem:** W4. A controls reader's first question is unanswered. Answered, it becomes a lesson: measurement choices decide verdicts when effects are a fraction of the action interface's resolution, which is where current VLA language evaluations operate.
- **Change, §II** (confirm that dy is in metres first):

```latex
One bin is 0.33~mm of commanded lateral motion per 5~Hz control step~\cite{walke2023bridgedata}, so the word effects below are fractions of a millimetre: properties of the model's output, not of executed motion, and the regime in which measurement choices decide verdicts.
```

- **Change, §IV:** "The decision threshold acts as a deadband (0, 0.33 or 0.65 mm per step)." Optionally add an offline row with a deadband of about 1 mm (3 bins).
- **Change, Table I header:** "Signed, bins (1 bin $=$ 0.33 mm)".
- **Space:** +2 lines. Pay with edit 12's intro ¶4 compression.

### 10. Mirror paragraph: give the plain rule, say why naive scoring is natural, restore the out-of-distribution caveat

- **Location:** §III l. 171 (whole paragraph); App. E Type E list (l. 455).
- **Problem:** "swapped and then negated" is correct but opaque: on opposite scenes the two operations cancel, which is exactly why naive negation is wrong. The paragraph never says why anyone would score naively, which invites "straw man". The mirrored run, where the reversal is strongest, has lost its out-of-distribution caveat.
- **Change** (replacement paragraph, about +1 line):

```latex
\head{Mirror scoring (observations; direction).}Mirroring images is a common way to cancel a layout bias. The rule is to score each instruction against the twin it names in the image the model sees: after a horizontal flip, ``the mug on the left'' still asks for image-left motion but names the other twin, which is why grounding codebases swap the words when they flip~\cite{kamath2021mdetr,deng2021transvg}. Mirror augmentation instead negates the demonstrated lateral action~\cite{zhuang2025mirrorduo}; negating the expected direction in the same way, without changing which twin each word names, exchanges the splits that go the right way with those that go the wrong way. In the bf16 mirrored run, 4 of 25 splits go the right way under correct scoring and 21 of 25 under naive scoring (both $p=.0009$); the signed effect flips from $-0.48$ to $+0.48$ bins. A significantly reversed result is scored as significant grounding. Mirrored robot images are out of distribution~\cite{zhuang2025mirrorduo}, so we treat them as an exploratory replication.
```

- **Change, App. E:** move "images" from Type E to Type U.
- **Space:** +1 line. Pay by dropping the both-correct clause (3.5% → 18.3%; it is in Table I) and the report line (edit 8).

### 11. dx paragraph: explain why the error is natural

- **Location:** §III l. 167, after "Open X-Embodiment does not align frames across datasets~\cite{oneill2024open}.".
- **Problem:** open question 5.4. Without a reason, reading dx looks like a straw man.
- **Change** (first verify that component 0 is forward; see Question 7):

```latex
Image coordinates put $x$ horizontal, whereas robot base frames conventionally put $x$ forward \note{cite: ROS REP~103}, so an evaluator who reasons in image terms reads component~0 as lateral.
```

- **Also:** end the paragraph with "a check any evaluation can run (card item~1)", and retag the heading "(action space; direction, strength)".
- **Space:** +2 lines. Pay with the Fig. 1 trims (edit 13).

### 12. Define terms before use; cut jargon; put the substrate thesis first

- **Location:**
  - Intro ¶1–4 (l. 55–61).
  - Abstract, last sentences.
  - §V labels (l. 195): "Neither grounded", "Not lexical".
  - §IV "prenominal".
- **Problem:** W5 (terms used before §II defines them).
- **Change:**
  - **Definitions in intro ¶2 (edit 1):**
    - opposite scenes: "one twin on each side of the gripper";
    - splits: "the two instructions move the arm apart";
    - direction-blind contrast, at first use: "do the instructions split more often in opposite than in same-side scenes?".
  - **Specification curve:** glossed as a "factorial ablation over analysis choices" (edit 5); the phrase is the call's own.
  - **§V labels:** "Neither grounded" → "No selection"; "Not lexical" → "The word rarely decides the direction".
  - **Main text:** quote the wording instead of saying "prenominal".
  - **Intro ¶1:** move the conclusion's thesis sentence ("For token-based policies the readout from logits to commands is part of the action space…") to the end of intro ¶1, so a controls reader meets the substrate claim before the VLA specifics.
  - **Intro ¶4:** compress the first sentence:

```latex
Measurement choices already decide LLM evaluations~\cite{sclar2024quantifying,wang2024answer,dutta2024accuracy}, and 2026 audits show that the evaluation and deployment substrate moves VLA success~\cite{choi2026vlaeval,wang2026industrialvla,islam2026deployment,xu2026vlaquantbench} and that closed-loop instruments can certify absent language following~\cite{kirouane2026measuring}.
```

- **Space:** about −2 lines net.

### 13. Fig. 1: show the stimulus logic and the verdicts

- **Location:** Fig. 1 (l. 66–112).
- **Problem:** Fig. 1 is a legible, useful map, and the layer band is the paper's clearest link to the workshop. But it takes about a quarter of p. 2 and shows neither the design that non-VLA readers most need nor any verdict.
- **Change:**
  - **Stimulus box:** replace the text with two pictograms:
    - opposite: twin · gripper · twin;
    - same-side: gripper · twin · twin;
    - each with the grounded prediction in black (opposite: arrows apart; same-side: arrows the same way) and the word-shortcut prediction in grey (always apart);
    - a 1 cm real thumbnail if it fits.
  - **Verdict box:** list four outcomes with the identical-output choice that produced each: "toward: naive mirror scoring"; "away: correct mirror scoring"; "'grounding': direction-blind contrast"; "stimuli fail: dx".
  - **Stimulus F box:** shorten to "wording; images; scenes". It is five lines today and sets the figure's height.
  - **Caption:** cut to two lines.
  - **Pending tags:** drop the `*pending` tags if B2 and B6 are not run.
- **Space:** about neutral; the F-box trim pays for the pictograms.

### 14. Describe the stimuli and the compute stack exactly; show examples

- **Location:**
  - Intro l. 55 ("two identical objects (twins)").
  - §II "Stimuli." (l. 120, "A second instance of an object in a real Bridge frame is pasted beside the original").
  - App. A "Scenes." (l. 266, "A gripper detector gives…") and "Runs." (l. 274).
- **Problem:** see the verification table in §2.3 above. The pasted copy is mirrored, the detector is unnamed, TF32 is off but unreported, and no stimulus image appears anywhere.
- **Change:**
  - **Intro:** "two copies of one object (twins)".
  - **§II:** "A copy of an object already in a real Bridge frame is cut along its mask, mirrored and pasted beside it, so that the twins, identical up to reflection, lie on opposite sides of the gripper or both on one side".
  - **App. A detector:** "An open-vocabulary detector (OWLv2 \note{cite}), queried for the arm, gives the gripper's horizontal position; boxes centred in the lower quarter of the frame are discarded; without a detection, the image centre is used."
  - **App. A Runs:** add "TF32 matmuls disabled".
  - **New appendix figure:** four example scenes (opposite, both-left, both-right, mirrored) and one rejected composite.
- **Space:** main text ±0; the appendix is unlimited.

### 15. Isolate the hardware fork from existing outputs and make it a headline

- **Location:** §IV "Precision and GPU" (l. 185, from "4-bit inference more than halves…" to "…differ on 8.1\% of lateral tokens."); Table I.
- **Problem:** the GPU effect is shown on tokens (8.1%) but never on the verdict, and `nf4log` confounds GPU with readout. Table III says the GH200 NF4 replay logged the 255-token readout, so a clean row needs no new run.
- **Change:**
  - Add row `gpu`: NF4, 255-token readout, A100 logs vs GH200 replay.
  - Text:

```latex
At the same precision and software, an A100 and a GH200, which also differ in CPU architecture and driver, disagree on 8.1\% of lateral tokens; the verdict \pending{does / does not} change (row \rowid{gpu}).
```

  - Give the median size of a changed token (bins and mm) next to "58%".
  - Shorten the memory sentence to a clause: "(4-bit, which more than halves memory~\cite{kim2024openvla}, is the usual choice on small GPUs)".
- **Space:** +1 table row and −1 line of text.

### Space ledger (column lines, approximate)

| Edit | Adds | Cuts |
|---|---|---|
| Title subtitle to one line | | −2 |
| 1 Hook | +1 | −1 (abstract) |
| 2 Effect figure | +15 (or +7 net replacing the placeholder) | |
| 3 Statistics | +3 | −3 (Winoground; bin-width sentence) |
| 4 Overclaims and pending | | −4 (two Table I rows), −6 (§V curve) |
| 5 Forks | | −3 |
| 6 What transfers | +7 | −1.5 (moved sentence) |
| 7 Table I | +2 (`gpu` row) | −4 (`tok`, `w255`) |
| 8 Card | +8 | −13 (`\report` lines) |
| 9 Units | +2 | |
| 10 Mirror | +1 | −1 |
| 11 dx | +2 | |
| 12 Jargon and intro ¶4 | +1 | −3 |
| 13 Fig. 1 | +1 | −1 |
| 15 GPU | | −1 |
| **Total** | **≈ +43** | **≈ −44** (−52 if the figure replaces the placeholder), plus the current ~6 lines of slack |

If B1 lands and needs space, put the per-facet summary table in the appendix and give it one sentence in §V. Do not add a second main-text figure.

---

## Appendix to this review: numbers behind edits 2, 3 and 9

### A. Decomposition for the effect figure (from Table I; both-correct = x·y reproduces Table I exactly)

| Row | Kind | Same outputs as ref? | n (opp., decided) | Splits | Toward twin | x = split % | y = toward % | Wilson 95% for y |
|---|---|---|---|---|---|---|---|---|
| ref | – | – | 112 | 21 | 7 | 18.8 | 33.3 | [17.2, 54.6] |
| dx | P | yes | 142 | 12 | 6 | 8.5 | 50.0 | [25.4, 74.6] |
| sign | P | yes | 112 | 21 | 14 | 18.8 | 66.7 | [45.4, 82.8] |
| zero | P | yes | 107 | 20 | 7 | 18.7 | 35.0 | [18.1, 56.7] |
| mir_naive | P | = mir outputs | 115 | 25 | 21 | 21.7 | 84.0 | [65.3, 93.6] |
| argmax | F | yes | 108 | 20 | 9 | 18.5 | 45.0 | [25.8, 65.8] |
| thr0 | F | yes | 151 | 35 | 16 | 23.2 | 45.7 | [30.5, 61.8] |
| thr2 | F | yes | 90 | 17 | 6 | 18.9 | 35.3 | [17.3, 58.7] |
| nf4 | F | no | 116 | 19 | 7 | 16.4 | 36.8 | [19.1, 59.0] |
| mir | F | no | 115 | 25 | 4 | 21.7 | 16.0 | [6.4, 34.7] |
| w_prenominal | F | no | 107 | 26 | 20 | 24.3 | 76.9 | [57.9, 89.0] |
| w_object | F | no | 113 | 31 | 23 | 27.4 | 74.2 | [56.8, 86.3] |
| w_table_side | F | no | 113 | 24 | 3 | 21.2 | 12.5 | [4.3, 31.0] |
| sub_det | F | subset | 66 | 11 | 3 | 16.7 | 27.3 | [9.7, 56.6] |

(`tok` and `w255` coincide with `ref`.) The Wilson intervals ignore base-frame clustering, which is presumably negligible with about one opposite scene per base frame; check this (Question 1).

### B. Holm correction across Table I

- 16 rows × 3 tests = 48 tests; the Bonferroni threshold is .00104.
- **Seven tests survive Holm at .05:**
  - w_table_side: signed (1.1e−5) and splits (.00028);
  - w_prenominal: signed (.00075);
  - mir and mir_naive: signed (.00077 each) and splits (.00091 each).
- **Holm stops at** sub_det signed (.002 > .00122).
- **Not surviving:**
  - the reference's signed effect (.011);
  - every direction-blind contrast (minimum p = .008, `thr2`);
  - `sign` (.011);
  - `w_object` (.010 and .011).
- **Robustness of the count:** the same seven survive if `tok` and `w255` leave Table I (42 tests), or if a null `gpu` row is added (45 tests).

### C. Physical units (dy: q01 = −0.04170, q99 = +0.04086; assuming metres; Bridge control rate 5 Hz)

| Quantity | Bins | mm per step |
|---|---|---|
| One bin (/254; /255 gives 0.324) | 1 | 0.325 |
| Reference signed effect | −0.15 | 0.05 |
| Prenominal wording | +0.35 | 0.11 |
| Table-side wording / mirrored run | −0.48 | 0.16 |
| Positive control (paste moved) | +1.4 | 0.46 |
| Paraphrase floor, left↔right vs grab↔take (bf16) | 2.46 vs 1.37 | 0.80 vs 0.45 |
| Prefix association, dx differs vs dx shared | 3.8 vs 0.95 | 1.24 vs 0.31 |
| 1st–99th percentile range of dy | 254 | 82.6 |
| Logging cost: 7 dims × 256 tokens × 4 bytes per prediction | | ≈ 7 kB (≈ 200 MB for the 27,966 predictions) |
