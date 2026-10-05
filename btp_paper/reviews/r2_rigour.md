# R2 rigour audit of the BtP draft

**Audited:** `btp_paper/main.tex` as of 18:25, 2 Oct 2026 (main text and appendix) and `btp_experiments_plan.md` as of 18:34 (B5 already updated to the 640×480 frames).
**Sources used:** `evidence/claims_ledger.md`, `evidence/one_at_a_time.{md,csv}`, `evidence/evidence.json`, the three lit reports, `refs.bib`, `brief.md` §5, the compiled `main.pdf`, and the repository code that produced the evidence (read only).
**New checks:** `reviews/r2_rigour_checks.py` writes `reviews/r2_rigour_checks.json` (about 3 s; imports `btp_evidence.py` read-only and modifies nothing else). It provides the base-frame structure, frame-level sign-flip tests, Holm tables, the grab/take placebo, the argmax zero-bin check, folded-readout versions of two appendix numbers, and the Table VII gaps. Numbers marked **[r2]** come from that file. They are not in the ledger yet, so the draft may use them only after `btp_evidence.py` and the ledger carry them (brief §5).

**Counts:** 4 BLOCKER · 25 MAJOR (12 draft, 13 plan) · 44 MINOR.

**Overall.** Every number in the main text matches the ledger, Table 1 and `evidence.json`. All cells of Tables I and IV were checked programmatically against the CSV. The problems are in four other places:
- **Interpretation and inference.** The hook's significant direction-blind result does not survive the paper's own Holm correction. The signed effect is called "grounded" although a word-to-direction shortcut also makes it positive. p-values ignore the base-frame clustering that the card says they respect.
- **One factual error about the stimuli** (frame size and source).
- **The novelty claim** depends on the specification curve, which is still a placeholder.
- **The plan.** Its joint test and outcomes would repeat, at curve level, the identifiability problem the paper criticises.

---

## 1. Issues, ranked by severity

### BLOCKER

**R-B1 · BLOCKER · App. F card item 7 (l. 487); §VII (l. 252); App. A "Scenes" (l. 266)**
- **What is wrong.** "256×256 frames passed to the model's processor" is false.
  - The frames are 640×480 RGB from the Open X-Embodiment release of Bridge (`gs://gresearch/robotics/bridge/0.1.0/`, `data.py` l. 31–53), passed straight to the HF processor.
  - OpenVLA trained on `bridge_orig` (256×256), and its README calls the OXE version "out of date (as of 12/20/2023)".
  - §VII mentions only the JPEG and Lanczos differences, so it understates the gap: the source release and the resolution also differ.
- **Evidence.** Orchestrator correction, 2 Oct 18:45. `brief.md` §8 correction (l. 173–177). Plan B5 (updated).
- **Fix.**
  - l. 487: "640$\times$480 frames from the Open X-Embodiment release of BridgeData~V2~\cite{oneill2024open}, passed straight to the model's processor, which resizes them to 224$\times$224. OpenVLA trained on the 256$\times$256 frames of the original BridgeData~V2 release, and its Bridge evaluation code applies a JPEG round trip and a Lanczos resize to 224$\times$224 (effect of both differences: \pending{B5 pre-processing run})."
  - l. 252: "Our frames come from the Open X-Embodiment release of Bridge (640$\times$480), not the 256$\times$256 release OpenVLA trained on, and skip the JPEG round trip and Lanczos resize of its Bridge evaluation code \pending{B5 pre-processing run};"
  - l. 266: "Each scene is a real BridgeData~V2 frame (640$\times$480, Open X-Embodiment release~\cite{oneill2024open}) into which …"
  - Table 2, item 7: "Source release and resolution; pre-processing; edits; transforms and re-scoring (…)".

**R-B2 · BLOCKER · Intro (l. 57); §III (l. 173); contradicted by §IV (l. 187)**
- **What is wrong.** The hook's lead example of the direction-blind pitfall rests on an uncorrected p that the paper's own correction rejects.
  - l. 57 says the contrast "reports significant grounding (+10.7 points, p=.039)". §III repeats .039 for the original and table-side wordings.
  - l. 187 says "in bf16 the Holm-corrected sign-agreement contrast detects none of them". "Them" includes the original wording.
  - So the paper says both that this contrast is significant and that it is not.
- **Evidence.**
  - bf16 contrast p by wording (`evidence.json: wording_grid.cf1_convention`): original .0386, prenominal .0127, table side .0386, object .607, absent noun .481, move .832.
  - Holm over the six wordings: prenominal .076, original .19, table side .19. Holm over the five alternative wordings: prenominal .064, table side .15 **[r2 D_holm.bf16]**. Nothing survives.
  - **Exact invariance.** The contrast does not change when the two instructions' labels are swapped. Swapping them in a random half of the scenes leaves it at +10.67 (p = .0386), while the splits move from 7/21 to 9/21 **[r2 K]**.
  - **Placebo.** Two instructions that name the *same* twin ("grab the {noun} on the left" vs "take the {noun} on the left", ladder paraphrase pair, bf16, original images) give **+11.7 points, McNemar p = .012** (77 pairs, 11 discordant). On mirrored images: +7.8, p = .039. At 4-bit: +2.4 (p = .77) **[r2 G]**.
- **Fix.**
  - l. 57: "With the original wording, a direction-blind contrast points to grounding (+10.7 points; $p=.039$ before correction, not after) while the signed effect points the wrong way ($-0.15$ bins, $p=.011$); at 4-bit NF4 on the same GPU both are null ($p=.55$ and $.22$)."
  - Better still, lead with the example that survives correction (4-bit, table side): "+14.8 points, McNemar $p=.004$, $.02$ after Holm, while its signed effect is reversed, $-0.53$ bins, $p=4\times10^{-7}$". Then add the placebo once it is in the ledger: "two instructions that name the same twin give $+11.7$ points".
  - l. 173: append "(uncorrected)" to both p = .039 values and add the invariance and placebo sentences (R-M8).

**R-B3 · BLOCKER · §II (l. 122) "positive is grounded"; Table I caption (l. 131) "positive = grounded"; Table VII caption (l. 400); App. E O1 (l. 449, 459)**
- **What is wrong.** The signed left−right effect is positive under grounding *and* under a word-to-direction shortcut.
  - §II (l. 120) itself says that only same-side scenes separate those two accounts.
  - The signed median pools all 340 pairs, so a positive value cannot be called "grounded".
  - A reviewer who reads l. 120 next to l. 122 will see the contradiction. It also propagates into the curve's O1 (R-P2).
- **Evidence.** `reanalysis.py` l. 203–205 defines `oriented = (a - b) * (-IMAGE_X_TO_LATERAL_SIGN)`, with the comment "Positive when 'left' moves further image-left than 'right' does, the direction both grounding and a word-to-direction mapping predict."
- **Fix.**
  - l. 122: "(positive when ``left'' moves further image-left than ``right'', as grounding predicts; a word-to-direction shortcut predicts it too, which only same-side scenes separate)".
  - l. 131 and l. 400: "positive = toward the named twin's side".
  - App. E O1: "the signed word effect toward the named twin's side", plus O4 from R-P2.

**R-B4 · BLOCKER (conditional on B1) · Intro (l. 61) novelty; contribution (ii) (l. 63); §V heading**
- **What is wrong.** "This is the first specification-curve analysis of a robot-learning evaluation" claims an analysis that the draft does not contain: Fig. 2 is a placeholder (l. 205).
- **Evidence.** `lit/lit_methods.md` §(d), "Strongest defensible statement", presupposes a curve that has actually been computed. `open_questions.md` §2.1.
- **Fix.**
  - If B1 lands before the freeze: keep the sentence and add "(Fig.~\ref{fig:curve})".
  - Otherwise, l. 61: "… this is the first audit of the measurement layer beneath a VLA language verdict, with a pre-declared specification-curve design (App.~\ref{app:curve}) for which we found no precedent in robot learning".
  - Otherwise, l. 63: "(ii)~a fork analysis and a pre-declared specification-curve design faceted by estimand".

### MAJOR — draft

**R-M1 · MAJOR · Abstract (l. 48)**
- **What is wrong.**
  - "One conclusion survives every one-at-a-time variation" contradicts §V (l. 195). "Not lexical" (opposite same-sign 69–92%) also holds in every row, and "follows the layout" fails only in two pitfall rows.
  - "Neither does any direction-blind statistic" is false. The same-sign rate is direction-blind and robust.
- **Evidence.** Tables I and IV: opposite same-sign 68.8–91.5% in every row. Ledger, "Which choices move which conclusion".
- **Fix.** "Two conclusions survive every one-at-a-time variation: the model does not reliably select the named twin, and its two instructions mostly move the same way; a third, that the action follows the layout, fails only under two pitfalls. The direction of the residual word effect, which the wording sets, does not survive, and neither does the significance of the direction-blind contrast."

**R-M2 · MAJOR · App. B Statistics (l. 305); card item 11 (l. 491); App. E (l. 447); §IV Holm counts (l. 187); Table VII "Sig. counts"**
- **What is wrong.**
  - The 340 scenes come from 191 base frames, and 149 frames hold two scenes **[r2 A]**.
  - The Wilcoxon p for the signed effect treats scenes as independent. Only the CIs and the McNemar contrast respect frames.
  - Yet the card says "unit: scene, clustered by base frame", and App. E says "CIs and permutations respect base-frame clusters".
  - At frame level, two of the paper's counted results lose significance after within-run Holm.
- **Evidence.** Two-sided sign-flip tests of the Wilcoxon statistic, flipping per base frame, 20,000 flips **[r2 L]**:

  | Cell | Scene-level Wilcoxon p | Frame-level p | Frame-bootstrap CI of the median |
  |---|---|---|---|
  | bf16, original wording | .0115 | .016 | [−0.52, −0.001] |
  | bf16, object | .010 | .021 | [+0.017, +1.03] |
  | bf16, prenominal | .00075 | .0014 | |
  | bf16, table side | 1.1e−5 | 5e−5 | |

  - Within-bf16 Holm with frame-level p: original .065 and object .065 (scene-level: .041 and .041).
  - Within-run Holm counts become **table side 4/4, prenominal 3/4, object 2/4, original 1/4** (draft: 4/3/3/2). Global-Holm counts are unchanged (4/2/2/1).
  - Split tests are already effectively frame-level: only 3 frames hold two opposite scenes.
- **Fix.**
  - Report signed-effect p from base-frame sign-flip tests. Add them to `btp_evidence.py`, then the ledger.
  - App. B: "Signed median: … two-sided sign-flip test of the Wilcoxon statistic with signs flipped per base frame (191 frames). Splits: binomial (opposite scenes come from distinct frames in all but three)."
  - l. 187: "… 4 of 4 runs for the table-side wording, 3 of 4 for the prenominal wording, 2 of 4 for the object wording and 1 of 4 for the original".
  - Card item 11: "unit: scene; $p$ from base-frame sign-flip tests; CIs from a base-frame bootstrap".
  - If scene-level tests are kept instead, write "$p$-values treat scenes as independent" and delete "clustered by base frame" from item 11.

**R-M3 · MAJOR · Multiplicity: App. B (l. 305) vs App. D (l. 439); l. 185; Table I**
- **What is wrong.**
  - App. B defines Holm over "six wordings" (including the original). App. D applies Holm to the contrast over "five alternative wordings" (original excluded). The main text never names the family.
  - Table I's p-values are uncorrected, and several decisive ones sit near .05.
- **Evidence.** What survives (scene-level p from `evidence.json`; frame-level p from [r2 L, D]):

  | Claim (location) | Raw p | Frame-level p | Holm within run | Holm over 24 cells | Status |
  |---|---|---|---|---|---|
  | bf16 original, signed −0.15 (l. 57, 173) | .011 | .016 | .041 / frame .065 | .10 / frame .16 | **fragile** |
  | bf16 original, contrast +10.7 (l. 57, 173) | .039 | — | six: .19 | — | **fails** |
  | bf16 table side, contrast +9.6 (l. 173) | .039 | — | six .19; five .15 | — | **fails** |
  | 4-bit table side, contrast +14.8 (App. D) | .0042 | — | six .021; five .017 | .096 | survives within run |
  | Prenominal splits 20/26, bf16 (l. 57) | .009 | ≈ same | splits .047 | — | survives, barely |
  | Prenominal signed +0.35, bf16 | .00075 | .0014 | .0037 / .0067 | .0104 / .021 | robust |
  | Table side 3/24; signed −0.48, bf16 | .00028; 1.1e−5 | 5e−5 | .0017; .0001 | .0002 / .0012 | robust |
  | Mirrored 4/25; signed −0.48, bf16 (l. 57, 171) | .0009; .00077 | .002 | .0055; .0031 / .0072 | .0104 / .025 | robust |
  | Object 23/31; signed +0.25, bf16 (Table I) | .011; .010 | .021 | .047; .041 / .065 | .10 / .19 | fragile |
  | Argmax signed, r = −0.16 (l. 181) | .038 | .044 (Pratt .032) | single test | — | uncorrected only |
  | thr2 .008, zero .012, sub_notrel .049 (Table I/IV) | — | — | — | — | uncorrected only |

  What each test is (none of this is stated in the draft):
  - All tests are two-sided.
  - **Binomial.** Null: right-way and wrong-way splits are equally likely, conditional on the number of splits.
  - **Wilcoxon.** Null: the left−right differences are symmetric about 0, over all 340 pairs including same-side scenes; zero differences are dropped.
  - **McNemar.** Null: discordance is symmetric across base frames that have one decided scene of each kind (75 pairs in the reference). It is invariant to the direction of the word effect.
- **Fix.**
  - App. B: "Multiplicity. Signed effect and split direction: Holm within each run over the six wordings, and for the signed effect also over all 24 wording × run cells. Sign-agreement contrast: Holm within each run over the five alternative wordings (the original wording is the reference). Table~\ref{tab:oat} rows are sensitivity analyses and are not corrected; $p$-values near .05 there do not survive any correction. All tests are two-sided."
  - Main text: tag .039 (l. 57, 173, 181, 185, 189) and .038 (l. 181) as "uncorrected".
  - Lead every "reversed" example with the robust cases: table side and mirrored images.

**R-M4 · MAJOR · §II (l. 122–124) and the hook: no physical scale for the effects**
- **What is wrong.**
  - The verdicts "grounds them backwards" and "points the wrong way" rest on median shifts of a fraction of a bin. The reader is never told what a bin is.
  - One bin is 0.000325 Bridge units. Bridge actions are in metres (the ledger's role statistics use a 0.005 = "5 mm" floor, L§10), so one bin is ≈0.33 mm per 0.2-s step.
  - In those units:
    - reference signed effect −0.15 bins ≈ 0.05 mm;
    - prenominal +0.35 ≈ 0.11 mm;
    - table side −0.48 ≈ 0.16 mm;
    - positive control +1.4 ≈ 0.45 mm.
- **Evidence.** bf16 baseline: median |dy| 4.3 bins (IQR 1.6–12.1). The left/right |Δ| median is 2.42 bins folded and 2.46 with the 255-token readout **[r2 scale check, N]**.
- **Fix.** Add to §II: "One bin is about 0.3\,mm of lateral end-effector motion in the first 0.2-s step; the median lateral action is 4.3 bins and the two instructions differ by a median of 2.4 bins, so the residual word effects below are a fraction of a bin." Add 4.3 and 2.4 to the ledger first. Also confirm the units with the data owner.

**R-M5 · MAJOR · §II (l. 122)**
- **What is wrong.** "A 4-bit run logged on an NVIDIA A100 replicates the reference" overstates what the logs show. They replicate the headline rates, but their contrast (+4.1, p = .549), splits (6/19) and signed effect (−0.02, p = .485) do not match the reference's +10.7 (.039), 7/21 and −0.15 (.011). The intro (l. 57) itself says that at 4-bit "both are null".
- **Evidence.** T1: nf4log vs ref.
- **Fix.** "A 4-bit run logged on an NVIDIA A100 replicates the reference's headline rates (App.~\ref{app:oat}); its logs carry only a 255-token readout."

**R-M6 · MAJOR · Intro (l. 57) "the evaluation finds correct grounding"; Conclusion (l. 254)**
- **What is wrong.** With prenominal wording, opposite both-correct is 18.7% [12.0, 26.4]. The 26 splits are a minority of the 107 decided opposite scenes. §V says the model "does not reliably select the named twin".
- **Evidence.** T1: w_prenominal.
- **Fix.**
  - l. 57: "the evaluation finds the residual word effect pointing toward the named twins when the instruction reads …, and away from them for …".
  - l. 254: "moved a language verdict between toward and away from the named twin, significance and null, …".

**R-M7 · MAJOR · §IV Prefix (l. 183): "these pairs carry 82% of the summed dy change"**
- **What is wrong.** "Carry" is causal. The ledger flags exactly this word.
- **Evidence.** `claims_ledger.md` §5: "'Carries' is causal, and the prefix-control experiment has not been run".
- **Fix.** "and these pairs account for 82\% of the summed $|\Delta\dy|$".

**R-M8 · MAJOR · §III direction-blind paragraph (l. 173)**
- **What is wrong.** The paragraph argues from one borderline p (R-B2). It leaves out the exact, assumption-free version of its own point and the placebo that makes the point unanswerable.
- **Evidence.** [r2 K, G], as listed under R-B2.
- **Fix.** Insert after the Type S sentence: "Sign agreement is unchanged when the two instructions' labels are swapped in any scene, so the contrast carries no information about the direction of the word effect. Two instructions that name the same twin (``grab'' and ``take the \{noun\} on the left'') give $+11.7$ points (McNemar $p=.012$) in the reference run, as much as the original wording." Add the placebo to `btp_evidence.py` and the ledger first.

**R-M9 · MAJOR · Bin width /254 vs /255**
- **Where.** l. 92 (Fig. 1 label), l. 122, l. 143 (row w255), l. 169, l. 299, l. 330, l. 451, l. 455; thr2 "(0.00065)" at l. 148 and l. 329.
- **What is wrong.**
  - App. E (l. 451) fixes the pitfall at "grid step /255".
  - The reference and App. B (l. 299) use /254 = 0.000325 as "one bin".
  - §III (l. 169) calls /254 the deviation.
  - So the paper's reference sits on the wrong side of its own pitfall.
- **Evidence.**
  - T1: w255 = T1: ref in every column.
  - Only 2 bf16-replay predictions lie between the two widths, both neutral-condition same-side scenes outside every Table I statistic. Two more are in the 4-bit ladder paraphrase pair **[r2 E]**.
  - `open_questions.md` §2.3.
- **Fix.**
  - Switch the reference to the grid step. l. 122: "decided if $|\dy|\geq$ one grid step, $(q_{99}-q_{01})/255$".
  - l. 299: "The decision threshold, one bin, is the grid step $(q_{99}-q_{01})/255=0.0003238$; taking a bin as $(q_{99}-q_{01})/254=0.000325$, which treats $q_{01}$ and $q_{99}$ as the outermost centres, changes no statistic (row \rowid{w254})."
  - Rename row w255 to w254 ("Bin width: one bin taken as /254, not the grid step /255") in Fig. 1 and Tables I and IV.
  - Re-run `btp_evidence.py` with `ONE_BIN = STEP_255` and confirm every row (thr2 becomes 0.000648).

**R-M10 · MAJOR · Stimulus description: l. 55 "two identical objects (twins)"; l. 120 "A second instance … is pasted beside the original"; l. 266**
- **What is wrong.** The second twin is the object's own pixels, cut out, **mirrored** and pasted. The twins are mirror images, not identical objects. The difference is visible for asymmetric objects (handles, text) and matters in a paper about mirroring.
- **Evidence.** `compose_scenes.py` docstring (orchestrator, 2 Oct). `brief.md` §8 correction.
- **Fix.**
  - l. 55: "to choose between an object and a mirrored copy of it (twins)".
  - l. 120: "The object's own pixels are cut out, mirrored and pasted beside it in a real Bridge frame, so that the twins lie on opposite sides of the gripper or both on one side (App.~\ref{app:stimuli})."
  - l. 266: "… into which a mirrored cutout of an object already in the frame is pasted, so that the scene holds mirror-image twins. An open-vocabulary detector (OWLv2 \note{cite}) locates the objects and the gripper; where the gripper is not found, the image centre is used."

**R-M11 · MAJOR · §IV Wording (l. 187)**
- **What is wrong.** "Wording is a fork, not noise, because the training data use the probed words in more than one role" is a causal explanation. `brief.md` §5 forbids it here: "say only that the training data uses the words differently by role and give the numbers".
- **Fix.** "Wording is a fork, not noise: swapping ``left'' and ``right'' changes the action more than a paraphrase does (App.~\ref{app:wording}), and the training data use the probed words in more than one role."

**R-M12 · MAJOR · "Strength" tags and §V "Fragile" (l. 181, 185, 189, 197)**
- **What is wrong.** Every switch in the contrast's significance moves only a handful of base frames. A "strength" change inferred from p crossing .05 near a borderline reference is not evidence that the choice changed the effect: the difference between significant and non-significant is not itself significant. The reader is not told this.
- **Evidence.** Discordant splits, derived from Table IV (pairs, discordant, points):

  | Row | Discordant pairs (same-side only vs opposite only) | p |
  |---|---|---|
  | ref | 10 vs 2 | .039 |
  | nf4 | 7 vs 4 | .55 |
  | argmax | 8 vs 3 | .23 |
  | thr0 | 18 vs 16 | .86 |
  | sub_det | 4 vs 1 | .38 |
  | mir | 12 vs 11 | 1.0 |
  | zero | 10 vs 1 | .012 |
  | thr2 | 8 vs 0 | .008 |
- **Fix.**
  - Add to §V "Fragile": "the contrast's significance turns on a few base frames (10 against 2 discordant pairs in the reference, 7 against 4 at 4-bit)".
  - Define "strength" as a change in significance or size without a sign change.

### MAJOR — plan (`btp_experiments_plan.md`)

**R-P1 · MAJOR · B0 "Joint test", B1 step 2; App. E l. 447 vs l. 461**
- **What is wrong.** Labels "flipped at random within each scene" ignore the 191 base-frame clusters. That contradicts "permutations respect base-frame clusters" two lines earlier. It is anticonservative when frames carry their own word responses (R-M2).
- **Fix.** "Flip the left/right labels jointly within each base frame: all scenes and both image transforms of a frame together. Use the same frame-level sign-flip p for the per-specification verdicts, and 10,000 flips."

**R-P2 · MAJOR · B0 Outcomes; App. E l. 449, 459; B1 "Reading the result"**
- **What is wrong.**
  - (a) O1–O3 cannot separate grounding from a word-to-direction shortcut, because both make O1 and O2 positive (R-B3). A curve built only on them would report a pure shortcut as "toward the named twin", the mirror image of the dominant-sign problem the paper rightly avoids.
  - (b) Under label flips, O3 = k/n with k ~ Binomial(S, ½), so its permutation test is O2's. The observed 6.2% is even below the flip mean S/(2n) = 9.4%.
  - (c) "Nothing reaches reliable twin selection" has no decision rule.
- **Fix.**
  - Add **O4**: the sign-agreement contrast for left/right *minus* the same contrast for the grab/take pair, paired by base frame. It separates grounding (O4 > 0) from a shortcut (O4 < 0) and from layout-only behaviour (O4 ≈ 0). For the reference: +10.7 vs +11.7 points, so O4 ≈ −1.
  - Call a facet "toward the named twin" only if O1 > 0 and O4 > 0.
  - Drop O3 from the joint test. Pre-declare "no reliable selection" as: frame-bootstrap upper limit of O3 below 50% in every specification.
  - Use B3 to check O4 under realistic noise. The left/right |Δ| is larger than grab/take's (2.4 vs 1.4 bins), so O4 needs calibration.

**R-P3 · MAJOR · B0 joint test (one-sided "toward"); App. E l. 461**
- **What is wrong.** The paper's main story is that wording sets the direction, with table side and the original wording reversed. But the joint statistics are all one-sided toward the named twin, with p = share of null values ≥ observed. For the reversed facets every p ≈ 1, so the curve would report them as "null" rather than "away".
- **Fix.** "Per facet, report the median effect and Stouffer's $Z$ with signs kept and two-sided permutation $p$, plus the shares of specifications significant toward and away (each with its own permutation $p$). A facet's direction is the sign of its median effect, pre-declared."

**R-P4 · MAJOR · B0 Type E: images (original / mirrored / pooled); thresholds; subsets**
- **What is wrong.**
  - Mirrored robot images are out of distribution (MirrorDuo, `lit_substrate.md` A6), and neither OpenVLA nor Octo trains with flips (A2.11, A3).
  - The mirrored estimates differ from the reference beyond noise: signed −0.48 vs −0.15; splits 4/25 vs 7/21; both-right image-right 86% vs 99% (T1: mir vs ref).
  - "Pooled" counts each scene twice. Act2Answer calls this "mildly anticonservative" (`lit_vla.md` C1).
  - Thresholds move O2 from 7/21 to 16/35 (thr0) and 6/17 (thr2). So Type E status needs a prior argument, not an assumption.
- **Fix.**
  - Make images Type U (an exploratory panel) or a Type N facet.
  - Drop "pooled", or average the two views per scene before testing.
  - Give one sentence each justifying thresholds and subsets as Type E: measurement precision for thresholds, label quality for subsets.

**R-P5 · MAJOR · B0 thresholds × argmax readout**
- **What is wrong.** Under argmax, physical zero falls in bin 128, which decodes to −0.31 bins (App. B l. 303).
  - With "any nonzero" (thr0), the zero-motion token counts as a decided image-right action.
  - At 1 bin, bin 127 (−1.3) is decided but bin 129 (+0.69) is not.
  - So the argmax × threshold cells embed the zero-point pitfall the curve claims to hold fixed.
- **Evidence** **[r2 F]**.
  - 8.7% of bf16 baseline argmax predictions sit on bin 128.
  - Argmax × thr0 as designed: image-right 48/66/96%, splits 16/32.
  - With bin 128 treated as zero: 41/63/97%, splits 11/24.
- **Fix.** "Under argmax, decisions are counted in grid steps from bin 128: decided iff $|b-128|\ge k$, sign from $b-128$; `any nonzero' means $b\neq128$."

**R-P6 · MAJOR · B0/B1 multiplicity across facets; permutation count; which outcome enters the joint test**
- **What is wrong.**
  - There are 16 facets (32 with the prefix), × 3 statistics, × up to 3 outcomes, with no correction across facets.
  - It is not said which outcome the joint statistics use.
  - 1,000 flips give a minimum p ≈ .001, which is coarse for Holm over 32+ facets.
- **Fix.** "One primary joint test per facet: Stouffer's $Z$ on O1, two-sided, Holm across facets. Other statistics are descriptive. 10,000 label-flip datasets."

**R-P7 · MAJOR · B2 decomposition and logging**
- **What is wrong.**
  - (a) The "direct effect", the mean over t* ∈ {dx_left, dx_right} of [E(dy | left, t*) − E(dy | right, t*)], is the average of the two natural direct effects (CDE at each instruction's own token). Indirect = total − direct is then the average natural indirect effect. That is a valid symmetric decomposition for a deterministic decoder, with no cross-world assumption, but the two CDEs and their difference (the word × dx interaction) should be reported too.
  - (b) Medians, which the draft reports, are not additive. The decomposition needs means or sums of oriented differences, with frame-bootstrap CIs.
  - (c) The "marginal" variant mixes over each instruction's *own* p(dx | w). It is therefore a total effect under sampling, not a direct effect. The plan's formula also omits the division by the covered mass that the ledger's version has.
  - (d) Only `cf1` is logged, so argmax × forced facets and the dist row cannot be computed.
  - (e) NF4 is "if time allows", so App. E's "32 facets" may not exist.
  - (f) O2 "with dx fixed" needs one common token, because splits cannot be averaged over t*.
  - (g) "Neutral" is undefined.
- **Fix.**
  - Report CDE(dx_left), CDE(dx_right), their mean and difference, as means of oriented differences with frame-bootstrap CIs, plus the share of pairs whose direct effect keeps the total's sign.
  - Define a marginal *direct* effect as Σₖ π(k)[E(dy | left, k) − E(dy | right, k)], with π = p(dx | neutral), normalised by covered mass.
  - Log the argmax and the full 256-way dy distribution for every variant.
  - Use t* = the neutral token for O2.
  - Name the neutral instruction (e.g. "pick up the {noun}").
  - Run NF4, or state "16 facets".

**R-P8 · MAJOR · Cross-cutting: all new runs on A100s; every reference number is from the GH200**
- **What is wrong.** At NF4, a GPU change alone alters 8.1% of lateral tokens (L§4). As designed:
  - B2's prefix row and B6's screen row change GPU and the choice together;
  - B2's gate compares a hand decode with GH200 numbers;
  - B4's analysis uses "the bf16 GH200 reference as the baseline";
  - B9's dist row comes from A100 sidecars.
- **Fix.**
  - "Every A100 row is compared with an A100 bf16 eager baseline on the same stimuli (B4 setting 1). B2's gate compares with `generate()` on the same A100. B4 reports A100-vs-GH200 as its own row. Table~I marks A100 rows."
  - B6: score the approved scenes on the same A100 too.

**R-P9 · MAJOR · B3 simulation (fills l. 173)**
- **What is wrong.** The expected truth table is wrong in two cells:
  - with realistic non-directional instruction noise, the direction-blind contrast also fires under layout-only behaviour (model 4), as the placebo shows (+11.7, p = .012);
  - the directional statistics fire "toward" under the lexical model (model 3).
  - "Rejection" for both-correct is undefined.
  - The calibration omits the component the two instructions share within a scene. That component sets the same-sign rates.
- **Fix.**
  - Expected table: the contrast fires for models 1, 2, 5, 6 and for 4; O1/O2 fire toward for 1, 3, 5 and away for 2, 6; only O4 separates 1 from 3.
  - Calibrate:
    - a scene-level shared component;
    - non-directional instruction noise matched to the grab/take |Δ| (bf16 median 1.38 bins, folded);
    - layout means matched to 44/71/99%;
    - a base-frame effect;
    - empirical heavy-tailed residuals;
    - the decided fraction per layout.
  - Report size under model 4 for scene-level vs frame-level tests.
  - The label-flip invariance (exact) and the placebo already answer half of B3 without simulation.

**R-P10 · MAJOR · B6 (fills Table I screen row; App. E Type U)**
- **What is wrong.**
  - Step 3 weights scenes by the inverse approval rate *of their layout*. That is constant within a layout, so it cannot change any opposite-scene statistic (O2, O3, opposite both-correct). Only pooled quantities move.
  - Step 1 scores rejected scenes by recorded labels, but hand relabelling changed 53 of 340 (15.6%) frozen-set layouts (L§7). A rejected-vs-approved difference would mix screening with label error.
- **Fix.**
  - Step 3: reweight by within-layout predictors of approval (object, `separation_px`, gripper source), or drop it.
  - Step 1: compare rejected and approved scenes *both* scored by recorded arrangement, on the same GPU (R-P8), and print step 2's agreement next to the row.

**R-P11 · MAJOR · B9 (fills Table IV row `dist`, App. E l. 453)**
- **What is wrong.**
  - The sidecars come from A100 runs (R-P8).
  - They store p(dy | greedy dx), so "sampled actions (T = 1, 20 samples)" would sample dy with dx frozen at its greedy token. That is not what a sampling deployment does.
- **Fix.**
  - Compute the median and mode on B4 setting 1, against its own expected value.
  - Get sampled readouts from an ancestral-sampling GPU run, or from B2's marginal.

**R-P12 · MAJOR · B7 reporting audit (fills §VI l. 246)**
- **What is wrong.**
  - κ between two LLM agents is not inter-rater reliability: agents can share errors.
  - Items 1–4 are N/A for most closed-loop papers.
  - The sample is the authors' own review list.
- **Fix.**
  - Freeze the paper list and the N/A rules before coding.
  - Have two humans double-code at least 25% of the papers and report human–agent κ.
  - Report k/N per item, excluding N/A.

**R-P13 · MAJOR · B1 display vs Fig. 2 caption (l. 206)**
- **What is wrong.** The caption promises "one panel per estimand facet". B1 step 3 plots "the O1 estimates sorted … coloured by wording" in one top panel. That pools Type N facets into one sorted curve, which the paper says it avoids (Del Giudice & Gangestad, E20).
- **Fix.** Fig. 2 shows the four wording facets (bf16, expected value, own prefix) as separate panels, plus a facet summary (one point per facet with its joint-test p). The other facets go in the appendix.

### MINOR

**Citations and quotations**
- **R-m1** l. 57, 505 — "remain largely undocumented" is not verbatim. The source has "their configuration in existing large-scale datasets remains largely undocumented" (`lit_substrate.md` A8). → "controller gains, whose configuration ``remains largely undocumented'' in large datasets~\cite{bronars2026tune}".
- **R-m2** l. 61 — Dutta et al. show *equal* accuracy with flipped answers (E89). They do not show that quantisation "decides" evaluations. → "prompt format moves accuracy by up to 76 points~\cite{sclar2024quantifying}, first-token readouts disagree with generated answers~\cite{wang2024answer}, and quantised models flip individual answers at equal accuracy~\cite{dutta2024accuracy}".
- **R-m3** l. 515 — Kurtic et al. show aggregate parity (E90), not flips. → "Compressed models match aggregate accuracy~\cite{kurtic2025give} yet flip individual answers~\cite{dutta2024accuracy}".
- **R-m4** l. 61 — IndustrialVLA-Bench argues that scores "can … reflect either policy capability or the evaluation path" (`lit_vla.md` C4). It does not show that the substrate moves success. → drop it from that list (it stays at l. 246 and l. 505).
- **R-m5** l. 59 — Auspurg & Brüderl reanalysed one many-analysts project (E23). → "traced the divergence in one many-analysts project mainly to an unclear research question".
- **R-m6** l. 59 — The pitfall rule comes from Simonsohn's admissibility criteria (E4–E5), not from Del Giudice & Gangestad. → "The split combines the admissibility rule of specification-curve analysis~\cite{simonsohn2020specification} with Del Giudice and Gangestad's typology~\cite{delgiudice2021travelers}: …".
- **R-m7** l. 165 — A wrong sign is not a "sensible test" of the question, which is criterion 1 rather than "statistically valid". → "is not a ``reasonable specification''~\cite{simonsohn2020specification}".
- **R-m8** l. 173 — Type S is a sampling-error concept (E29). → "akin to a sign (Type~S) error".
- **R-m9** l. 63 "do not forecast" → "forecast closed-loop success poorly", matching l. 505 and `lit_vla.md` C4.
- **R-m10** l. 501, 503, 519 — unhedged negatives ("not as evaluation readouts", "None places both twins…", "None has items…") → add "to our knowledge".
- **R-m11** l. 507 "millions of synthetic commands" → "nearly two million" (`lit_substrate.md` A4.13).
- **R-m12** l. 505 — QAIL (`park2024quantization`) partly reports action-level divergence (`lit_substrate.md` A7). → move it out of "success-only".
- **R-m13** l. 505 — CogACT is a diffusion action module, not a tokeniser or grid. → list it with flow- and diffusion-based heads.
- **R-m14** l. 118 — "greedily" is cited to OFT, which supports only autoregressive decoding. → "greedy decoding is the default in OpenVLA's released code".
- **R-m15** l. 118, 482 — "256 bins (255 centres)" reads as a contradiction. → "256 action tokens that decode to 255 bin centres".
- **R-m16** l. 252 — "Continuous action heads remove … prefix". It is parallel decoding that removes the prefix. → "Parallel continuous decoding~\cite{kim2025finetuning} and flow heads~\cite{black2025pi0} remove …".

**Definitions and consistency**
- **R-m17** Terms used before they are defined, or never defined in the main text:
  - the intro (l. 57) uses "splits", "signed effect", "direction-blind contrast" and "term-free" before §II;
  - the value of "one bin" is never given in the main text;
  - "headline" is never defined.

  → add short glosses at l. 57, and define "headline rates (same-sign, both-correct and image-right shares by layout)" in §II.
- **R-m18** App. B (l. 305) — sidedness is not stated. → "All tests are two-sided."
- **R-m19** l. 305 — the Wilcoxon drops zero differences (30% of pairs under argmax). → state it. Pratt's method gives p = .032 and the frame-level test .044, so the conclusion holds **[r2 C, L]**.
- **R-m20** l. 57 "at 4-bit precision" → "at 4-bit NF4 on the same GPU". §II introduces the A100 logs, whose signed p is .49.
- **R-m21** l. 167 — uses the 4-bit logs (86.1% vs 85.7%), while the intro uses bf16. → bf16: 83.1% vs 82.1%, Fisher p = 1.0 (`evidence.json: claim1…dx_cf0_bf16_own_bin.manipulation_check`).
- **R-m22** l. 57 "no longer depends on" → "is unrelated to" (the ledger's wording). dx still shifts when the paste moves (bf16 p = .0025, L§1).
- **R-m23** l. 197 — "zero point" does not switch significance (p .039 → .012), and neither does thr2 (.008). → "switches with the threshold (any nonzero action), readout, precision, images, subset and wording".
- **R-m24** l. 185 "from two or three wordings to none" → "from two wordings at 4-bit (three with the 255-token readout) to none in bf16, Holm over the five alternative wordings".
- **R-m25** l. 439 — "prenominal on 4-bit mirrored images (255-token readout)". The folded readout also rejects (Holm-adjusted .022) **[r2 D]**. → "(both readouts)".
- **R-m26** Table VII:
  - bf16 prenominal signed p ".0008" → ".0007" (folded .000746; Table I shows .00075). ".0008" is the 255-token value.
  - 4-bit object ".0003" → ".0002" (folded .000248).
  - Fill the gaps from `wording_grid.cf1_convention`:

    | Wording, run | Splits | Signed (p) | Both-correct | Contrast (p) |
    |---|---|---|---|---|
    | Absent noun, 4-bit | 10/21 | −0.37 (.0007) | 9.6 | −3.1 (.79) |
    | Absent noun, 4-bit mirrored | 1/18 | −0.49 (5e−6) | 0.9 | −11.3 (.19) |
    | Absent noun, bf16 mirrored | 6/18 | −0.32 (.0005) | 5.7 | −9.6 (.23) |
    | Move, 4-bit | 14/28 | −0.08 (.90) | 16.7 | +5.4 (.65) |
    | Move, 4-bit mirrored | 29/43 | +0.77 (.0001) | 29.3 | +6.0 (.48) |
    | Move, bf16 mirrored | 21/42 | +0.15 (.18) | 22.3 | 0.0 (1.0) |

    Mirrored-run contrasts:

    | Run | Original | Prenominal | Object | Table side |
    |---|---|---|---|---|
    | 4-bit mirrored | +2.7 (.82) | +17.1 (.004) | +8.3 (.17) | −3.9 (.63) |
    | bf16 mirrored | +1.3 (1.0) | +6.5 (.33) | −2.2 (.82) | +2.3 (.84) |
  - Once filled, the grid's maximum both-correct is 29.3% (4-bit mirrored, move), not 24.3% as `open_questions.md` §4 says. §V stays true because it is scoped to one-at-a-time rows.
- **R-m27** Table V (l. 361–365) and card item 6 (l. 486) — the GH200-vs-GH200 r and sign columns use the 255-token readout. Folded values: replay .709 / 90.1%, ladder .647 / 87.3%, unedited .714 / 92.4% (`evidence.json: claim4_precision.comparisons`). → switch, or caption "r: 255-token expected dy (the only readout in the A100 logs)". Card: "r=.71".
- **R-m28** App. D (l. 394) and card item 8 (l. 488) — the paraphrase floor uses the 255-token readout (`reanalysis.paraphrase_floor` → `paired(…, "c1")`). Folded values **[r2 N]**: 4-bit 2.57 vs 1.72 bins, 58.5%, p = .0003; bf16 2.42 vs 1.38, 59.1%, p = .0015.
- **R-m29** l. 294 — "The unedited frames enter this paper only through the token-map count" is contradicted by Table V row 5. "Unedited" also names two datasets: the 3,103 natural frames and the 98 axis frames. → add "and Table~\ref{tab:agree}", and use distinct names.
- **R-m30** Table III (l. 288) — "Ladder (6 wordings)" includes the paraphrase pair, whereas the Holm "six wordings" are the original plus five alternatives. → "Ladder (5 wordings + paraphrase pair)".
- **R-m31** Fig. 1:
  - the Scoring stage, which carries mirror scoring, sits in the "evaluation protocol" band, while §III (l. 171) and Table 2 item 7 place mirror scoring under observations;
  - "GPU" is not a Table I row, although the caption says the labels are rows.
- **R-m32** Table 2 item 5 (l. 226) — "Sc", but horizon was not varied. → "--".
- **R-m33** Card item 11 (l. 491) — "clustered by base frame" (see R-M2).
- **R-m34** References — internal provenance notes print in [17], [18], [21], [31], [34], [50], [51], [58], [65], [98], [115], [129]. Examples: "(proceedings page not checked)", "venue per the arXiv comment", a lab-page URL in [50]. → move them to an unprinted field such as `annote` in `refs.bib`.
- **R-m35** l. 461 "per-specification Z averaged with weight 1/√K" → "$Z=\sum_k Z_k/\sqrt{K}$, each $Z_k$ signed toward the named twin".
- **R-m36** l. 455 — "/254 … redundant: it changed nothing" justifies redundancy by the observed outcome. → justify it by construction: "differs from the grid step by 0.4\%".
- **R-m37** l. 463 — the variance decomposition lists "threshold" for O1, but O1 has no threshold.
- **R-m38** Table I "Changes" — the tags are never defined, and dx "scope" fits poorly: dx erases the layout gradient. → define the tags in the caption ("direction: a directional statistic changes sign; strength: significance or size changes without a sign change; scope: the scenes or conditions a claim covers change") and tag dx "strength".
- **R-m39** l. 48, l. 59 — "silently reverse or erase / change a verdict". tok and w255 change nothing on dy, and zero changes strength. → "can silently reverse or erase".
- **R-m40** l. 187 — the role statistics are used without their definition (single lateral word; first motion > 5 mm; L§10). → state it in App. A or card item 9.
- **R-m41** CIs (l. 317 caption) — the percentile cluster bootstrap undercovers rare rates. It is narrower than exact intervals that ignore clustering:

  | Row | Bootstrap CI | Exact CI |
  |---|---|---|
  | table side 3/113 | [0.0, 6.1] | [0.6, 7.6] |
  | sub_det 3/66 | [0.0, 10.6] | [0.9, 12.7] |
  | ref 7/112 | [1.8, 10.8] | [2.5, 12.5] |
  | object 23/113 | [12.6, 27.7] | [13.4, 29.0] |

  → note this, or use a cluster-adjusted Wilson or BCa interval. "Not reliable" still holds.
- **R-m42** l. 388 "5.1% of 680 baseline predictions" → add "(4-bit logs)". Similarly App. F item 1's model flip rates come from the 4-bit logs; say so.
- **R-m43** App. B (l. 299) — c is both a centre index and a centre value; use k for the index. l. 305 "when the median is a tie" → "when the median is zero".
- **R-m44** Remaining small items:
  - l. 63: give the anonymous URL, from an anonymised mirror. Do not use the public dataset or fork.
  - l. 48: "mapped to the workshop's checklist" → "mapped to the announced categories of the workshop's checklist". The checklist will be produced after the workshop (E75).
  - l. 254: "For token-based policies" → "For token-based policies such as OpenVLA".
  - Plan B1 step 4 refers to "the abstract's 'N specifications'", which the abstract does not contain.
  - Plan B6 step 4: κ needs a named second rater. Otherwise it is intra-rater agreement.

---

## 2. Plan coverage

Every `\pending{}` maps to a plan item. No pending lacks one. Three claims depend on plan items but carry no `\pending`:
- the novelty sentence (l. 61), which depends on B1 (R-B4);
- §V's "Fig. 2 will cross the forks …", which depends on B1;
- the anonymous URL (l. 63), which depends on the Wed 7 Oct release snapshot and needs anonymisation.

| `\pending` (line) | Plan ID | Adequate? |
|---|---|---|
| Table I, row prefix (155) | B2 | **No, as written.** A100 vs GH200 confound (R-P8). The row needs a named forced variant. "Marginal" is not a direct effect (R-P7). |
| Table I, row screen (156) | B6 step 1 | **Partly.** Recorded-label error (15.6%) and GPU confound. Compare rejected and approved scenes under recorded labels on the same A100 (R-P10). |
| §III, B3 sentence (173) | B3 | **No, as written.** The expected truth table is wrong for models 3 and 4; the calibration lacks the shared and non-directional noise components (R-P9). With the fixes, yes. The invariance and the placebo already fill most of the slot. |
| §IV prefix (183) | B2 | **Partly** (R-P7, R-P8). |
| §IV screening (189) | B6 | **Partly.** Step 3 reweighting cannot change within-layout rates (R-P10). |
| Fig. 2 (205) | B0 → B1 (+B2) | **No, as written.** Within-scene flips (R-P1), no O4 (R-P2), one-sided (R-P3), images as Type E (R-P4), argmax zero bin (R-P5), facet multiplicity (R-P6), pooled display (R-P13). |
| §VI, B7 (246) | B7 | **Partly.** Agent κ is not reliability; N/A rules are undefined (R-P12). |
| §VII, B5 pre-processing (252) | B5 step 1 | **Yes** after the 18:34 update (three paths), provided path (i) on the same A100 is the baseline and TF fidelity is checked. Fix the 256×256 text now (R-B1). |
| §VII, B6 relabelling (252) | B6 step 4 | **Unclear.** The second rater is unnamed (R-m44). |
| Table IV, prefix (344) | B2 | As for l. 155. |
| Table IV, screen (345) | B6 | As for l. 156. |
| Table IV, dist (346) | B9 via B4 | **No.** A100 sidecars; sampled readouts need ancestral sampling (R-P11). |
| App. E banner (445) | B0 | **Yes**, once App. E's internal contradictions are fixed before freezing (l. 447 vs 461; /254 vs /255, R-M9). |
| App. E prefix facet (453) | B2 | **Partly.** 32 facets need NF4 and argmax logging under forced prefixes (R-P7). |
| App. E distributional readouts (453) | B9 | As for l. 346. |
| App. E Type U, screened-out (457) | B6 | **Partly** (R-P10). |
| Card item 4 (484) | B2 | **Partly** (R-P7). |
| Card item 6, compute ladder (486) | B4 | **Partly.** The baseline must be A100 bf16 (setting 1), not the GH200 (R-P8). |
| Card item 7 (487) | B5 step 1 | **Yes** (after R-B1). |
| Card item 8, appended token (488) | B5 step 3 | **Yes**, compared on the same A100. |
| Card item 10, κ (490) | B6 step 4 | **Unclear** (R-m44). |
| Card item 12, curve (492) | B1 | **No, as written** (as for l. 205). |

---

## 3. Checks passed (safe as written)

- **Tables I and IV.** Every numeric cell matches `one_at_a_time.csv`, checked programmatically: rates, CIs, image-right shares, contrast with pairs and discordant counts, splits, signed medians and every p.
- **Tables V, VI and VII.** All populated values match `evidence.json` and ledger §2, §4 and §9, apart from the two Table VII p-values and the readout of Table V's r column (R-m26, R-m27). The signed and split significance counts (raw / Holm-run / Holm-24) reproduce exactly at scene level.
- **Main-text numbers.** Every main-text number matches the ledger and is attributed to the right run and readout:

  | Group | Numbers |
  |---|---|
  | Hook | 20/26, .009; 3/24, .0003; 4/25, .0009; +10.7, .039; −0.15, .011; .55 and .22; dx p = 1.0 (bf16: 83.1% vs 82.1%) |
  | Behaviour | 81.2%, 6.2% (7/112), 44/71/99; 83.8%, 5.1% (6/117), 38/71/94 |
  | Positive control | +1.4 bins, 73.2% of 149, p = 5.6×10⁻⁷ |
  | Axis and sign | 82/82/78, 86.1/85.7, 56/29/1, 14/21, −1.3 bins, 71→56, 96.9%, ≤57% |
  | Token map | 1.2% of 27,966; 0.995→0.30; 68%; >99.7% |
  | Mirror | 4/25, 21/25, ∓0.48, 3.5→18.3 |
  | Readout and prefix | 32.6%, 8.3%, .23, 30% ties, r = −0.16, .038; 59.7%, 3.8 vs 0.95, 82% |
  | Precision | 58% of 2,720; ≤2.5 points; ≤8.3 points; .55→.039; 8.1% |
  | Wording | 60–80%, 12.5–32%, 16–37%, 4/3/3/2 at scene level |
  | Bridge language | 3,574 / 3,333 / 183; 39.3%, 58.8%, 81.8%, 8.3% with their n |
  | Threshold and screening | .86; 54.1% (437/808) vs 45.8% (383/836), .0008; 141/340 (41.5%), 58%; 4.5% (3/66) |
  | §V | ≤ 20.4%; 69–92% |
- **Appendix A arithmetic.**
  - 210 + 437 + 173 = 820 approved; 781 + 26 + 17 = 824 rejected.
  - 340 − 141 = 199; 340 − 53 = 287; 182 − 141 = 41.
  - 2 × (2,720 + 8,160 + 3,103) = 27,966.
  - The "both" subset (detected and not relabelled) has 191 scenes [r2 A].
- **Appendix B decoder and zero point.**
  - ids 31745 and 31744 → centre 254; 31999 → centre 0.
  - Grid step 0.0003238.
  - Normalised zero −0.000424 = −1.30 bins.
  - Bin 128 → −0.31 bins; bin 129 → +0.69 bins.
  - 7.2% of logged argmax predictions sit on bin 127.
- **Robustness checks [r2].**
  - The McNemar pairing's arbitrary keep-first rule (`analysis._paired_contrast`) never applies in the reference. None of the 75 paired frames has two decided scenes of the same kind, and 400 random re-orderings all return +10.7, p = .039 [M].
  - The argmax signed-effect p holds under different zero handling (Pratt .032; sign test .044).
  - Table-side reversal, mirrored reversal and prenominal toward survive frame-level tests and both Holm families.
  - "Three cells with a significant positive contrast and backwards splits" reproduces with the folded readout [H].
- **/254 vs /255.** No Table I statistic changes. The four predictions between the two widths enter no reported statistic [E].
- **Reference readout.** The folded expected value renormalised over 256 action tokens, the prompt template and token 29871 are as stated (confirmed by the orchestrator).
- **Literature.** Each of these claims has a supporting quote in the lit reports:
  - STAGE concession; LIBERO-CF (3 cups, 1 vs 20 demos); RefGuard 33–46%; vla-eval 55 points; SmolVLA ONNX (INT8 → FP32; 41 → 75%); VLAQuantBench;
  - Kirouane "false conclusions"; Bridge annotation quote; OXE unaligned frames; Octo gripper; RT-2 constrained decoding; RT-1 256 bins;
  - MDETR/TransVG swaps; RefTR; Augment-the-Pairs; MirrorDuo out-of-distribution; Act2Answer "mildly anticonservative"; ECoT −y = "right";
  - OpenVLA's 4-bit claim (different checkpoint, unnamed type, FP4 default hedged as "our reading of the code");
  - Simonsohn E4/E8/E10/E11; Del Giudice & Gangestad E19–E21; Lundberg; Winoground; Bender data statements; Kress-Gazit; Gelman & Loken; many-analyst figures.
- **Novelty hedge.** "To our knowledge, as of early October 2026" is present (l. 61). R-B4 concerns its content, not its hedge.
- **Anonymity.**
  - PDF metadata has no author or title.
  - No names, institutions, course codes, cluster names, notebooks, "dissertation" or repository/dataset URLs appear in the PDF text; the GPUs are named only by model.
  - The Overleaf zip holds only `main.tex` and `refs.bib`.
  - `refs.bib` has no self-citation; the only "Lewis" is a QLoRA co-author.
  - The dual-submission footnote is present, verbatim from the brief. The companion paper is cited anonymously (`\anon`).
- **Bibliography.** All 177 cited keys exist in `refs.bib`; `main.blg` is clean. Merged-key metadata matches the lit reports. kim2024openvla, li2024evaluating and mandlekar2021what carry PMLR years with a CoRL-year note, which is acceptable.
- **Page budget.** The main text ends on page 4; References start on p. 4.
