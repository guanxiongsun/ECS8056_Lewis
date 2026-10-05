# R2 verification of draft v2

**Checked:** `main.tex` v2 (19:56), `main.pdf` (20:00), ledger §11.1–§11.13, the regenerated `one_at_a_time.{csv,md}`, `addenda.json`, both figure scripts and their outputs, and the plan (19:12).
**Method:**
- Every Table I and Table IV cell, including both `gpu` lines, was compared with the CSV by script.
- Every Table C (decomposition) cell was compared with `addenda.json: A5`.
- The Fig. 2 script's own `load()` assertions were run against the CSV; they pass.
- Table VII, the placebo table, the App. B/C/D text and the card were checked against ledger §11.

**Result:** 0 BLOCKER · 2 MAJOR · 10 MINOR. No number in v2 is wrong. Two values are rounded or expressed in a different bin unit from the rest of the paper (V-m2, V-m3).

## 1. Status of the R2 blockers and majors

| ID | Status | Where in v2 |
|---|---|---|
| R-B1 (256×256 frames) | RESOLVED | l. 128, 258, 272, 554; Table II item 7 (l. 224) |
| R-B2 (hook's p = .039) | RESOLVED | l. 57 (placebo hook), l. 175 (".039 uncorrected"; 4-bit table side with Holm .017); consistent with l. 193 |
| R-B3 ("positive is grounded") | RESOLVED | l. 130, 139, 321, 449, 516 |
| R-B4 (novelty claim) | RESOLVED | l. 61, 63 |
| R-M1 (abstract count) | RESOLVED | l. 48 (see V-m4) |
| R-M2 (clustering) | PARTLY | l. 130, 193, 321, 443, 449, 558 are done; l. 187 still quotes the scene-level argmax p (V-m1) |
| R-M3 (Holm families) | RESOLVED | l. 321, 329; "uncorrected" tags at l. 175, 187, 191 |
| R-M4 (physical scale) | RESOLVED | l. 126, 139, 315 |
| R-M5 ("replicates the reference") | RESOLVED | l. 130 |
| R-M6 ("finds correct grounding") | RESOLVED | l. 57, 260 |
| R-M7 ("carry") | RESOLVED | l. 189 |
| R-M8 (invariance + placebo) | RESOLVED | l. 175; Table (placebo) l. 492–503; l. 506 |
| R-M9 (/254 vs /255) | RESOLVED | l. 99, 126, 313, 347–348, 518, 522 (residual unit mix in Table VII: V-m2) |
| R-M10 (mirrored twins) | RESOLVED | l. 55, 128, 272 (the OWLv2 `\note` remains) |
| R-M11 (causal "because") | RESOLVED | l. 193 (but see V-M1) |
| R-M12 (discordant pairs, tag definitions) | RESOLVED | l. 139, 242 |

## 2. What was verified as correct

**Hook and abstract.** 21/25 and 4/25 (p = .0009); placebo +11.7 (p = .012) vs +10.7; dx +1.0 point, p = 1.0 (bf16); 58% of lateral tokens.

**§II.**
- One bin = 3.2×10⁻⁴, about 0.3 mm (§11.2).
- Median first lateral command 4.3 bins; left/right difference 2.4; paraphrases 1.4 (§11.3, §11.10).
- Positive control: 1.4 bins = 32% of the median command (§11.3).
- 191 base frames (§11.8).

**Table I.**
- Every cell matches the regenerated CSV, including the new `zero` row (6.2, 24/57/99, +12.0 (.012), 7/21) and `gpu` with its GH200 comparator.
- Verdict column: each entry is the sign of the signed effect at p < .05. Each holds whether p is computed per scene or per base frame.
- ‡ marks exactly the rows with a survivor in the 48-test Holm (§11.4): `mir_naive`, `mir`, `w_prenominal`, `w_table_side`.
- \* (dx) and † (argmax ties) are correct.

**Fig. 2.**
- All 14 plotted points equal the x, y values and Wilson CIs in A5 / Table C.
- Fill coding (reference outputs) is correct, and the two arrows join identical-output pairs.
- The text claims hold: x ranges over 8–27% and y over 12.5–84% (§V, l. 240).

**§V.**
- ≤ 20.4% (Tables I and IV).
- 69–92% same-sign. These numbers are in Table IV only, since Table I no longer has a same-sign column.
- 12 discordant of 75 paired frames.
- Five distinct Holm survivors (§11.4).

**§IV.**
- Frame-level within-run counts 4/3/2/1 (§11.9).
- Precision and GPU sentence (§11.1).
- 0.95 vs 3.8 bins, 82%.
- Screening numbers.

**Appendix.**
- Units: 0.324 mm, 1.6 mm/s, IQR, 84.1%; and 3.5%, 8.1%, 11.2% and 31.6% of the median command.
- Zero: 59/680 predictions on bin 128; 72→63%; contrast +6.9→0.0 and +11.7 vs +11.5 (§11.12).
- Statistics: 21 splits in 21 frames; bootstrap vs Wilson CI.
- App. A frames (149/42; 151 opposite scenes in 148 frames) and run stack (§11.1).
- Table C: every cell matches A5.
- Table V caption fixed.
- Table VII: every cell matches §11.7/§11.9. The two v1 p-value slips are fixed (.00075, .00025) and the gaps are filled.
- Placebo table (§11.10), the label-swap sentence (§11.6) and the App. D counts (§11.9).
- Card items 1, 6, 8, 10 and 11.

**Example-stimuli figure (`fig_examples.pdf`).**
- The four analysed panels are frozen-set scenes in the 340:
  - `c000086_opposite_left`;
  - `c000086_same_side_left_left` (same base frame 86, as the caption says);
  - `c000507_same_side_right_left`;
  - the flip of the first.
- Each is the first of its layout in ID order with a detected gripper, and each is labelled the same way in the analysis's geometry table.
- The fifth panel (`c000000_same_side_left_left`) is, as the caption says, the first composite rejected for an implausible paste. It is not in the frozen set, by design.
- The panel titles and the "left" instruction under each image match the caption.
- No identifying text appears in the images or the PDF text.

**Anonymity and layout.**
- `main.pdf` metadata is empty of author and title, and the figure PDFs' metadata does not reach it.
- No identifying strings appear in the PDF text.
- The references no longer print internal provenance notes.
- The main text ends on p. 4.
- Every v2 `\pending` maps to a plan item, including B4.1 (plan, 19:12).

## 3. Issues, ranked

### MAJOR

**V-M1 · §IV l. 193 ↔ App. F item 9 (l. 556): dangling, circular support for "the two roles predict opposite first motions".**
- **What is wrong.**
  - v2 deleted the role statistics from §IV and points to App. F.
  - App. F item 9 says "Role statistics (Sec. IV) use instructions with a single lateral word…", pointing back to §IV.
  - The destination numbers (39.3% vs 58.8%) appear nowhere in the paper. Only the starting-place numbers survive, in item 1.
- **Fix.** In l. 556 replace "Role statistics (Sec.~\ref{sec:forks}) use instructions with a single lateral word and a first motion above 5\,mm." with: "After a destination ``left'' or ``right'' the first demonstrated motion is image-left in 39.3\% ($n=1{,}693$) and 58.8\% ($n=1{,}463$) of episodes; after a starting-place ``left'' or ``right'', in 81.8\% ($n=22$) and 8.3\% ($n=24$) (single lateral word; first motion above 5\,mm)." (L§10.)

**V-M2 · "What transfers" (l. 256) overgeneralises.**
- **What is wrong.**
  - "the edge-token fold applies to any binned tokeniser" is unsupported. The fold comes from OpenVLA's 256-edge, 255-centre code (`lit_substrate.md` A2.3–A2.4). Nothing in the lit reports shows that RT-1/RT-2-style binning, or any other binned tokeniser, shares it.
  - "closed-loop twin tests inherit all three" is also too broad. A test that scores which twin is reached reads no action sign.
- **Fix.** "Some findings hold for any model by construction: a flipped sign or a negated mirror target reverses every directional verdict, and sign agreement ignores which instruction is which; closed-loop twin tests that mirror scenes or compare same-or-different outcomes inherit the last two. Every binned tokeniser needs its token-to-bin map checked at the edges, as OpenVLA's fold shows."

### MINOR

- **V-m1** (l. 187)
  - **Problem.** "p=.038 uncorrected" is the scene-level Wilcoxon. App. B (l. 321) says signed-effect p in the text are frame-level.
  - **Fix.** "(rank-biserial $-0.16$; Wilcoxon $p=.038$ over scenes, uncorrected)". Alternatively, add the frame-level value to the ledger (r2 check: .044) and cite that.
- **V-m2** (Table VII l. 449–483; App. D l. 443; card l. 555)
  - **Problem.** These use bins of 0.000325, while the paper defines one bin as the grid step (×1.00383). So the bf16-mirrored original signed effect reads −0.48 in Table VII but −0.49 in Table I, l. 173 and l. 524.
  - **Fix.** Convert to grid steps and caption "in grid steps". Changed cells:

    | Cell | Now | Should read |
    |---|---|---|
    | bf16 mirrored original | −0.48 | −0.49 |
    | 4-bit object | +0.80 | +0.81 |
    | 4-bit mirrored move | +0.77 | +0.78 |
    | bf16 mirrored table side | −0.80 | −0.81 |
    | bf16 mirrored move | +0.15 | +0.16 |
    | paraphrase floor, left/right (App. D, card) | 2.42 | 2.43 |

    All other cells are unchanged at display precision (r2 check). Alternatively, keep 0.000325 and add "(Table~\ref{tab:oat} uses grid steps; values differ by 0.4\%)".
- **V-m3** (App. C l. 329)
  - **Problem.** "mirrored signed and naive-mirror signed (.035 each)". The adjusted p is 0.03447 (`addenda.json: A4_holm_table1`). Ledger §11.4 has the same slip; §11.13 has .034.
  - **Fix.** ".034 each".
- **V-m4** (abstract l. 48)
  - **Problem.** "On identical outputs" governs the placebo clause too, but the grab/take pair is different prompts and outputs.
  - **Fix.** "On identical outputs, naive scoring of mirrored scenes points the word effect toward the named twin (21 of 25 splits) and correct scoring points it away (4 of 25), and reading the wrong axis makes the stimuli seem to fail. A direction-blind contrast scores two instructions that name the same twin as high as ``left'' versus ``right''."
- **V-m5** (§VI l. 250)
  - **Problem.** "only the decisive statistic needs a new category", but Table II items 11 *and* 12 are "[new: evaluation protocol]".
  - **Fix.** "only the evaluation protocol (items 11--12) needs a new category".
- **V-m6** (l. 59; §IV headers l. 195)
  - **Problem.** "scope" is still announced as a tag, but Table I's caption defines only direction and strength, and no row carries scope.
  - **Fix.** Either define it in the caption ("\emph{scope}, the scenes or conditions a claim covers changes") or drop "or scope" at l. 59 and "scope" at l. 195.
- **V-m7** (Fig. 2 caption, l. 201)
  - **Problem.** "Wilson 95% CIs" are drawn on y only. "A direction-blind contrast reads only x" is wrong: it reads x and the same-side agreement, never y. The diamonds (the GPU pair, two points for one row) are unexplained.
  - **Fix.** "Wilson 95\% CIs on $y$ … (diamonds: the \rowid{gpu} pair) … A direction-blind contrast never reads $y$."
- **V-m8** (Table I caption l. 139)
  - **Problem.** "1 bin ≈ 0.32 mm" drops the hedge used at l. 126.
  - **Fix.** "1~bin $\approx$ 0.32\,mm if the units are metres".
- **V-m9** (`figures/*.pdf`)
  - **Problem.** Both figure PDFs carry a Matplotlib Creator and a CreationDate with a +01:00 offset. `main.pdf` does not inherit these, but the source upload would.
  - **Fix.** `fig.savefig(..., metadata={"Creator": None, "Producer": None, "CreationDate": None})` in both scripts.
- **V-m10** (carry-overs from R2, still open)
  - R-m2: l. 61 "Measurement choices already decide LLM evaluations~[… dutta2024accuracy]". Dutta et al. show flips at equal accuracy; App. G l. 582 already has the right sentence.
  - l. 63: no anonymous URL yet.
  - l. 272: OWLv2 is still a `\note`.
  - "term-free" (l. 132, 177) is defined only in App. A. → "term-free (no lateral word)" at its first use.
