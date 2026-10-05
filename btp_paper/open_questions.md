# Open questions after draft v3.2 (3 Oct 2026, afternoon)

Draft v3.2 builds to 21 pages:
- the main text still ends at the foot of page 4, and References start at the top of page 5, so there is **no slack**;
- `main.log` has no overfull boxes and no undefined references, and the anonymity grep is clean;
- 1 `\pending{}` item remains in the main text (B7) and 8 in the appendix;
- every number is traced in `claims_trace.md`; §F and §H there list where v3 and v3.2 depart from the instructions.

Settled earlier (reference run, second annotator, title) is not repeated here.

## 1. Decisions on the departures from the instructions (v3 and v3.2)

- **Prefix tag.** The §IV prefix paragraph is tagged "(action space; strength)", not "direction" as instructed.
  - Holding dx fixed keeps every wording's sign. Table I defines "direction" as a sign change and "strength" as a change in size or significance without one.
  - The original wording's null → away is the mirror image of ref → nf4 (away → null), which is already tagged "strength".
  - Switching to "direction" is a one-word edit, but the Table I caption would then need a different definition.
- **Table II items 4, 6, 7, 8, 10 and 12.**
  - The main-text card keeps generic checks. Item 6 is now "compute ladder: dtype, kernel, GPU, repeat", item 7 "flip swaps the named twin; own pre-processing", item 8 "paraphrase floor; lower-casing" and item 10 "approval rate per condition; score the rejected".
  - This study's results went into the filled-in card (Table XVI, App. F).
  - Short "here: …" results in Table II would cost lines on page 4 that it does not have.
- **"Four- to fivefold".** v3.2 writes this instead of the recommended "about fourfold", following the ledger's own 4–5×; one of the three wordings is 5.35×.
- **Table I `screen` contrasts and split p-values.** They come from `round2.json` because the ledger's B6 table omits them. Add them to the ledger, or drop the cells.

## 2. Pending experiments: what changes when each lands

| Item | Where it is pending | When it lands |
|---|---|---|
| **B1 stage 2** (O4 facets, the A100 facet, prefix facets, NF4 prefix run) | App. E opening paragraph; Table XVI item 12 | Add rows to Table XIV, or a second table, plus one sentence in App. E. The main text needs a clause only if a facet contradicts §V, and by the design's rule the curve then wins. |
| **B6 steps 2–4**: hand check of 100 rejected layouts; reweighting; the **inter-annotator κ** (L and S, 100 frozen scenes, in progress) | App. C screening paragraph; App. E Type U; Table XVI item 10; §VII limitations | The κ goes into Table XVI item 10 and the limitations sentence. Step 2 qualifies Table XI only if agreement is poor. |
| **B5 step 3** (appended empty token) | Table XVI item 8 | One clause in item 8. |
| **B7** (double-coded reporting audit) | §VI; §VII limitations | Replace the §VI pending with the count (same length) and drop it from the limitations. |
| **B9** (median and mode readouts from the logged distributions) | Table IV `dist`; App. E facets | Table IV row; readout facet in stage 2. |

## 3. Main-text space

There is no slack on page 4; v3.2 already used the cuts listed after v3. Next candidates, least costly first:
1. The two examples in §IV's prefix paragraph ("the left {noun}" and table side): about one line.
2. "with threshold and scene subset varied within each" in §V's curve paragraph: about half a line.
3. The optional abstract clause added in v3.2: one abstract line.
4. The §III 4-bit table-side example of the Type S problem: two lines (the O4 sentence makes a similar point).

## 4. Ledger requests (numbers the draft does not print because the ledger lacks them)

- **A100 rows in Table IV.** The `prefix` row and its A100 comparator have no bootstrap CIs, same-sign rates or McNemar pairs, so Table IV points to Table I and Table VIII instead.
- **Fig. 2 and Table V.** The A100 prefix pair is not plotted because the ledger gives no Wilson CIs for it.
- **Table XIV.** It prints "≤ .046" for the four "the left {noun}" facets, as the ledger does. The individual Holm values are in `outputs/btp/spec_curve.json`.
- **Earlier R1/R2 suggestions** that still need ledger entries:
  - a pooled "word across phrasings" estimate (R1 edit 5);
  - a row with a deadband of about 1 mm, i.e. 3 bins (R1 edit 9);
  - bf16 dx image-right shares by paste side (R2 R-m21).

## 5. Release

"Released at an anonymous URL" needs the release snapshot (plan, Wed 7 Oct). It must not point at the public fork or the public data mirror. The release should include `analysis_plan_frozen_2026-10-03.md`, whose hash App. E relies on.
