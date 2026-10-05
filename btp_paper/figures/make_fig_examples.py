"""Appendix figure: example stimuli, chosen by a fixed rule (no hand-picking).

Rule
- Analysed scenes = the 400 frozen scenes (constructed/evaluation_set.csv) whose hand label
  (human_configuration) is opposite, same_side_left or same_side_right (340 scenes).
- For each layout, take the first such scene in construct_id order whose gripper was detected.
- Mirrored example: the horizontal flip of the opposite example (as the mirrored runs do).
- Rejected example: the first composite in construct_id order rejected in blind screening
  for an implausible paste (constructed/constructed_review.csv).

Run:  /Users/s3057498/code/ECS8056_Lewis/.venv/bin/python docs/btp_paper/figures/make_fig_examples.py
Writes figures/fig_examples.pdf (and .png) next to this script.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "google_drive" / "v2" / "constructed"


def main():
    man = {r["construct_id"]: r for r in csv.DictReader(open(DATA / "constructed_manifest.csv"))}
    frozen = sorted(r["construct_id"] for r in csv.DictReader(open(DATA / "evaluation_set.csv")))
    review = sorted(csv.DictReader(open(DATA / "constructed_review.csv")), key=lambda r: r["construct_id"])

    picks = []
    for layout, label in (("opposite", "opposite"), ("same_side_left", "both-left"),
                          ("same_side_right", "both-right")):
        cid = next(c for c in frozen
                   if man[c]["human_configuration"] == layout and man[c]["gripper_source"] == "detected")
        picks.append((cid, label, False))
    picks.append((picks[0][0], "opposite, mirrored", True))
    rej = next(r["construct_id"] for r in review
               if r["decision"] == "rejected" and r["reject_reason"] == "paste_implausible")
    picks.append((rej, "rejected: implausible paste", False))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7})
    fig, axes = plt.subplots(1, len(picks), figsize=(7.1, 1.25))
    for ax, (cid, label, flip) in zip(axes, picks):
        img = mpimg.imread(DATA / man[cid]["image_path"])
        if flip:
            img = img[:, ::-1]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xlabel(f"“{man[cid]['instr_a']}”", fontsize=5.5, labelpad=1.5)
        print(cid, label, man[cid]["image_width"], man[cid]["image_height"], man[cid]["instr_a"])
    fig.tight_layout(pad=0.2, w_pad=0.3)
    fig.savefig(HERE / "fig_examples.pdf", dpi=200, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(HERE / "fig_examples.png", dpi=160, metadata={"Software": None})


if __name__ == "__main__":
    main()
