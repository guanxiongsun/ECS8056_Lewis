"""Fig. 2 of the BtP draft: split share x direction share, one point per Table I row.

Every number comes from the evidence ledger (claims_ledger.md section 11.5):
    evidence/addenda.json -> A5_decomposition.rows  (x, y and their Wilson 95% intervals)
and is cross-checked against evidence/one_at_a_time.csv (opp_n, splits, splits_right_way).

    x = share of decided opposite scenes in which the two instructions move apart (splits)
    y = share of those splits in which each instruction moves toward its own twin
    x * y = opposite both-correct (selection), drawn as dotted iso-curves.

Run:  /Users/s3057498/code/ECS8056_Lewis/.venv/bin/python docs/btp_paper/figures/make_fig_split_direction.py
Writes figures/fig_split_direction.pdf next to this script.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EVID = HERE.parent / "evidence"

PIT = "#D55E00"   # pitfalls (vermillion), as in Fig. 1
FORK = "#0072B2"  # forks (blue)
REF = "black"

# row -> (kind, filled?, label, label offset in points (dx, dy), horizontal alignment)
# filled = the row reads the reference run's own model outputs; open = different outputs (weights, inputs, GPU).
ROWS = {
    "ref":          ("ref",  True,  "reference",                       (6, -3),  "left"),
    "sign":         ("pit",  True,  "sign flipped",                    (-6, 0),  "right"),
    "mir_naive":    ("pit",  False, "mirrored, naive scoring",         (-6, 0),  "right"),
    "argmax":       ("fork", True,  "argmax",                          (-6, 0),  "right"),
    "thr0":         ("fork", True,  "any nonzero",                     (6, 0),   "left"),
    "thr2":         ("fork", True,  "2 bins",                          (6, 3),   "left"),
    "nf4":          ("fork", False, "4-bit",                           (-7, 2),  "right"),
    "gpu":          ("fork", False, "4-bit, A100",                     (-6, -1), "right"),
    "gpu_cmp":      ("fork", False, None,                              (0, 0),   "left"),
    "mir":          ("fork", False, "mirrored",                        (6, 2),   "left"),
    "w_prenominal": ("fork", False, "\u201cthe left {noun}\u201d",      (0, 8),   "center"),
    "w_object":     ("fork", False, "\u201cthe object\non the left\u201d", (6, 0), "left"),
    "w_table_side": ("fork", False, "\u201c\u2026 left side of the table\u201d", (6, -4), "left"),
    "sub_det":      ("fork", True,  "detected gripper",                (0, -8),  "center"),
}
# Rows not drawn: dx ('toward' is undefined on dx); zero, tok and w254 coincide with ref.
COINCIDE_WITH_REF = ("zero", "tok", "w254")


def load():
    a5 = json.loads((EVID / "addenda.json").read_text())["A5_decomposition"]["rows"]
    with open(EVID / "one_at_a_time.csv") as fh:
        csv_rows = {r["row"]: r for r in csv.DictReader(fh)}
    # Cross-checks against the one-at-a-time table (the paper's Table I/IV source).
    for row in list(ROWS) + list(COINCIDE_WITH_REF):
        d = a5[row]
        assert abs(d["x_split_share"] * d["y_right_way_share"] - d["both_correct"]) < 1e-12, row
        if row == "gpu_cmp":
            c = csv_rows["gpu"]
            n, s, k = int(c["cmp_opp_n"]), int(c["cmp_splits"]), int(c["cmp_splits_right_way"])
        else:
            c = csv_rows[row]
            n, s, k = int(c["opp_n"]), int(c["splits"]), int(c["splits_right_way"])
        assert (n, s, k) == (d["n"], d["splits"], d["right_way"]), (row, (n, s, k), d)
    for row in COINCIDE_WITH_REF:
        assert (a5[row]["n"], a5[row]["splits"], a5[row]["right_way"]) == (
            a5["ref"]["n"], a5["ref"]["splits"], a5["ref"]["right_way"]), row
    return a5


def main():
    a5 = load()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    xmin, xmax = 0.13, 0.315

    # Iso-curves of opposite both-correct = x * y.
    xs = np.linspace(xmin, xmax, 400)
    for c in (0.05, 0.10, 0.20):
        ys = c / xs
        keep = ys <= 1.0
        ax.plot(100 * xs[keep], 100 * ys[keep], ls=":", lw=0.7, color="0.55", zorder=1)
        lab_x = xmax - 0.004
        ax.text(100 * lab_x, 100 * c / lab_x - 1.5, f"{int(c * 100)}%", fontsize=5.8, color="0.4",
                ha="right", va="top", zorder=1)
    ax.axhline(50, ls="--", lw=0.6, color="0.6", zorder=1)

    pos = {}
    for row, (kind, filled, label, off, ha) in ROWS.items():
        d = a5[row]
        x, y = 100 * d["x_split_share"], 100 * d["y_right_way_share"]
        lo, hi = (100 * v for v in d["y_ci"])
        col = {"ref": REF, "pit": PIT, "fork": FORK}[kind]
        marker = "o" if kind != "ref" else "s"
        if row.startswith("gpu"):
            marker = "D"
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col, elinewidth=0.6,
                    alpha=0.45, capsize=0, zorder=2)
        ax.plot(x, y, marker=marker, ms=4.2 if marker != "D" else 3.6, mec=col,
                mfc=col if filled else "white", mew=0.9, ls="none", zorder=3)
        pos[row] = (x, y)
        if label:
            ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=5.8,
                        color=col, ha=ha, va="center", zorder=4, linespacing=0.95,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))

    def arrow(a, b, text, tx, ty):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.annotate("", xy=(x1, y1 - (2.2 if y1 > y0 else -2.2)), xytext=(x0, y0 + (2.2 if y1 > y0 else -2.2)),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.25", shrinkA=0, shrinkB=0,
                                    mutation_scale=6), zorder=2)
        ax.text(tx, ty, text, fontsize=5.6, color="0.25", ha="center", va="center", style="italic")

    arrow("ref", "sign", "sign only", pos["ref"][0] - 1.5, 57)
    arrow("mir", "mir_naive", "scoring only", pos["mir"][0] + 2.3, 28)

    ax.set_xlim(100 * xmin, 100 * xmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Decided opposite scenes in which the instructions split (%)")
    ax.set_ylabel("Splits toward the named twin (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend: colour = kind, fill = same model outputs as the reference.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mec=PIT, mfc=PIT, ms=4, label="pitfall"),
        Line2D([], [], marker="o", ls="none", mec=FORK, mfc=FORK, ms=4, label="fork"),
        Line2D([], [], marker="o", ls="none", mec="0.3", mfc="white", ms=4, label="different outputs"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.8, frameon=False, handletextpad=0.2,
              borderaxespad=0.2, labelspacing=0.25, ncol=3, columnspacing=0.8)
    fig.tight_layout(pad=0.25)
    out = HERE / "fig_split_direction.pdf"
    fig.savefig(out, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(HERE / "fig_split_direction.png", dpi=220, metadata={"Software": None})
    print("wrote", out)


if __name__ == "__main__":
    main()
