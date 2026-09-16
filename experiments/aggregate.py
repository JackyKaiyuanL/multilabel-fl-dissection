"""Aggregate the comprehensive bake-off (experiments/full/) into the paper's
LaTeX results table + key figures. FedPrior is the pLasso method (
blend); FedPrior-NT is the studied not-true variant.
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full")
PAPER = os.path.join(HERE, "..", "paper")
FIG = os.path.join(PAPER, "figures")
os.makedirs(FIG, exist_ok=True)

# (file tag, short column header, metric)
# single-label and multi-label split into two tables in the paper (too wide otherwise)
COLS_SL = [
    ("cifar_a0.1", "C10-.1", "acc"), ("cifar_a0.5", "C10-.5", "acc"),
    ("cifar100_a0.1", "C100-.1", "acc"), ("cifar100_a0.5", "C100-.5", "acc"),
    ("synthetic", "Syn", "acc"),
]
COLS_ML = [
    ("voc_a0.1", "VOC-.1", "mAP"), ("voc_a0.5", "VOC-.5", "mAP"),
    ("coco_a0.1", "COCO-.1", "mAP"), ("coco_a0.5", "COCO-.5", "mAP"),
]
COLS = COLS_SL + COLS_ML
# display name -> json key
ROWS = [("FedAvg", "FedAvg"), ("FedProx", "FedProx"), ("SCAFFOLD", "SCAFFOLD"),
        ("FedNTD", "FedNTD"), ("FedSAM", "FedSAM"),
        ("FedAwS", "FedAwS"), ("FedMLP", "FedMLP"),
        ("FedPrior", "FedPrior")]
COLORS = {"FedAvg": "#1f77b4", "FedProx": "#2ca02c", "SCAFFOLD": "#9467bd",
          "FedNTD": "#ff7f0e", "FedSAM": "#8c564b", "FedAwS": "#bcbd22",
          "FedMLP": "#17becf", "FedPrior-NT": "#7f7f7f", "FedPrior": "#d62728"}


def load(tag):
    p = os.path.join(FULL, f"results_{tag}.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    if "FedPrior-pL" in d and "FedPrior" not in d:   # collapse -pL -> FedPrior
        d["FedPrior"] = d.pop("FedPrior-pL")
    return d


def _finals(hists, k=10):
    """Per-seed metric = mean over the last k rounds (robust to oscillation)."""
    return [statistics.mean(h["test_acc"][-k:]) * 100 for h in hists]


# FedProx on the deep CIFAR tasks uses the FIXED-mu re-tuned runs (the adaptive-mu
# runs in results_*.json ran mu away to ~29 and diverged); see fedprox_grid.py.
_RETUNE_PATH = os.path.join(FULL, "results_fedprox_retune.json")
RETUNE = json.load(open(_RETUNE_PATH)) if os.path.exists(_RETUNE_PATH) else {}


def _seed_vals(disp, key, tag, data):
    if disp == "FedProx" and isinstance(RETUNE.get(tag), list):
        return RETUNE[tag]                       # re-tuned per-seed accuracies
    return _finals(data[tag][key])


def _row_cells(disp, key, cols, data, best):
    cells = []
    for tag, _, _ in cols:
        if key not in data.get(tag, {}):
            cells.append("--"); continue
        fs = _seed_vals(disp, key, tag, data); m = statistics.mean(fs)
        sd = statistics.pstdev(fs) if len(fs) > 1 else 0.0
        txt = f"{m:.1f}" + (f"\\tiny$\\pm${sd:.1f}" if len(fs) > 1 else "")
        if best is not None and disp == best[tag]:      # control rows pass best=None
            txt = r"\textbf{" + txt + "}"
        cells.append(txt)
    return cells


def _one_table(cols, caption, label, control_rows=None):
    control_rows = control_rows or []
    data = {tag: load(tag) for tag, _, _ in cols}
    # per-column best is computed over the proposed/baseline ROWS only, not controls
    best = {}
    for tag, _, _ in cols:
        vals = {disp: statistics.mean(_seed_vals(disp, key, tag, data))
                for disp, key in ROWS if key in data.get(tag, {})}
        best[tag] = max(vals, key=vals.get) if vals else None
    head = "Method & " + " & ".join(h for _, h, _ in cols) + r" \\"
    lines = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{5pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head,
             r"\midrule"]
    for disp, key in ROWS:
        if all(key not in data.get(tag, {}) for tag, _, _ in cols):
            continue
        cells = _row_cells(disp, key, cols, data, best)
        name = r"\textbf{FedPrior}" if disp == "FedPrior" else disp
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    for disp, key in control_rows:                       # flagged below a midrule, never bold
        if all(key not in data.get(tag, {}) for tag, _, _ in cols):
            continue
        cells = _row_cells(disp, key, cols, data, None)
        lines.append(r"\midrule")
        lines.append(disp + r"$^\dagger$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def final_table():
    sl = _one_table(COLS_SL,
        r"Single-label final test accuracy (\%), mean$\pm$std over 3 seeds. "
        r"C10/C100 = CIFAR-10/100 at Dirichlet $\alpha$; Syn = synthetic. "
        r"FedProx uses a fixed $\mu{=}0.1$ (grid $\{0.01,0.1,1.0\}$). "
        r"Best method per column in \textbf{bold}.", "tab:sl")
    ml = _one_table(COLS_ML,
        r"Multi-label mAP (\%), mean$\pm$std over 3 seeds, on VOC and COCO at "
        r"Dirichlet $\alpha$. FedAwS/FedMLP are dedicated multi-label FL baselines; "
        r"best method per column in \textbf{bold}. "
        r"$^\dagger$FedNTD-BCE is a causal control (not a proposed method): it "
        r"restores BCE on the negatives, which reverses the not-true collapse "
        r"(Sec.~\ref{sec:why}).",
        "tab:ml", control_rows=[("FedNTD-BCE", "FedNTD-BCE")])
    open(os.path.join(PAPER, "results_table_input.tex"), "w").write(sl + "\n\n" + ml + "\n")
    print(sl); print(); print(ml)


def curve(tag, title, fname, metric_label):
    per = load(tag)
    plt.figure(figsize=(6.2, 4.2))
    for disp, key in ROWS:
        if key not in per:
            continue
        hists = per[key]
        L = min(len(h["test_acc"]) for h in hists)
        h = [statistics.mean(hh["test_acc"][i] for hh in hists) for i in range(L)]
        plt.plot([v * 100 for v in h], label=disp, color=COLORS[disp],
                 linewidth=2.2 if disp == "FedPrior" else 1.4,
                 linestyle="-" if "FedPrior" not in disp or disp == "FedPrior" else "--")
    plt.xlabel("Communication round"); plt.ylabel(metric_label)
    plt.title(title); plt.legend(fontsize=7, loc="best"); plt.grid(alpha=0.3)
    out = os.path.join(FIG, fname)
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print("saved", out)


if __name__ == "__main__":
    final_table()
    curve("cifar_a0.1", "CIFAR-10, Dir(0.1)", "full_cifar_a01.png", "Test accuracy (%)")
    curve("cifar100_a0.1", "CIFAR-100, Dir(0.1)", "full_cifar100_a01.png", "Test accuracy (%)")
    curve("voc_a0.1", "VOC multi-label, Dir(0.1)", "full_voc_a01.png", "mAP (%)")
    curve("coco_a0.1", "COCO multi-label, Dir(0.1)", "full_coco_a01.png", "mAP (%)")
