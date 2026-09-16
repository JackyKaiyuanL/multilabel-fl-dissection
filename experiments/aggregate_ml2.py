"""Tables for the re-centered (negative-class-balancing) papers B/C:
 (1) multi-label mAP on the PRETRAINED backbone (lead/realistic regime),
 (2) multi-label mAP from-scratch,
 (3) the 2x2 dissection (class-balancing x negative-distillation), both backbones.
Reads experiments/full/results_{pt_,}{voc,coco}_a{0.1,0.5}.json. Missing cells -> '--'.
Writes the table block to experiments/tables/tables_generated.tex (Tables I-III of the paper, before light hand-editing).
    python -m experiments.aggregate_ml2
"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__)); FULL = os.path.join(HERE, "full")
PAPER = os.path.join(HERE, "tables")

# display name -> results-file key
ROWS = [("FedAvg", "FedAvg"), ("FedNTD (masking)", "FedNTD"),
        ("FedAwS", "FedAwS"), ("FedMLP", "FedMLP"), ("FLAG", "FLAG"),
        ("FedPrior (blend)", "FedPrior"),
        ("\\textbf{BalBCE}", "BalBCE"),
        ("FedND (bal+distill)", "FedNTD-BCE")]
COLS = [("VOC-.1", "voc_a0.1"), ("VOC-.5", "voc_a0.5"),
        ("COCO-.1", "coco_a0.1"), ("COCO-.5", "coco_a0.5")]


def _mean_std(tag, key, pref=""):
    p = os.path.join(FULL, f"results_{pref}{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if key not in d:
        return None
    fs = [statistics.mean(h["test_acc"][-10:]) * 100 for h in d[key]]
    return statistics.mean(fs), (statistics.pstdev(fs) if len(fs) > 1 else 0.0)


def _ml_table(pref, caption, label):
    # best among "methods" (exclude FedND control-ish? here all are methods; bold col max)
    best = {}
    for cn, tag in COLS:
        vals = {}
        for disp, key in ROWS:
            ms = _mean_std(tag, key, pref)
            if ms: vals[disp] = ms[0]
        best[cn] = max(vals, key=vals.get) if vals else None
    lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
             "Method & " + " & ".join(cn for cn, _ in COLS) + r" \\", r"\midrule"]
    for disp, key in ROWS:
        cells = []
        for cn, tag in COLS:
            ms = _mean_std(tag, key, pref)
            if ms is None:
                cells.append("--"); continue
            m, sd = ms
            txt = f"{m:.1f}\\tiny$\\pm${sd:.1f}"
            if disp == best[cn]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(disp + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _twobytwo():
    out = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
           r"\caption{Disentangling the multi-label gain: class-balanced BCE vs.\ "
           r"negative distillation (mAP, mean over 3 seeds, Dir$(0.1)$). Balancing is "
           r"the robust lever; negative distillation is situational (helps only "
           r"from-scratch VOC).}", r"\label{tab:2x2}",
           r"\begin{tabular}{llcc}", r"\toprule",
           r"Backbone & BCE & no distill & +neg distill \\", r"\midrule"]
    pairs = [("From-scratch", "", [("std", "FedAvg", "FedND-std"),
                                   ("balanced", "BalBCE", "FedNTD-BCE")]),
             ("Pretrained", "pt_", [("std", "FedAvg", "FedND-std"),
                                    ("balanced", "BalBCE", "FedNTD-BCE")])]
    for bk, pref, rows in pairs:
        for i, (bce, k_no, k_di) in enumerate(rows):
            def cell(key):
                a = _mean_std("voc_a0.1", key, pref); b = _mean_std("coco_a0.1", key, pref)
                av = f"{a[0]:.1f}" if a else "--"; bv = f"{b[0]:.1f}" if b else "--"
                return f"{av}/{bv}"
            lab = bk if i == 0 else ""
            out.append(f"{lab} & {bce} & {cell(k_no)} & {cell(k_di)} \\\\")
        out.append(r"\midrule" if bk == "From-scratch" else "")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\\[2pt]{\footnotesize Cells are VOC/COCO mAP\%.}", r"\end{table}"]
    return "\n".join(x for x in out if x)


def main():
    pt = _ml_table("pt_",
        r"\textbf{Multi-label mAP (\%), ImageNet-pretrained backbone} (realistic "
        r"regime), 3 seeds. Simply class-balancing the negatives (\textbf{BalBCE}) "
        r"is the strongest lever and beats the dedicated method FedMLP; masking "
        r"(FedNTD) collapses; blending (FedPrior) tracks FedAvg.", "tab:ml_pt")
    fs = _ml_table("",
        r"Multi-label mAP (\%), from-scratch GroupNorm backbone, 3 seeds.",
        "tab:ml_fs")
    twob = _twobytwo()
    block = pt + "\n\n" + fs + "\n\n" + twob + "\n"
    os.makedirs(PAPER, exist_ok=True)
    open(os.path.join(PAPER, "tables_generated.tex"), "w").write(block)
    print(block)
    print("AGG_ML2_DONE")


if __name__ == "__main__":
    main()
