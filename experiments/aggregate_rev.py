"""Aggregate the AGCS camera-ready revision runs into LaTeX tables + a numbers summary.

Reads  experiments/full/results_pt_*.json (original 3-seed pretrained results),
       experiments/full/rev/*.json           (new groups),
       experiments/full/results_b2/*.json    (FedMLP-bal orthogonality runs).
Writes experiments/tables/tables_rev.tex, numbers_summary.json, figures/diag_margin_voc.png
    python -m experiments.aggregate_rev
"""
import glob, json, os, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full"); REV = os.path.join(FULL, "rev")
B2 = os.path.join(FULL, "results_b2")
CR = os.path.join(HERE, "tables")          # output dir for generated tables / figure
CELLS = [("VOC-.1", "voc", 0.1), ("VOC-.5", "voc", 0.5), ("COCO-.1", "coco", 0.1), ("COCO-.5", "coco", 0.5)]
LRS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


def last10(h):
    return statistics.mean(h["test_acc"][-10:]) * 100


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def orig(task, alpha, method):
    """Original pretrained runs (3 seeds), plus FedMLP-bal from results_b2."""
    d = load(os.path.join(FULL, f"results_pt_{task}_a{alpha}.json"))
    if method in d:
        return [last10(h) for h in d[method]]
    d = load(os.path.join(B2, f"results_pt_{task}_a{alpha}.json"))
    return [last10(h) for h in d.get(method, [])]


def rev(group, task, alpha, method, lr=0.003):
    d = load(os.path.join(REV, f"{group}_{task}_a{alpha}.json"))
    return [last10(h) for h in d.get(f"{method}@lr{lr}", [])]


def ms(vals):
    if not vals:
        return None
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def cell(vals, bold=False, n_note=False):
    r = ms(vals)
    if r is None:
        return "--"
    m, sd, n = r
    t = f"{m:.1f}\\tiny$\\pm${sd:.1f}"
    if n_note and n != 3:
        t += f"\\tiny(n={n})"
    return r"\textbf{" + t + "}" if bold else t


MEAN_VARIANT = {"FocalBCE": "FocalBCE-m", "ASL": "ASL-m", "DBLoss": "DBLoss-m"}


def rev_any(groups, task, alpha, method, lr):
    out = []
    for g in groups:
        out += rev(g, task, alpha, method, lr)
    return out


def best_lr(group_default, group_ref, task, alpha, method):
    """Best configuration (by 3-seed mean) over the LR grid and, for focal/ASL/DB, over
    the two reductions (class-sum reference code 's' vs. entry-mean 'm'). Returns (vals, lr, tag)."""
    cands = []
    for lr in LRS:
        if method in ("FedAvg", "BalBCE"):
            vals = orig(task, alpha, method) if lr == 0.003 else rev_any(["losses_ref", "losses_hi", "losses_ref_x"], task, alpha, method, lr)
            if vals:
                cands.append((statistics.mean(vals), lr, "", vals))
        else:
            vals = rev_any(["losses", "losses_lo", "losses_hi"], task, alpha, method, lr)
            if vals:
                cands.append((statistics.mean(vals), lr, "s" if method in MEAN_VARIANT else "", vals))
            if method in MEAN_VARIANT:
                vals = rev_any(["losses_m", "losses_m_hi"], task, alpha, MEAN_VARIANT[method], lr)
                if vals:
                    cands.append((statistics.mean(vals), lr, "m", vals))
    if not cands:
        return [], None, ""
    _, lr, tag, vals = max(cands)
    return vals, lr, tag


out = []
summary = {}

# ---------------- Table: weighting axis widened ----------------
rows = [("FedAvg (std BCE)", "FedAvg"), ("WBCE (pos-weight)", "WBCE"), ("Focal ($\\gamma{=}2$)", "FocalBCE"),
        ("ASL", "ASL"), ("DB-loss", "DBLoss"), ("\\textbf{BalBCE}", "BalBCE")]
out.append(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Weighting axis widened} (pretrained ResNet-18, mAP\,\%, 3 seeds). Established imbalance-aware objectives vs.\ the two BCE references, every row at its \emph{best} configuration over a six-point base-LR grid from $3{\cdot}10^{-4}$ to $10^{-1}$; the bracket gives that best LR (e.g.\ [3e-2] $=0.03$). Focal, ASL, and DB-loss are additionally reported at the better of their two loss reductions (class-sum as in the reference code, or entry-mean; Sec.~\ref{sec:variants}), so that no competitor loses to a loss-scale artefact. Tuning the LR lifts plain FedAvg substantially, yet explicit positive/negative re-weighting (WBCE, BalBCE) stays far ahead on COCO, whereas the focusing-based losses (focal, ASL) and DB-loss, with their reference hyperparameters, do not beat a tuned FedAvg under our federated SGD protocol. Lower block: adding the FedND negative-distillation term ($+$ND, LR $3{\cdot}10^{-3}$) changes each base loss by at most $1.4$ mAP.}
\label{tab:losses}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Loss & VOC-.1 & VOC-.5 & COCO-.1 & COCO-.5 \\
\midrule""")
best_by_cell = {}
for cn, task, alpha in CELLS:
    vals = {}
    for disp, m in rows:
        v, lr, tag = best_lr("losses", "losses_ref", task, alpha, m)
        if v:
            vals[m] = statistics.mean(v)
    best_by_cell[cn] = max(vals, key=vals.get) if vals else None
for disp, m in rows:
    cells = []
    for cn, task, alpha in CELLS:
        v, lr, tag = best_lr("losses", "losses_ref", task, alpha, m)
        c = cell(v, bold=(best_by_cell[cn] == m))
        if lr is not None and c != "--":
            c += r"\,{\tiny[" + f"{lr:.0e}".replace("e-0", "e-") + "]}"
        cells.append(c)
        summary[f"loss/{m}/{cn}"] = (ms(v), lr, tag)
    out.append(disp + " & " + " & ".join(cells) + r" \\")
out.append(r"\midrule")
for disp, base, nd, src in [("Focal $\\to$ +ND", "FocalBCE", "Focal+ND", "rev"),
                            ("ASL $\\to$ +ND", "ASL", "ASL+ND", "rev"),
                            ("BalBCE $\\to$ +ND (FedND)", "BalBCE", "FedNTD-BCE", "orig")]:
    cells = []
    for cn, task, alpha in CELLS:
        if src == "rev":
            pairs = [(rev("losses", task, alpha, base, 0.003), rev("distill", task, alpha, nd, 0.003)),
                     (rev("losses_m", task, alpha, base + "-m", 0.003),
                      rev("distill_m", task, alpha, base.replace("BCE", "") + "-m+ND", 0.003))]
            pairs = [(b, d) for b, d in pairs if b and d]
            b, d = max(pairs, key=lambda bd: statistics.mean(bd[0])) if pairs else ([], [])
        else:
            b = orig(task, alpha, base); d = orig(task, alpha, nd)
        if b and d:
            cells.append(f"{statistics.mean(b):.1f}$\\to${statistics.mean(d):.1f}")
            summary[f"nd/{base}/{cn}"] = (statistics.mean(b), statistics.mean(d))
        else:
            cells.append("--")
    out.append(disp + " & " + " & ".join(cells) + r" \\")
out.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- Table: distillation axis widened ----------------
out.append(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{\textbf{Distillation axis widened} (pretrained ResNet-18, mAP\,\%, 3 seeds). Five output-space or parameter-space terms added \emph{on top of} balanced BCE. Every term stays within $1.3$ mAP of BalBCE (and at most $0.7$ above it) except FedMLP's pseudo-labelling on VOC; the parameter-space proximal term is likewise inert. The marginal value of the regulariser, once the loss is balanced, is therefore small across formulations, not only for the negative-KL term.}
\label{tab:distill}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
On top of BalBCE & VOC-.1 & VOC-.5 & COCO-.1 & COCO-.5 \\
\midrule""")
drows = [("BalBCE (none)", "BalBCE", "orig"),
         ("+ neg.\\ KL (FedND)", "FedNTD-BCE", "orig"),
         ("+ all-class KL", "BalBCE+KD", "distill"),
         ("+ response blend ($\\eta{=}0.1$)", "FedPrior-bal", "distill"),
         ("+ FedMLP consist.\\,+\\,pseudo-lab.", "FedMLP-bal", "orig"),
         ("+ FedProx prox.\\ ($\\mu{=}0.1$)", "FedProx-bal", "distill")]
for disp, m, src in drows:
    cells = []
    for cn, task, alpha in CELLS:
        v = orig(task, alpha, m) if src == "orig" else rev("distill", task, alpha, m)
        cells.append(cell(v)); summary[f"distill/{m}/{cn}"] = ms(v)
    out.append(disp + " & " + " & ".join(cells) + r" \\")
out.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- Table: robustness (backbone / resolution / FLAIR) ----------------
KEY = [("FedAvg", "FedAvg"), ("FedNTD (masking)", "FedNTD"), ("FedMLP", "FedMLP"),
       ("FedPrior (blend)", "FedPrior"), ("\\textbf{BalBCE}", "BalBCE"), ("FedND (bal+distill)", "FedNTD-BCE")]
cols = [("ConvNeXt-T", "cnx", "voc", 0.1, "VOC-.1"), ("ConvNeXt-T", "cnx", "coco", 0.1, "COCO-.1"),
        ("R18@224", "res224", "voc", 0.1, "VOC-.1"), ("R18@224", "res224", "coco", 0.1, "COCO-.1"),
        ("FLAIR", "flair", "flair", None, "real users")]
cnx_lr = 0.003
lr_path = os.path.join(REV, "cnx_lr_voc_a0.1.json")
if os.path.exists(lr_path):
    d = load(lr_path)
    cand = [(statistics.mean(last10(h) for h in v), float(k.split("@lr")[1])) for k, v in d.items() if k.startswith("FedAvg@")]
    if cand:
        cnx_lr = max(cand)[1]
summary["cnx_lr"] = cnx_lr
out.append(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Robustness of the ordering} (mAP\,\%, 3 seeds, Dir$(0.1)$ where applicable): a second backbone family (ImageNet ConvNeXt-Tiny, LayerNorm, all parameters trained), higher resolution ($224{\times}224$, pretrained ResNet-18), and \textbf{FLAIR} with its \emph{real} per-user partition (100 Flickr users, 10 per round, 17 coarse labels, $128{\times}128$). The ordering of Table~\ref{tab:ml_pt} is unchanged in every column.}
\label{tab:robust}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{ConvNeXt-T} & \multicolumn{2}{c}{ResNet-18 @224} & FLAIR \\
Method & VOC-.1 & COCO-.1 & VOC-.1 & COCO-.1 & 17-cls \\
\midrule""")
best = {}
for ci, (_, g, task, alpha, _) in enumerate(cols):
    vals = {}
    for disp, m in KEY + [("FLAG", "FLAG"), ("ASL", "ASL"), ("Focal", "FocalBCE"), ("FedND-std", "FedND-std")]:
        v = rev(g, task, alpha, m, cnx_lr if g == "cnx" else 0.003)
        if v:
            vals[m] = statistics.mean(v)
    best[ci] = max(vals, key=vals.get) if vals else None
for disp, m in KEY + [("FLAG", "FLAG"), ("Focal", "FocalBCE"), ("ASL", "ASL")]:
    cells = []
    for ci, (_, g, task, alpha, _) in enumerate(cols):
        v = rev(g, task, alpha, m, cnx_lr if g == "cnx" else 0.003)
        cells.append(cell(v, bold=(best[ci] == m), n_note=True)); summary[f"robust/{g}/{task}/{m}"] = ms(v)
    if all(c == "--" for c in cells):
        continue
    out.append(disp + " & " + " & ".join(cells) + r" \\")
out.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- Table: 2x2 with 5 seeds (pretrained) ----------------
def merged(task, alpha, m):
    return orig(task, alpha, m) + rev("seeds", task, alpha, m)
fs = {"voc": load(os.path.join(FULL, "results_voc_a0.1.json")), "coco": load(os.path.join(FULL, "results_coco_a0.1.json"))}
def fsv(task, m):
    return [last10(h) for h in fs[task].get(m, [])]
out.append(r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Disentangling the multi-label gain: class-balanced BCE vs.\ negative distillation (mAP\,\%, Dir$(0.1)$; cells are VOC/COCO, mean$\pm$std). Pretrained rows use \textbf{five} seeds, from-scratch rows three. Balancing is the robust lever; negative distillation is situational (it helps only from-scratch VOC). On from-scratch VOC the two $+$distill cells coincide ($20.68$ and $20.73$): balancing is inert there, so it does not matter whether the restored BCE is standard or balanced.}
\label{tab:2x2}
\begin{tabular}{llcc}
\toprule
Backbone & BCE & no distill & +neg distill \\
\midrule""")
def pair(a, b):
    ra, rb = ms(a), ms(b)
    if ra is None or rb is None:
        return "--"
    return f"{ra[0]:.1f}\\tiny$\\pm${ra[1]:.1f}/{rb[0]:.1f}\\tiny$\\pm${rb[1]:.1f}"
out.append("From-scratch & std & " + pair(fsv("voc", "FedAvg"), fsv("coco", "FedAvg")) + " & " + pair(fsv("voc", "FedND-std"), fsv("coco", "FedND-std")) + r" \\")
out.append(" & balanced & " + pair(fsv("voc", "BalBCE"), fsv("coco", "BalBCE")) + " & " + pair(fsv("voc", "FedNTD-BCE"), fsv("coco", "FedNTD-BCE")) + r" \\")
out.append(r"\midrule")
out.append("Pretrained & std & " + pair(merged("voc", 0.1, "FedAvg"), merged("coco", 0.1, "FedAvg")) + " & " + pair(merged("voc", 0.1, "FedND-std"), merged("coco", 0.1, "FedND-std")) + r" \\")
out.append(" & balanced & " + pair(merged("voc", 0.1, "BalBCE"), merged("coco", 0.1, "BalBCE")) + " & " + pair(merged("voc", 0.1, "FedNTD-BCE"), merged("coco", 0.1, "FedNTD-BCE")) + r" \\")
out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
for m in ["FedAvg", "FedND-std", "BalBCE", "FedNTD-BCE"]:
    for task in ["voc", "coco"]:
        for alpha in [0.1, 0.5]:
            summary[f"2x2_5seed/{m}/{task}{alpha}"] = ms(merged(task, alpha, m))

os.makedirs(CR, exist_ok=True)
open(os.path.join(CR, "tables_rev.tex"), "w").write("\n".join(out))

# ---------------- Fig. 1 over three seeds ----------------
mp = os.path.join(REV, "margin_voc_a0.1.json")
if os.path.exists(mp):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = load(mp)
    style = {"FedAvg": ("#1f77b4", "-", "FedAvg (std BCE)"), "FedNTD": ("#ff7f0e", "-", "FedNTD (masking)"),
             "FedND-std": ("#9467bd", "-.", "FedND-std (neg. BCE, std)"),
             "FedNTD-BCE": ("#2ca02c", "--", "FedND (neg. BCE, balanced)"),
             "FedPrior": ("#d62728", "-", "FedPrior (blend)")}
    plt.figure(figsize=(4.8, 4.0))
    for m, (c, ls, lab) in style.items():
        hs = d.get(f"{m}@lr0.01", [])
        if not hs:
            continue
        R = np.array(hs[0]["margin"]["rounds"])
        M = np.array([np.array(h["margin"]["pos"]) - np.array(h["margin"]["neg"]) for h in hs])
        maps = [last10(h) for h in hs]
        plt.plot(R, M.mean(0), color=c, ls=ls, lw=2.0,
                 label=f"{lab.replace(chr(92),'')} (mAP {statistics.mean(maps):.1f}$\\pm${statistics.pstdev(maps):.1f}, n={len(hs)})")
        if len(hs) > 1:
            plt.fill_between(R, M.min(0), M.max(0), color=c, alpha=0.15, lw=0)
        summary[f"margin/{m}"] = (ms(maps), float(M.mean(0)[-1]))
    plt.axhline(0, color="gray", lw=0.8, ls=":")
    plt.xlabel("Communication round"); plt.ylabel("Positive$-$negative score margin")
    plt.title("VOC Dir(0.1), from-scratch: 3-seed mean (band = min--max)", fontsize=10)
    plt.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=1.0, handlelength=1.8)
    plt.grid(alpha=0.3)
    os.makedirs(os.path.join(CR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(CR, "figures", "diag_margin_voc.png"), dpi=150, bbox_inches="tight"); plt.close()

json.dump({k: v for k, v in summary.items()}, open(os.path.join(CR, "numbers_summary.json"), "w"), indent=1, default=str)
for k, v in summary.items():
    print(f"{k:45s} {v}")
print("AGGREGATE_REV_DONE")
