"""Reviewer-requested VOC controls (cheap, multi-label):

M4 -- not-true-with-BCE control: does restoring the BCE downward pressure on the
      negatives prevent the mAP collapse? (FedNTD-BCE vs the collapsing FedNTD.)
M2 -- L1 ablation: is the prior-LASSO L1 term (lambda=1e-5) doing anything?
      Compare FedPrior (l1=1e-5) vs l1=0 in mAP and in induced weight sparsity.

Saves experiments/full/results_review_voc.json. ~30 min on one GPU.
    python -m experiments.review_voc
"""
import json
import os
import statistics
from dataclasses import replace

import torch

from fedprior.algorithms import TrainConfig, MethodSpec, METHODS, federated
from fedprior.models import build_model
from fedprior import data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "results_review_voc.json")

ROUNDS = 100
SEEDS = (0, 1, 2)


def _cfg(seed):
    return TrainConfig(model="resnet_ml", lr=0.01, momentum=0.9, sam_rho=0.02,
                       augment=True, lr_schedule="cosine", batchsize=32,
                       local_steps=5, select_clients=8, device="cuda", seed=seed,
                       eval_batch=128, multilabel=True)


@torch.no_grad()
def _margin(state, cfg, tc):
    dev = torch.device(cfg.device)
    net = build_model(cfg.model, **cfg.model_kw).to(dev); net.load_state_dict(state); net.eval()
    P, Y = [], []
    for i in range(0, tc.number, cfg.eval_batch):
        xb = tc.x[i:i + cfg.eval_batch].to(dev).float()
        P.append(torch.sigmoid(net(xb)).cpu()); Y.append(tc.y[i:i + cfg.eval_batch].cpu())
    P = torch.nan_to_num(torch.cat(P), nan=0.0); Y = torch.cat(Y)
    pos = P[Y == 1]; neg = P[Y == 0]
    return (pos.mean().item() if pos.numel() else 0.0) - (neg.mean().item() if neg.numel() else 0.0)


def _sparsity(state, thr=1e-3):
    """Fraction of weight-matrix entries with |w| < thr (induced sparsity)."""
    tot = zero = 0
    for k, v in state.items():
        if v.dim() > 1:
            tot += v.numel(); zero += (v.abs() < thr).sum().item()
    return zero / max(tot, 1)


def run(spec, label, want_margin=False, want_sparsity=False):
    maps, margins, sparss = [], [], []
    for seed in SEEDS:
        cfg = _cfg(seed)
        cl, gc, tc = data.build_voc_clients(40, 0.1, seed)
        state, h = federated(spec, ROUNDS, cl, gc, tc, cfg)
        maps.append(statistics.mean(h["test_acc"][-10:]) * 100)
        if want_margin and seed == 0:
            margins.append(_margin(state, cfg, tc))
        if want_sparsity:
            sparss.append(_sparsity(state))
        print(f"[{label} s{seed}] mAP={maps[-1]:.2f}", flush=True)
    rec = {"mAP": maps, "mAP_mean": statistics.mean(maps),
           "mAP_std": statistics.pstdev(maps)}
    if margins:
        rec["margin_seed0"] = margins[0]
    if sparss:
        rec["sparsity_mean"] = statistics.mean(sparss)
    return rec


def main():
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}

    # M4: not-true-with-BCE control (vs the standard collapsing FedNTD reference)
    out["FedNTD-BCE"] = run(METHODS["FedNTD-BCE"], "FedNTD-BCE", want_margin=True)
    json.dump(out, open(PATH, "w"))
    out["FedNTD"] = run(METHODS["FedNTD"], "FedNTD", want_margin=True)
    json.dump(out, open(PATH, "w"))

    # M2: L1 ablation (mAP + induced sparsity)
    fp = METHODS["FedPrior"]
    out["FedPrior_l1_1e-5"] = run(fp, "FedPrior_l1_1e-5", want_sparsity=True)
    json.dump(out, open(PATH, "w"))
    out["FedPrior_l1_0"] = run(replace(fp, name="FedPrior-noL1", l1=0.0),
                               "FedPrior_l1_0", want_sparsity=True)
    json.dump(out, open(PATH, "w"))

    print("\n=== REVIEW VOC SUMMARY ===")
    for k, v in out.items():
        extra = ""
        if "margin_seed0" in v: extra += f" margin={v['margin_seed0']:.3f}"
        if "sparsity_mean" in v: extra += f" sparsity={v['sparsity_mean']:.3f}"
        print(f"  {k:18s} mAP={v['mAP_mean']:.1f}+/-{v['mAP_std']:.1f}{extra}")
    print("REVIEW_VOC_DONE", flush=True)


if __name__ == "__main__":
    main()
