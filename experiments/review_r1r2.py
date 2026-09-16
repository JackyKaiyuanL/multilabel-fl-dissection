"""Round-2 review fixes that need numbers:
R1 -- re-run the L1 ablation on the CURRENT bare-blend FedPrior (VOC Dir(0.1)):
      FedPrior (l1=1e-5) vs l1=0, mAP + induced near-zero-weight fraction.
R2 -- run the FedNTD-BCE control on COCO (Dir(0.1)) to show the fix generalizes
      beyond VOC.
Saves experiments/full/results_r1r2.json.
    python -m experiments.review_r1r2
"""
import json, os, statistics
from dataclasses import replace
import torch
from fedprior.algorithms import TrainConfig, METHODS, federated
from fedprior import data

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "full", "results_r1r2.json")
SEEDS = (0, 1, 2)


def _voc_cfg(seed):
    return TrainConfig(model="resnet_ml", lr=0.01, momentum=0.9, sam_rho=0.02,
                       augment=True, lr_schedule="cosine", batchsize=32,
                       local_steps=5, select_clients=8, device="cuda", seed=seed,
                       eval_batch=128, multilabel=True)


def _coco_cfg(seed):
    return TrainConfig(model="resnet_ml", model_kw={"num_classes": 80}, lr=0.01,
                       momentum=0.9, sam_rho=0.02, augment=True, lr_schedule="cosine",
                       batchsize=32, local_steps=5, select_clients=8, device="cuda",
                       seed=seed, eval_batch=128, multilabel=True)


def _sparsity(state, thr=1e-3):
    tot = zero = 0
    for k, v in state.items():
        if v.dim() > 1:
            tot += v.numel(); zero += (v.abs() < thr).sum().item()
    return zero / max(tot, 1)


def run_voc(spec, label, rounds=100):
    maps, sp = [], []
    for seed in SEEDS:
        cfg = _voc_cfg(seed); cl, gc, tc = data.build_voc_clients(40, 0.1, seed)
        state, h = federated(spec, rounds, cl, gc, tc, cfg)
        maps.append(statistics.mean(h["test_acc"][-10:]) * 100); sp.append(_sparsity(state))
        print(f"[{label} s{seed}] mAP={maps[-1]:.2f} sparsity={sp[-1]:.3f}", flush=True)
    return {"mAP_mean": statistics.mean(maps), "mAP_std": statistics.pstdev(maps),
            "sparsity_mean": statistics.mean(sp)}


def run_coco(spec, label, rounds=150):
    maps = []
    for seed in SEEDS:
        cfg = _coco_cfg(seed); cl, gc, tc = data.build_coco_clients(40, 0.1, seed)
        _, h = federated(spec, rounds, cl, gc, tc, cfg)
        maps.append(statistics.mean(h["test_acc"][-10:]) * 100)
        print(f"[{label} s{seed}] mAP={maps[-1]:.2f}", flush=True)
    return {"mAP_mean": statistics.mean(maps), "mAP_std": statistics.pstdev(maps)}


def main():
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    fp = METHODS["FedPrior"]
    assert fp.blend and not fp.per_class and fp.sam_rho == 0.0 and fp.l1 == 1e-5
    # R1: L1 ablation on the bare blend
    out["FedPrior_l1_1e-5_VOC"] = run_voc(fp, "FedPrior_l1_1e-5")
    json.dump(out, open(PATH, "w"))
    out["FedPrior_l1_0_VOC"] = run_voc(replace(fp, name="FedPrior-noL1", l1=0.0), "FedPrior_l1_0")
    json.dump(out, open(PATH, "w"))
    # R2: FedNTD-BCE on COCO
    out["FedNTD-BCE_COCO"] = run_coco(METHODS["FedNTD-BCE"], "FedNTD-BCE_COCO")
    out["FedNTD_COCO"] = run_coco(METHODS["FedNTD"], "FedNTD_COCO")
    json.dump(out, open(PATH, "w"))
    print("\n=== R1/R2 SUMMARY ===")
    for k, v in out.items():
        extra = f" sparsity={v['sparsity_mean']:.3f}" if "sparsity_mean" in v else ""
        print(f"  {k:24s} mAP={v['mAP_mean']:.1f}+/-{v['mAP_std']:.1f}{extra}")
    print("REVIEW_R1R2_DONE", flush=True)


if __name__ == "__main__":
    main()
