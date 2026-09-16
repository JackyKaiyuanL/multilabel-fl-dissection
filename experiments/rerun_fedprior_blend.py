"""Re-run FedPrior as the bare response blend (the original pLasso idea: blended
target + L1, eta=0.1, no per-class, no SAM) across all testbeds, overwriting the
"FedPrior" entry in each experiments/full/results_<tag>.json. Other methods are
left untouched. 3 seeds.

    python -m experiments.rerun_fedprior_blend
"""
import json
import os

from fedprior.algorithms import METHODS, federated
from experiments.full_benchmark import build_task

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")

CONFIGS = [("synthetic", None),
           ("cifar", 0.1), ("cifar", 0.5),
           ("cifar100", 0.1), ("cifar100", 0.5),
           ("voc", 0.1), ("voc", 0.5),
           ("coco", 0.1), ("coco", 0.5)]
SEEDS = (0, 1, 2)


def main():
    spec = METHODS["FedPrior"]
    assert spec.blend and not spec.per_class and spec.sam_rho == 0.0, \
        "FedPrior must be the bare blend for this re-run"
    # NOTE: the original (blend + per-class + SAM) FedPrior is preserved under the
    # key "FedPrior-pL" in every results file; this writes the bare blend to the
    # separate key "FedPrior" and never touches "FedPrior-pL" (so we can revert).
    for task, alpha in CONFIGS:
        tag = f"{task}_a{alpha}" if task != "synthetic" else task
        path = os.path.join(OUT, f"results_{tag}.json")
        per = json.load(open(path)) if os.path.exists(path) else {}
        hists = per.get("FedPrior", [])              # resume: keep finished seeds
        if len(hists) >= len(SEEDS):
            print(f"[{tag}] FedPrior blend already done ({len(hists)} seeds), skip",
                  flush=True)
            continue
        for si, seed in enumerate(SEEDS):
            if si < len(hists):
                continue                             # this seed already done
            cfg, cl, gc, tc, rounds, _ = build_task(task, alpha, seed)
            _, h = federated(spec, rounds, cl, gc, tc, cfg)
            hists.append(h)
            per["FedPrior"] = hists                  # separate key; FedPrior-pL untouched
            json.dump(per, open(path, "w"))
            metric = "mAP" if cfg.multilabel else "acc"
            print(f"[{tag} s{seed}] FedPrior(blend) {metric}={h['test_acc'][-1]:.4f}",
                  flush=True)
        print(f"[{tag}] FedPrior blend done", flush=True)
    print("RERUN_FEDPRIOR_BLEND_DONE", flush=True)


if __name__ == "__main__":
    main()
