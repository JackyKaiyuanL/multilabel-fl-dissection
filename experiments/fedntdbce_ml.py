"""Run the FedNTD-BCE causal control on all four multi-label configs (VOC and COCO
at Dir 0.1/0.5), 3 seeds, saving full histories to the main results_<tag>.json
under key "FedNTD-BCE" so it can be reported as a flagged control row in Table III.
    python -m experiments.fedntdbce_ml
"""
import json, os
from fedprior.algorithms import METHODS, federated
from experiments.full_benchmark import build_task

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")
CONFIGS = [("voc", 0.1), ("voc", 0.5), ("coco", 0.1), ("coco", 0.5)]
SEEDS = (0, 1, 2)


def main():
    spec = METHODS["FedNTD-BCE"]
    for task, alpha in CONFIGS:
        tag = f"{task}_a{alpha}"
        path = os.path.join(OUT, f"results_{tag}.json")
        per = json.load(open(path)) if os.path.exists(path) else {}
        hists = per.get("FedNTD-BCE", [])
        if len(hists) >= len(SEEDS):
            print(f"[{tag}] FedNTD-BCE already done", flush=True); continue
        for si, seed in enumerate(SEEDS):
            if si < len(hists):
                continue
            cfg, cl, gc, tc, rounds, _ = build_task(task, alpha, seed)
            _, h = federated(spec, rounds, cl, gc, tc, cfg)
            hists.append(h)
            per["FedNTD-BCE"] = hists
            json.dump(per, open(path, "w"))
            print(f"[{tag} s{seed}] FedNTD-BCE mAP={h['test_acc'][-1]:.4f}", flush=True)
        print(f"[{tag}] FedNTD-BCE done", flush=True)
    print("FEDNTDBCE_ML_DONE", flush=True)


if __name__ == "__main__":
    main()
