"""Disentangle FedND's multi-label gain: is it class-BALANCED BCE or NEGATIVE
DISTILLATION? Runs the two missing 2x2 cells (BalBCE, FedND-std) on the pretrained
backbone for VOC-0.1 and COCO-0.1, appending to the gate's result files (which
already hold FedAvg and FedNTD-BCE). Then prints the 2x2.

    2x2:  std BCE / balanced BCE   x   no distill / + neg distill
          FedAvg  / BalBCE              FedND-std  / FedNTD-BCE(=FedND)

    python -m experiments.disentangle_ml
"""
import json, os, statistics
from fedprior.algorithms import METHODS, federated
from experiments.pretrained_gate import _cfg, _build, ROUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full")
CONFIGS = [("voc", 0.1), ("coco", 0.1)]
NEW = ["BalBCE", "FedND-std"]
SEEDS = (0, 1, 2)


def main():
    for task, alpha in CONFIGS:
        tag = f"pt_{task}_a{alpha}"
        path = os.path.join(OUT, f"results_{tag}.json")
        per = json.load(open(path)) if os.path.exists(path) else {}
        for name in NEW:
            done = per.get(name, [])
            if len(done) >= len(SEEDS):
                continue
            for si, seed in enumerate(SEEDS):
                if si < len(done):
                    continue
                cfg = _cfg(task, seed); cl, gc, tc = _build(task, alpha, seed)
                _, h = federated(METHODS[name], ROUNDS, cl, gc, tc, cfg)
                done.append(h); per[name] = done
                json.dump(per, open(path, "w"))
                print(f"[{tag} s{seed}] {name:10s} mAP={h['test_acc'][-1]:.4f}", flush=True)

    print("\n=== 2x2 DISENTANGLING (pretrained, mAP%) ===")
    for task, alpha in CONFIGS:
        tag = f"pt_{task}_a{alpha}"
        d = json.load(open(os.path.join(OUT, f"results_{tag}.json")))
        def m(k):
            return statistics.mean(statistics.mean(h["test_acc"][-10:])*100 for h in d[k]) if k in d else float('nan')
        print(f"\n{tag}:")
        print(f"               no-distill   +neg-distill")
        print(f"  std BCE      {m('FedAvg'):8.1f}   {m('FedND-std'):8.1f}")
        print(f"  balanced BCE {m('BalBCE'):8.1f}   {m('FedNTD-BCE'):8.1f}")
    print("DISENTANGLE_DONE", flush=True)


if __name__ == "__main__":
    main()
