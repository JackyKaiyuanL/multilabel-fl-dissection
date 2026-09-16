"""Fill the experiment gaps needed for the re-centered (balancing) papers B/C:
 - BalBCE + FedND-std on alpha=0.5 (from-scratch AND pretrained) -> completes the
   2x2 at both skews and gives BalBCE a full multi-label row everywhere.
 - FedAwS on the pretrained backbone (all 4 cells) -> the dedicated multi-label
   baseline in the realistic regime.
Appends to the existing results files. 3 seeds.
    python -m experiments.gap_runs
"""
import json, os
from fedprior.algorithms import METHODS, federated
from experiments.full_benchmark import build_task
from experiments.pretrained_gate import _cfg as pt_cfg, _build as pt_build, ROUNDS as PT_ROUNDS

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "full")
SEEDS = (0, 1, 2)


def _append(path, name, runner):
    per = json.load(open(path)) if os.path.exists(path) else {}
    done = per.get(name, [])
    if len(done) >= len(SEEDS):
        return per
    for si, seed in enumerate(SEEDS):
        if si < len(done):
            continue
        h = runner(seed)
        done.append(h); per[name] = done; json.dump(per, open(path, "w"))
        print(f"[{os.path.basename(path)} s{seed}] {name} mAP={h['test_acc'][-1]:.4f}", flush=True)
    return per


def fs_runner(task, alpha, name):
    def r(seed):
        cfg, cl, gc, tc, rounds, _ = build_task(task, alpha, seed)
        return federated(METHODS[name], rounds, cl, gc, tc, cfg)[1]
    return r


def pt_runner(task, alpha, name):
    def r(seed):
        cfg = pt_cfg(task, seed); cl, gc, tc = pt_build(task, alpha, seed)
        return federated(METHODS[name], PT_ROUNDS, cl, gc, tc, cfg)[1]
    return r


def main():
    # 1) from-scratch alpha=0.5: BalBCE + FedND-std
    for task in ("voc", "coco"):
        path = os.path.join(OUT, f"results_{task}_a0.5.json")
        for name in ("BalBCE", "FedND-std"):
            _append(path, name, fs_runner(task, 0.5, name))
    # 2) pretrained alpha=0.5: BalBCE + FedND-std
    for task in ("voc", "coco"):
        path = os.path.join(OUT, f"results_pt_{task}_a0.5.json")
        for name in ("BalBCE", "FedND-std"):
            _append(path, name, pt_runner(task, 0.5, name))
    # 3) FedAwS on pretrained, all 4 cells
    for task in ("voc", "coco"):
        for alpha in (0.1, 0.5):
            path = os.path.join(OUT, f"results_pt_{task}_a{alpha}.json")
            _append(path, "FedAwS", pt_runner(task, alpha, "FedAwS"))
    print("GAP_RUNS_DONE", flush=True)


if __name__ == "__main__":
    main()
