"""M3: re-tune FedProx with a FIXED proximal mu (the adaptive-mu heuristic ran
away to ~29.5 over 600 rounds and over-regularized).

Phase 1: grid mu in {0.01, 0.1, 1.0} on CIFAR-10 Dir(0.1), 1 seed, 600 rounds.
Phase 2: re-run the best mu on the 4 CIFAR configs, 3 seeds, for the paper table.

Saves experiments/full/results_fedprox_retune.json:
  {"grid": {mu: acc}, "best_mu": mu, "<tag>": [acc per seed]}
    python -m experiments.fedprox_grid
"""
import json
import os
import statistics

from fedprior.algorithms import MethodSpec, federated
from experiments.full_benchmark import build_task

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "full", "results_fedprox_retune.json")

GRID = [0.01, 0.1, 1.0]
CONFIGS = [("cifar", 0.1), ("cifar", 0.5), ("cifar100", 0.1), ("cifar100", 0.5)]


def _spec(mu):
    return MethodSpec(f"FedProx-mu{mu}", prox=True, adaptive_mu=False, mu0=mu,
                      excludes_stragglers=False)


def _final(h):
    return statistics.mean(h["test_acc"][-10:]) * 100


def run_one(task, alpha, mu, seed):
    cfg, cl, gc, tc, rounds, _ = build_task(task, alpha, seed)
    _, h = federated(_spec(mu), rounds, cl, gc, tc, cfg)
    return _final(h)


def main():
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}

    # Phase 1: grid on CIFAR-10 Dir(0.1), seed 0
    grid = out.get("grid", {})
    for mu in GRID:
        if str(mu) in grid:
            continue
        acc = run_one("cifar", 0.1, mu, 0)
        grid[str(mu)] = acc
        print(f"[grid] mu={mu:<5} CIFAR10-0.1 acc={acc:.2f}", flush=True)
        out["grid"] = grid; json.dump(out, open(PATH, "w"))
    best = max(grid, key=grid.get)
    out["best_mu"] = float(best); json.dump(out, open(PATH, "w"))
    print(f"[grid] BEST mu={best} (acc={grid[best]:.2f})", flush=True)

    # Phase 2: best mu, 3 seeds, on the 4 CIFAR configs
    mu = float(best)
    for task, alpha in CONFIGS:
        tag = f"{task}_a{alpha}"
        accs = out.get(tag, [])
        for seed in range(len(accs), 3):
            acc = run_one(task, alpha, mu, seed)
            accs.append(acc)
            print(f"[retune] {tag} mu={mu} s{seed} acc={acc:.2f}", flush=True)
            out[tag] = accs; json.dump(out, open(PATH, "w"))

    print("\n=== FEDPROX RETUNE SUMMARY ===")
    print("grid:", {k: round(v, 1) for k, v in grid.items()}, "best_mu:", best)
    for task, alpha in CONFIGS:
        a = out[f"{task}_a{alpha}"]
        print(f"  {task}_a{alpha}: {statistics.mean(a):.1f}+/-{statistics.pstdev(a):.1f}")
    print("FEDPROX_RETUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
