"""Ablation for FedPrior (faithful pLasso): isolate the two add-ons on top of
the pLasso response-space prior (blend + L1). 2x2 over {per-class eta} x {SAM}.

    base (pLasso prior)   blend, l1, eta=0.1, per_class=0, sam=0
    +per-class            blend, l1, eta=0.1, per_class=1, sam=0
    +SAM                  blend, l1, eta=0.1, per_class=0, sam=0.05
    +both (FedPrior)      blend, l1, eta=0.1, per_class=1, sam=0.05

MNIST Dirichlet(0.1), 100 rounds, single seed.
"""
import json
import os

from fedprior.algorithms import TrainConfig, MethodSpec, federated
from fedprior.data import build_mnist_clients

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    "base (pLasso prior)": dict(per_class=False, sam_rho=0.0),
    "+per-class":          dict(per_class=True,  sam_rho=0.0),
    "+SAM":                dict(per_class=False, sam_rho=0.05),
    "+both (FedPrior)":    dict(per_class=True,  sam_rho=0.05),
}


def main(rounds=100, seed=0, device="cuda"):
    cfg = TrainConfig(model="cnn", lr=0.01, batchsize=64, local_steps=5,
                      select_clients=10, device=device, seed=seed)
    clients, gc, tc = build_mnist_clients(100, 0.1, seed)
    out = {}
    for name, kw in VARIANTS.items():
        spec = MethodSpec(name, prior=True, blend=True, not_true=False,
                          eta=0.1, temperature=1.0, l1=1e-5, **kw)
        _, h = federated(spec, rounds, clients, gc, tc, cfg)
        out[name] = h["test_acc"][-1]
        print(f"{name:22s} test_acc={out[name]:.4f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "results_ablation.json"), "w"))
    print("ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
