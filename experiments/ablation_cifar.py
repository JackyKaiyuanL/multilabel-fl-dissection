"""Ablation for FedPrior on CIFAR-10 Dir(0.1): the two add-ons on top of the
pLasso response-space prior (blend + L1). 2x2 over {per-class eta} x {SAM}.
3 seeds, 600 rounds, cosine LR decay (matches the main CIFAR setup;
"+both" reproduces the main-table FedPrior CIFAR-10 Dir(0.1) number).

    base (pLasso prior)   blend, l1, eta=0.1, per_class=0, sam=0
    +per-class            per_class=1
    +SAM                  sam=0.02
    +both (FedPrior)      per_class=1, sam=0.02
"""
import json
import os
import statistics

from fedprior.algorithms import TrainConfig, MethodSpec, federated
from fedprior.data import build_cifar_clients

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {
    "base (pLasso prior)": dict(per_class=False, sam_rho=0.0),
    "+per-class":          dict(per_class=True,  sam_rho=0.0),
    "+SAM":                dict(per_class=False, sam_rho=0.02),
    "+both (FedPrior)":    dict(per_class=True,  sam_rho=0.02),
}


def main(rounds=600, seeds=(0, 1, 2), device="cuda"):
    out = {}
    for name, kw in VARIANTS.items():
        accs = []
        for seed in seeds:
            cfg = TrainConfig(model="resnet", model_kw={"num_classes": 10},
                              lr=0.03, momentum=0.9, augment=True,
                              lr_schedule="cosine", batchsize=64, local_steps=5,
                              select_clients=10, device=device, seed=seed,
                              eval_batch=1000)
            cl, gc, tc = build_cifar_clients(100, 0.1, seed)
            spec = MethodSpec(name, prior=True, blend=True, not_true=False,
                              eta=0.1, temperature=1.0, l1=1e-5,
                              per_class=kw["per_class"], sam_rho=kw["sam_rho"])
            _, h = federated(spec, rounds, cl, gc, tc, cfg)
            accs.append(statistics.mean(h["test_acc"][-10:]) * 100)
            json.dump(out, open(os.path.join(HERE, "results_ablation_cifar.json"), "w"))
        out[name] = accs
        print(f"{name:22s} {statistics.mean(accs):.2f} +/- {statistics.pstdev(accs):.2f}",
              flush=True)
        json.dump(out, open(os.path.join(HERE, "results_ablation_cifar.json"), "w"))
    print("ABLATION_CIFAR_DONE", flush=True)


if __name__ == "__main__":
    main()
