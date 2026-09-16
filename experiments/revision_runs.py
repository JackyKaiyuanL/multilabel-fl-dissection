"""AGCS'26 camera-ready revision experiments (reviewer-requested), all on VOC/COCO.

Groups (each writes its own JSON under experiments/full/rev/ so groups can run in
parallel processes):
  losses      weighting axis widened: Focal / ASL / DB-loss / WBCE, pretrained ResNet-18,
              LR swept over {0.003 (paper default), 0.001, 0.01}, 4 cells x 3 seeds.
  losses_ref  FedAvg and BalBCE at the two non-default LRs (symmetry for the LR sweep).
  distill     distillation axis widened on top of balancing: BalBCE+KD (all-class KL),
              FedPrior-bal (balanced blend), FedProx-bal (parameter-space control),
              Focal+ND / ASL+ND (neg-distill under other balancers).
  seeds       seeds 3,4 for the pretrained 2x2 (FedAvg, FedND-std, BalBCE, FedNTD-BCE).
  cnx_lr      ConvNeXt-Tiny LR check (FedAvg, BalBCE; seed 0; VOC-.1).
  cnx         second backbone family: ConvNeXt-Tiny (LayerNorm), key methods.
  res224      224x224 inputs (pretrained ResNet-18), key methods, Dir(0.1).
  margin      Fig. 1 over 3 seeds: from-scratch VOC-.1, score-margin trajectories.

    python -m experiments.revision_runs --group losses [--cells voc0.1 coco0.1] [--lr 0.003]
"""
import argparse, json, os, statistics, sys, time
import torch
from fedprior.algorithms import TrainConfig, METHODS, federated
from fedprior import data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "full", "rev")
os.makedirs(OUT, exist_ok=True)
ALL_CELLS = ["voc0.1", "coco0.1", "voc0.5", "coco0.5"]
KEY = ["FedAvg", "BalBCE", "FedNTD-BCE", "FedNTD", "FedMLP", "FedPrior"]

GROUPS = {
    "losses":     dict(model="resnet_ml_pt", lrs=[0.003, 0.001, 0.01], rounds=60,
                       methods=["FocalBCE", "ASL", "DBLoss", "WBCE"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "losses_m":   dict(model="resnet_ml_pt", lrs=[0.003, 0.01, 0.001], rounds=60,
                       methods=["FocalBCE-m", "ASL-m", "DBLoss-m"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "distill_m":  dict(model="resnet_ml_pt", lrs=[0.003], rounds=60,
                       methods=["Focal-m+ND", "ASL-m+ND"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    # grid extensions so that every loss's best LR is interior to its grid (or reported at the edge)
    "losses_m_hi": dict(model="resnet_ml_pt", lrs=[0.03, 0.1], rounds=60,
                        methods=["FocalBCE-m", "ASL-m", "DBLoss-m"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "losses_lo":   dict(model="resnet_ml_pt", lrs=[0.0003], rounds=60,
                        methods=["FocalBCE", "ASL", "DBLoss"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "losses_hi":   dict(model="resnet_ml_pt", lrs=[0.03], rounds=60,
                        methods=["WBCE", "FedAvg", "BalBCE"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "losses_ref_x": dict(model="resnet_ml_pt", lrs=[0.1], rounds=60,
                         methods=["FedAvg"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "losses_ref": dict(model="resnet_ml_pt", lrs=[0.001, 0.01], rounds=60,
                       methods=["FedAvg", "BalBCE"], cells=ALL_CELLS, seeds=[0, 1, 2]),
    "distill":    dict(model="resnet_ml_pt", lrs=[0.003], rounds=60,
                       methods=["BalBCE+KD", "FedPrior-bal", "FedProx-bal", "Focal+ND", "ASL+ND"],
                       cells=ALL_CELLS, seeds=[0, 1, 2]),
    "seeds":      dict(model="resnet_ml_pt", lrs=[0.003], rounds=60,
                       methods=["FedAvg", "FedND-std", "BalBCE", "FedNTD-BCE"], cells=ALL_CELLS, seeds=[3, 4]),
    "cnx_lr":     dict(model="convnext_ml_pt", lrs=[0.003, 0.001, 0.01], rounds=60,
                       methods=["FedAvg", "BalBCE"], cells=["voc0.1"], seeds=[0]),
    "cnx":        dict(model="convnext_ml_pt", lrs=[0.003], rounds=60,
                       methods=KEY, cells=["voc0.1", "coco0.1", "voc0.5", "coco0.5"], seeds=[0, 1, 2]),
    "res224":     dict(model="resnet_ml_pt", lrs=[0.003], rounds=60, size=224,
                       methods=KEY, cells=["coco0.1", "voc0.1"], seeds=[0, 1, 2]),
    "flair":      dict(model="resnet_ml_pt", lrs=[0.003], rounds=60,
                       methods=KEY + ["FedND-std", "FLAG", "ASL", "FocalBCE"], cells=["flair"], seeds=[0, 1, 2]),
    "margin":     dict(model="resnet_ml", lrs=[0.01], rounds=100, margin=True,
                       methods=["FedAvg", "FedNTD", "FedND-std", "FedNTD-BCE", "FedPrior"],
                       cells=["voc0.1"], seeds=[0, 1, 2]),
}


def make_cfg(model, task, seed, lr):
    ncls = {"coco": 80, "flair": 17}.get(task, 20)
    return TrainConfig(model=model, model_kw={"num_classes": ncls}, lr=lr, momentum=0.9,
                       sam_rho=(0.02 if model == "resnet_ml" else None), augment=True,
                       lr_schedule="cosine", batchsize=32, local_steps=5, select_clients=8,
                       device="cuda", seed=seed, eval_batch=128, multilabel=True)


def build(task, alpha, seed, size=128):
    for attempt in range(4):                      # retry transient image-read errors
        try:
            if task == "flair":               # real user partition; 100 users, 10/round
                return data.build_flair_clients(seed=seed, size=size)
            if task == "voc":
                return data.build_voc_clients(40, alpha, seed, size=size)
            return data.build_coco_clients(40, alpha, seed, size=size)
        except Exception as e:                    # pragma: no cover
            print(f"  [build retry {attempt}] {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
    raise RuntimeError("data build failed")


@torch.no_grad()
def _margin(state, cfg, test_client):
    """Mean positive-label sigmoid score minus mean negative-label score (test set)."""
    from fedprior.models import build_model
    dev = torch.device(cfg.device)
    net = build_model(cfg.model, **cfg.model_kw).to(dev); net.load_state_dict(state); net.eval()
    P, Y = [], []
    for i in range(0, test_client.number, cfg.eval_batch):
        xb = test_client.x[i:i + cfg.eval_batch].to(dev).float()
        P.append(torch.sigmoid(net(xb)).cpu()); Y.append(test_client.y[i:i + cfg.eval_batch].cpu())
    P = torch.nan_to_num(torch.cat(P), nan=0.0); Y = torch.cat(Y)
    return P[Y == 1].mean().item(), P[Y == 0].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--lrs", nargs="*", type=float, default=None)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    g = dict(GROUPS[a.group])
    cells = a.cells or g["cells"]; methods = a.methods or g["methods"]
    seeds = a.seeds or g["seeds"]; lrs = a.lrs or g["lrs"]
    size = g.get("size", 128)
    for cell in cells:
        task, alpha = (cell, None) if cell == "flair" else (cell[:-3], float(cell[-3:]))
        path = os.path.join(OUT, f"{a.group}_{task}_a{alpha}.json")
        for seed in seeds:
            per = json.load(open(path)) if os.path.exists(path) else {}
            todo = [(m, lr) for lr in lrs for m in methods
                    if not any(h.get("seed") == seed for h in per.get(f"{m}@lr{lr}", []))]
            if not todo:
                continue
            cl, gc, tc = build(task, alpha, seed, size)
            for m, lr in todo:
                key = f"{m}@lr{lr}"
                cfg = make_cfg(g["model"], task, seed, lr)
                if task == "flair":
                    cfg.select_clients = 10       # 10 of 100 users per round
                logged = {"rounds": [], "pos": [], "neg": []}
                hook = None
                if g.get("margin"):
                    def hook(r, state, _l=logged):
                        if r % 2 == 0:
                            pm, nm = _margin(state, cfg, tc)
                            _l["rounds"].append(r); _l["pos"].append(pm); _l["neg"].append(nm)
                t0 = time.time()
                _, h = federated(METHODS[m], g["rounds"], cl, gc, tc, cfg, on_round=hook)
                h["seed"] = seed; h["lr"] = lr; h["model"] = g["model"]; h["size"] = size
                if g.get("margin"):
                    h["margin"] = logged
                per = json.load(open(path)) if os.path.exists(path) else {}
                per.setdefault(key, []).append(h)
                json.dump(per, open(path, "w"))
                last10 = statistics.mean(h["test_acc"][-10:]) * 100
                print(f"[{a.group} {cell} s{seed}] {key:22s} last10 mAP={last10:5.2f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            del cl, gc, tc; torch.cuda.empty_cache()
    print(f"REVISION_GROUP_DONE {a.group}", flush=True)


if __name__ == "__main__":
    main()
