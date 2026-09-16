# Dissecting Output-Space Methods for Multi-Label Federated Learning: A Controlled Study

Code and per-round result logs for

> Jacky Kaiyuan Li. *Dissecting Output-Space Methods for Multi-Label Federated Learning: A Controlled Study.*
> Workshop on Recent Advancement in Agentic and Federated AI (RAAF-AI) at IEEE AGCS 2026, Paris.

The paper is a controlled **dissection**, not a new method. It separates two axes of
multi-label federated learning: how the local loss **weights** the abundant negative
classes (standard vs. class-balanced BCE) and whether it **distills** them toward the
global model. Class-balanced BCE (BalBCE) is held fixed as a reference baseline and the
question is what output-space machinery (not-true masking, distillation, response
blending) adds once it is present. Datasets: Pascal VOC 2007, an MS-COCO subset, FLAIR
(real user partition), plus CIFAR-10/100 and a synthetic benchmark for the single-label
control.

## Layout

```
fedprior/                 model-agnostic FL package (the name is historical)
  algorithms.py           all methods as MethodSpec entries: FedAvg, FedProx, SCAFFOLD, FedSAM,
                          FedNTD (+ our sigmoid/BCE port), FedPrior (response blend), FedAwS, FLAG,
                          FedMLP, BalBCE, FedND(-std), focal/ASL/DB/WBCE losses, BalBCE+KD,
                          FedPrior-bal, FedMLP-bal, FedProx-bal; the federated driver; mAP evaluation
  models.py               GroupNorm ResNet-18 (from scratch), ImageNet ResNet-18/50 with frozen BN,
                          ConvNeXt-Tiny
  data.py                 Dirichlet partitions (CIFAR, VOC, COCO on the rarest present label),
                          FLAIR loader with the natural per-user partition
experiments/
  full_benchmark.py       from-scratch benchmark (Tables II, III; synthetic/CIFAR/VOC/COCO)
  pretrained_gate.py      pretrained ResNet-18 runs (Table I core rows)
  gap_runs.py, run_flag.py, rerun_fedaws_pt.py, fedntdbce_ml.py, disentangle_ml.py,
  fs_disentangle.py       remaining Table I / II / VII cells
  b2_orthogonality.py     FedMLP + balanced BCE (indented row of Table I)
  revision_runs.py        camera-ready groups: loss sweep (Table IV), distillation forms (Table V),
                          extra seeds (Table VII), ConvNeXt / 224 px / FLAIR (Table VI), Fig. 1
  aggregate_ml2.py        Tables I, II, VII from experiments/full/*.json -> experiments/tables/
  aggregate_sl.py         Table III (single-label CIFAR) -> experiments/tables/table_sl.tex
  aggregate_rev.py        Tables IV-VII and Fig. 1 from experiments/full/rev/*.json -> experiments/tables/
  full/                   per-run histories (test mAP every round, per seed) for every number in
                          the paper; rev/ holds the camera-ready groups; results_b2/ the FedMLP-bal runs
  logs/                   stdout logs of the camera-ready runs
data/                     empty; see data/README.md (datasets are not redistributed)
```

## Setup

```bash
conda create -n mlfl python=3.12 -y && conda activate mlfl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt
```

Datasets are third-party and **not included**. Follow `data/README.md`; VOC, CIFAR and
MNIST download themselves through torchvision, COCO `val2017` and FLAIR must be fetched
from their owners. All commands below run from the repository root, which is where
`./data` is resolved.

## Reproducing the paper

Every table is regenerated from the committed JSON histories without training:

```bash
python -m experiments.aggregate_ml2      # -> experiments/tables/tables_generated.tex (Tables I, II, VII)
python -m experiments.aggregate_sl       # -> experiments/tables/table_sl.tex (Table III)
python -m experiments.aggregate_rev      # -> experiments/tables/tables_rev.tex (Tables IV-VII) + figures/diag_margin_voc.png
```

To re-run the training (one RTX 5090; a pretrained ResNet-18 run takes ~3 min, a
ConvNeXt run ~15 min, a from-scratch run 5-10 min):

```bash
# Tables I-III (from scratch and pretrained ResNet-18, shared LR)
python -m experiments.full_benchmark --tasks synthetic cifar cifar100 voc coco --alphas 0.1 0.5 --seeds 0 1 2
python -m experiments.pretrained_gate
python -m experiments.gap_runs && python -m experiments.run_flag && python -m experiments.rerun_fedaws_pt
python -m experiments.b2_orthogonality

# Tables IV-VII and Fig. 1 (camera-ready groups; each writes experiments/full/rev/<group>_*.json)
python -m experiments.revision_runs --group losses      # + losses_lo losses_hi losses_m losses_m_hi losses_ref losses_ref_x
python -m experiments.revision_runs --group distill     # + distill_m
python -m experiments.revision_runs --group seeds
python -m experiments.revision_runs --group cnx_lr && python -m experiments.revision_runs --group cnx --lrs 0.03 --cells voc0.1 coco0.1
python -m experiments.revision_runs --group res224
python -m experiments.revision_runs --group flair
python -m experiments.revision_runs --group margin
```

`experiments/launch_rev.sh` shows how the groups were run in parallel. Every group is
resumable: finished (method, LR, seed) entries are skipped.

Protocol: 40 clients (VOC/COCO), 8 per round, 5 local epochs, batch 32, SGD with momentum
0.9 and cosine decay, 60 rounds pretrained / 100-150 from scratch; reported numbers are the
mean of the last 10 per-round test mAPs, averaged over seeds. FLAIR uses 100 sampled users
with their natural data, 10 per round, and the 17 coarse labels.

## Headline numbers (pretrained ResNet-18, mAP %, 3 seeds; see the paper for std)

| | VOC-.1 | COCO-.1 |
|---|---|---|
| FedAvg, shared LR 3e-3 | 62.0 | 13.6 |
| FedAvg, best LR (3e-2) | 67.8 | 26.0 |
| BalBCE, shared LR | 69.0 | 35.6 |
| BalBCE, best LR | 70.1 | 41.9 |
| FedNTD port (masking) | 33.2 | 9.5 |
| FedMLP port / + balanced BCE | 61.4 / 51.7 | 14.1 / 28.7 |
| FLAIR (real users): FedAvg / BalBCE / FedNTD | 39.6 / 44.2 / 27.3 | |

All named comparators are controlled re-implementations that share the standard/balanced
BCE pair, not faithful reproductions of the published systems; see Sec. IV of the paper.

## Checkpoints

Final global models for the from-scratch runs (~8 GB of `.pt` files) are not in this
repository; they are available on request from the author.

## Citation

```bibtex
@inproceedings{li2026dissecting,
  title     = {Dissecting Output-Space Methods for Multi-Label Federated Learning: A Controlled Study},
  author    = {Li, Jacky Kaiyuan},
  booktitle = {Workshop on Recent Advancement in Agentic and Federated AI (RAAF-AI), IEEE AGCS},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). Third-party datasets keep their own licenses.
