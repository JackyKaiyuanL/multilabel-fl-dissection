#!/bin/bash
cd "$(dirname "$0")/.."
PY=python
$PY -m experiments.revision_runs --group cnx_lr --lrs 0.1
LR=$($PY - <<'PYEOF'
import json, statistics
d = json.load(open("experiments/full/rev/cnx_lr_voc_a0.1.json"))
best = max((statistics.mean(statistics.mean(h["test_acc"][-10:]) for h in v), k.split("@lr")[1]) for k, v in d.items() if k.startswith("FedAvg@"))
print(best[1])
PYEOF
)
echo "ConvNeXt LR chosen by FedAvg best: $LR"
$PY -m experiments.revision_runs --group cnx --lrs $LR --cells voc0.1 coco0.1
