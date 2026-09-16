#!/bin/bash
# Four parallel pipelines on the single RTX 5090 (each process ~2-6 GB).
cd "$(dirname "$0")/.."
PY=python
L=experiments/rev_logs
nohup bash -c "$PY -m experiments.revision_runs --group losses" > $L/p1_losses.log 2>&1 &
nohup bash -c "$PY -m experiments.revision_runs --group distill && $PY -m experiments.revision_runs --group seeds && $PY -m experiments.revision_runs --group losses_ref" > $L/p2_distill_seeds_ref.log 2>&1 &
nohup bash -c "$PY -m experiments.revision_runs --group margin && $PY -m experiments.revision_runs --group res224" > $L/p3_margin_res224.log 2>&1 &
nohup bash -c "$PY -m experiments.revision_runs --group cnx_lr" > $L/p4_cnx_lr.log 2>&1 &
echo launched
