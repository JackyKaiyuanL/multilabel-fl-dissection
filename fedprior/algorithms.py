"""Model-agnostic federated optimisation algorithms.

All methods share one driver (:func:`federated`) operating on full model
``state_dict``s, so they work for both the synthetic linear model and the MNIST
CNN. A method is described by a :class:`MethodSpec`.

Implemented methods
-------------------
* ``FedAvg``     – McMahan et al. 2017.
* ``FedProx``    – Li et al. 2020; ``+ 0.5*mu*||w-w0||^2`` (adaptive mu).
* ``SCAFFOLD``   – Karimireddy et al. 2020; control variates correct client drift.
* ``FedNTD``     – Lee et al. 2022; not-true distillation toward the global model.
* ``FedSAM``     – Qu et al. 2022; FedAvg with a sharpness-aware local step.
* ``FedPrior``   – *this work*; pLasso-style response-space prior: distillation
  toward the global model with **per-class adaptive prior strength** (eta_c,
  inverse local-frequency), not-true masking, and a **sharpness-aware (SAM)
  local step**. Removing the per-class weighting and SAM reduces it to FedNTD.

The prior / distillation term lives in *output (response) space*, which is the
pLasso philosophy and distinguishes this family from FedProx/SCAFFOLD/FedDyn
(parameter space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model, Client


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    model: str = "linear"            # 'linear' | 'cnn'
    lr: float = 0.0005
    batchsize: int = 10
    local_steps: int = 20            # local epochs for an active client
    select_clients: int = 10
    straggler_rate: float = 0.0
    device: str = "cpu"
    eval_batch: int = 4096
    seed: int = 0
    momentum: float = 0.0            # local SGD momentum (resets each round)
    sam_rho: float = None            # overrides MethodSpec.sam_rho when set
    augment: bool = False            # per-batch flip + random crop (image tasks)
    lr_schedule: str = None          # None or 'cosine' (decay over rounds)
    model_kw: dict = field(default_factory=dict)   # extra kwargs for build_model
    multilabel: bool = False                        # multi-label (sigmoid/BCE) task


@dataclass
class MethodSpec:
    name: str
    prox: bool = False               # add proximal term
    adaptive_mu: bool = True
    mu0: float = 0.0
    prior: bool = False              # add distillation prior
    not_true: bool = False           # FedNTD-style not-true masking
    blend: bool = False              # pLasso: fit to blended response (no masking)
    l1: float = 0.0                  # pLasso L1 sparsity penalty (lambda)
    per_class: bool = False          # per-class adaptive eta (inverse freq)
    eta: float = 0.1                 # base prior strength (beta in FedNTD)
    temperature: float = 1.0
    sam_rho: float = 0.0             # sharpness-aware radius (0 = off)
    scaffold: bool = False
    spreadout: float = 0.0           # FedAwS: server-side embedding spreadout
    fedmlp: bool = False             # FedMLP: consistency + pseudo-label (multi-label)
    nt_keep_bce: bool = False        # M4 control: not-true that RETAINS BCE on negatives
    balanced_bce: bool = False       # ablation: class-balanced BCE (pos/neg normalized apart)
    neg_distill: bool = False        # ablation: + per-class KL distillation of negatives
    flag: bool = False               # FLAG: label-adaptive per-class head aggregation
    loss: str = "bce"                # multi-label base loss (weighting axis): bce|bal|focal|asl|db|wbce
    full_distill: bool = False       # + all-class binary KL toward the global teacher (plain KD)
    loss_reduction: str = "sum"      # focal/ASL/DB: "sum" = class-sum/batch-mean (reference code),
                                     #               "mean" = entry-mean (same scale as BCE)
    excludes_stragglers: bool = True  # FedAvg drops stragglers; others run partial


# Registry of the methods used in the benchmark.
METHODS = {
    "FedAvg":     MethodSpec("FedAvg"),
    "FedProx":    MethodSpec("FedProx", prox=True, adaptive_mu=True,
                             excludes_stragglers=False),
    "SCAFFOLD":   MethodSpec("SCAFFOLD", scaffold=True, excludes_stragglers=False),
    "FedNTD":     MethodSpec("FedNTD", prior=True, not_true=True, eta=1.0,
                             temperature=1.0),
    # M4 control (multi-label): not-true masking that RETAINS BCE on the negatives,
    # to test whether restoring negative suppression prevents the mAP collapse.
    "FedNTD-BCE": MethodSpec("FedNTD-BCE", prior=True, not_true=True, eta=1.0,
                             temperature=1.0, nt_keep_bce=True),
    # 2x2 disentangling ablation (multi-label): is the gain class-balancing or
    # negative distillation? FedAvg=std BCE; BalBCE=balanced BCE; FedND-std=std
    # BCE + neg distill; FedND(=FedNTD-BCE)=balanced BCE + neg distill.
    "BalBCE":      MethodSpec("BalBCE", prior=True, balanced_bce=True, eta=1.0),
    "FedND-std":   MethodSpec("FedND-std", prior=True, neg_distill=True, eta=1.0),
    # FLAG (Chu et al. 2023): label-adaptive aggregation -- per-class classifier
    # rows averaged by clients' per-class positive counts; backbone is FedAvg.
    "FLAG":        MethodSpec("FLAG", flag=True, excludes_stragglers=False),
    "FedSAM":     MethodSpec("FedSAM", sam_rho=0.05),
    # FedPrior (this work): the prior-LASSO (pLasso) response-space prior -- fit
    # each client to the blended response (y + eta*q^g)/(1+eta) with an L1 penalty
    # (eta=0.1). This is the bare blend (the original pLasso idea); the per-class
    # adaptive strength and SAM step are explored extensions (ablated, not part of
    # the headline method, since they do not help -- see ablation).
    "FedPrior":    MethodSpec("FedPrior", prior=True, blend=True, not_true=False,
                              per_class=False, eta=0.1, temperature=1.0,
                              l1=1e-5, sam_rho=0.0),
    # (The not-true-masking counterpart, bare, is exactly FedNTD above, so the
    #  blend-vs-mask comparison is FedPrior vs FedNTD; no separate FedPrior-NT.)
    # --- dedicated multi-label FL baselines (VOC only) ---
    # FedAwS (Yang et al., ICML 2020): server-side spreadout of class embeddings.
    "FedAwS": MethodSpec("FedAwS", spreadout=1.0),
    # FedMLP (Liu et al., MICCAI 2024): global-teacher consistency + pseudo-label
    # tagging of high-confidence classes (our compact reimplementation).
    "FedMLP": MethodSpec("FedMLP", prior=True, fedmlp=True, eta=1.0),
    # ---- AGCS camera-ready revision: widen both axes of the dissection ----
    # Weighting axis: established imbalance-aware multi-label objectives (no teacher).
    "FocalBCE":  MethodSpec("FocalBCE", loss="focal"),      # Lin et al. 2017, gamma=2
    "ASL":       MethodSpec("ASL", loss="asl"),             # Ridnik et al. 2021, g+=0, g-=4, m=0.05
    "DBLoss":    MethodSpec("DBLoss", loss="db"),           # Wu et al. 2020 (re-balanced + NTR + focal)
    "WBCE":      MethodSpec("WBCE", loss="wbce"),           # per-class pos_weight BCE (local freq.)
    # ... each with the SAME negative distillation as FedND (distillation axis under other balancers)
    "Focal+ND":  MethodSpec("Focal+ND", prior=True, loss="focal", neg_distill=True, eta=1.0),
    # entry-mean variants (loss scale matched to BCE, so the shared LR grid applies as-is)
    "FocalBCE-m": MethodSpec("FocalBCE-m", loss="focal", loss_reduction="mean"),
    "ASL-m":      MethodSpec("ASL-m", loss="asl", loss_reduction="mean"),
    "DBLoss-m":   MethodSpec("DBLoss-m", loss="db", loss_reduction="mean"),
    "Focal-m+ND": MethodSpec("Focal-m+ND", prior=True, loss="focal", loss_reduction="mean",
                             neg_distill=True, eta=1.0),
    "ASL-m+ND":   MethodSpec("ASL-m+ND", prior=True, loss="asl", loss_reduction="mean",
                             neg_distill=True, eta=1.0),
    "ASL+ND":    MethodSpec("ASL+ND", prior=True, loss="asl", neg_distill=True, eta=1.0),
    # Distillation axis on top of balanced BCE: alternative output-space formulations.
    "BalBCE+KD":    MethodSpec("BalBCE+KD", prior=True, balanced_bce=True, full_distill=True, eta=1.0),
    "FedPrior-bal": MethodSpec("FedPrior-bal", prior=True, blend=True, balanced_bce=True,
                               eta=0.1, l1=1e-5),
    # B2 orthogonality check: FedMLP with its BCE class-balanced.
    "FedMLP-bal":   MethodSpec("FedMLP-bal", prior=True, fedmlp=True, eta=1.0, balanced_bce=True),
    # Parameter-space control on top of balanced BCE (fixed mu=0.1 as in the paper's FedProx).
    "FedProx-bal":  MethodSpec("FedProx-bal", prox=True, adaptive_mu=False, mu0=0.1,
                               balanced_bce=True, excludes_stragglers=False),
}


# --------------------------------------------------------------------------- #
# state_dict helpers
# --------------------------------------------------------------------------- #
def _clone_state(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


def _zeros_like_state(sd):
    return {k: torch.zeros_like(v) for k, v in sd.items()}


def _avg_states(states, weights):
    """Weighted average of a list of state_dicts (weights sum-normalised).
    Non-float buffers (e.g. a frozen BatchNorm's integer ``num_batches_tracked``)
    are identical across clients, so they are copied rather than averaged."""
    total = float(sum(weights))
    out = {}
    for k, v in states[0].items():
        if not v.is_floating_point():
            out[k] = v.clone(); continue
        acc = torch.zeros_like(v)
        for sd, w in zip(states, weights):
            acc += sd[k] * (w / total)
        out[k] = acc
    return out


# --------------------------------------------------------------------------- #
# Distillation losses (the pLasso "prior" term), returned per-sample
# --------------------------------------------------------------------------- #
def _soft_distill_per_sample(student_logits, teacher_logits, T):
    q = F.softmax(teacher_logits / T, dim=1)
    logp = F.log_softmax(student_logits / T, dim=1)
    return -(q * logp).sum(dim=1) * (T * T)


def _not_true_distill_per_sample(student_logits, teacher_logits, y_true, T):
    """KD over the not-true (non ground-truth) classes only (FedNTD)."""
    mask = torch.zeros_like(student_logits, dtype=torch.bool)
    mask[torch.arange(len(y_true), device=y_true.device), y_true] = True
    s = (student_logits / T).masked_fill(mask, float("-inf"))
    t = (teacher_logits / T).masked_fill(mask, float("-inf"))
    logp = F.log_softmax(s, dim=1)
    q = F.softmax(t, dim=1)
    # per-sample KL(q || p) over the kept classes
    kl = (q * (torch.log(q.clamp_min(1e-12)) - logp)).sum(dim=1)
    return kl * (T * T)


def _binary_kl(a, b, eps=1e-6):
    """Per-element KL between Bernoulli(a) (teacher) and Bernoulli(b) (student)."""
    a = a.clamp(eps, 1 - eps); b = b.clamp(eps, 1 - eps)
    return a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))


def _reduce(per_entry, reduction):
    return per_entry.mean() if reduction == "mean" else per_entry.sum(1).mean()


def _focal_ml(logits, y, gamma=2.0, reduction="sum"):
    """Multi-label focal BCE (Lin et al. 2017; symmetric, gamma=2). Reduction as in the
    ASL/DB reference code: sum over classes, mean over the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    return _reduce(((1 - pt) ** gamma) * bce, reduction)


def _asl_ml(logits, y, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="sum"):
    """Asymmetric loss (Ridnik et al. 2021), reference defaults; class-sum/batch-mean."""
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos
    if clip > 0:                                     # probability shifting of negatives
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    los_pos = y * torch.log(xs_pos.clamp_min(eps))
    los_neg = (1 - y) * torch.log(xs_neg.clamp_min(eps))
    loss = los_pos + los_neg
    pt = xs_pos * y + xs_neg * (1 - y)
    gamma = gamma_pos * y + gamma_neg * (1 - y)
    loss = loss * (1 - pt) ** gamma                  # asymmetric focusing
    return -_reduce(loss, reduction)


def _db_ml(logits, y, st, reduction="sum"):
    """Distribution-balanced loss (Wu et al. 2020): re-balanced instance/class weighting
    (alpha + sigmoid(beta*(r - mu))), negative-tolerant regularisation (class-specific bias
    kappa*log(n_k/(N-n_k)) and negative logit scale lambda), and focal (gamma=2), following
    the authors' ResampleLoss. Class frequencies are the client's local ones (the only
    ones a federated client can see)."""
    rep_rate = (y * st["freq_inv"]).sum(1, keepdim=True).clamp_min(1e-8)
    w = st["alpha"] + torch.sigmoid(st["beta"] * (st["freq_inv"].unsqueeze(0) / rep_rate - st["mu"]))
    z = logits + st["init_bias"]
    z = z * (1 - y) * st["neg_scale"] + z * y
    w = w / st["neg_scale"] * (1 - y) + w * y
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = torch.exp(-bce)
    return _reduce(w * ((1 - pt) ** 2) * bce, reduction)


def _augment_batch(xb):
    """Per-sample random horizontal flip + a (batch-shared) reflect-pad crop."""
    n, c, h, w = xb.shape
    flip = torch.rand(n, device=xb.device) < 0.5
    if flip.any():
        xb = xb.clone()
        xb[flip] = torch.flip(xb[flip], dims=[3])
    pad = max(2, h // 8)
    xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    oy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    ox = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    return xp[:, :, oy:oy + h, ox:ox + w]


# --------------------------------------------------------------------------- #
# Local solver
# --------------------------------------------------------------------------- #
class LocalTrainer:
    """Runs local training for one client from global ``state``. Manual SGD so
    proximal / distillation / SAM / SCAFFOLD terms compose cleanly."""

    def __init__(self, client: Client, state, spec: MethodSpec, cfg: TrainConfig):
        self.spec, self.cfg = spec, cfg
        self.device = torch.device(cfg.device)
        self.net = build_model(cfg.model, **cfg.model_kw).to(self.device)
        self.net.load_state_dict(state)
        self.x = client.x.to(self.device).float()
        self.y = client.y.to(self.device)
        self.y = self.y.float() if cfg.multilabel else self.y.long()
        self.num = client.number
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.w0 = [p.detach().clone() for p in self.params]   # for prox

        self.teacher = None
        if spec.prior:
            self.teacher = build_model(cfg.model, **cfg.model_kw).to(self.device)
            self.teacher.load_state_dict(state)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        # per-class adaptive eta = eta * (1 - local_class_frequency)
        self.eta_vec = None
        if spec.prior and spec.per_class:
            if cfg.multilabel:
                freq = self.y.sum(0) / max(self.num, 1)        # per-class positive rate
            else:
                counts = torch.bincount(self.y, minlength=self._num_classes()).float()
                freq = counts / counts.sum().clamp_min(1)
            self.eta_vec = (spec.eta * (1.0 - freq)).to(self.device)

        # local label statistics for the frequency-aware losses (WBCE, DB-loss)
        self.pos_weight = self.db_stats = None
        if cfg.multilabel and spec.loss in ("wbce", "db"):
            n_k = self.y.sum(0)
            N = float(self.num)
            if spec.loss == "wbce":
                self.pos_weight = ((N - n_k) / n_k.clamp_min(1.0)).clamp(1.0, 100.0)
            else:
                nk = n_k.clamp(1.0, max(N - 1.0, 1.0))
                self.db_stats = dict(freq_inv=1.0 / nk,
                                     init_bias=-torch.log(N / nk - 1.0) * 0.05,
                                     neg_scale=2.0, alpha=0.1, beta=10.0, mu=0.2)

    def _num_classes(self):
        # last Linear out_features
        for m in reversed(list(self.net.modules())):
            if isinstance(m, torch.nn.Linear):
                return m.out_features
        return 10

    @torch.no_grad()
    def cal_loss(self):
        logits = self.net(self.x)
        if self.cfg.multilabel:
            return F.binary_cross_entropy_with_logits(logits, self.y).item()
        return F.cross_entropy(logits, self.y).item()

    def _base_ml_loss(self, logits, y):
        """Multi-label base loss on (hard or pseudo) labels ``y``: the *weighting axis*."""
        kind = "bal" if self.spec.balanced_bce else self.spec.loss
        if kind == "bce":                          # standard mean BCE over all entries (FedAvg)
            return F.binary_cross_entropy_with_logits(logits, y)
        if kind == "bal":                          # class-balanced: pos-mean + neg-mean (BalBCE)
            bce_all = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            return (bce_all * y).sum() / y.sum().clamp_min(1) \
                + (bce_all * (1 - y)).sum() / (1 - y).sum().clamp_min(1)
        if kind == "wbce":                         # per-class pos_weight = n_neg/n_pos (local)
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        if kind == "focal":
            return _focal_ml(logits, y, reduction=self.spec.loss_reduction)
        if kind == "asl":
            return _asl_ml(logits, y, reduction=self.spec.loss_reduction)
        if kind == "db":
            return _db_ml(logits, y, self.db_stats, reduction=self.spec.loss_reduction)
        raise ValueError(f"unknown multi-label loss '{self.spec.loss}'")

    def _batch_loss_ml(self, xb, yb):
        """Multi-label (sigmoid/BCE) local loss = base loss (weighting axis) + optional
        output-space term toward the global teacher (distillation axis)."""
        spec = self.spec
        logits = self.net(xb)                      # (B, C)
        p = torch.sigmoid(logits)
        tg = torch.sigmoid(self.teacher(xb)) if spec.prior else None   # teacher probs
        eta_c = (self.eta_vec if self.eta_vec is not None
                 else torch.full((logits.shape[1],), spec.eta,
                                 device=logits.device)).unsqueeze(0)    # (1, C)
        if spec.fedmlp:                            # FedMLP: pseudo-label + consistency
            y_aug = torch.maximum(yb, (tg > 0.8).float())   # tag confident classes
            loss = self._base_ml_loss(logits, y_aug) + spec.eta * _binary_kl(tg, p).mean()
        elif spec.blend:                           # pLasso per-class response blend
            target = (yb + eta_c * tg) / (1.0 + eta_c)
            bce_all = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            if spec.balanced_bce:                  # FedPrior-bal: balance by the hard label
                loss = (bce_all * yb).sum() / yb.sum().clamp_min(1) \
                    + (bce_all * (1 - yb)).sum() / (1 - yb).sum().clamp_min(1)
            else:
                loss = bce_all.mean()
        elif spec.not_true and not spec.nt_keep_bce:   # FedNTD port: BCE on positives only,
            bce_all = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce_pos = (bce_all * yb).sum() / yb.sum().clamp_min(1)   # + distill the negatives
            loss = bce_pos + (eta_c * _binary_kl(tg, p) * (1 - yb)).sum() \
                / (1 - yb).sum().clamp_min(1)
        else:
            if spec.nt_keep_bce:                   # FedND (=FedNTD-BCE): balanced BCE + neg distill
                bce_all = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                loss = (bce_all * yb).sum() / yb.sum().clamp_min(1) \
                    + (bce_all * (1 - yb)).sum() / (1 - yb).sum().clamp_min(1)
            else:
                loss = self._base_ml_loss(logits, yb)
            if spec.neg_distill or spec.nt_keep_bce:   # + per-class KL distillation of negatives
                loss = loss + (eta_c * _binary_kl(tg, p) * (1 - yb)).sum() \
                    / (1 - yb).sum().clamp_min(1)
            if spec.full_distill:                  # + all-class KL toward the teacher (plain KD)
                loss = loss + spec.eta * _binary_kl(tg, p).mean()
        if spec.prox:
            loss = loss + 0.5 * self.mu * sum(
                ((q - w0) ** 2).sum() for q, w0 in zip(self.params, self.w0))
        if spec.l1 > 0:
            loss = loss + spec.l1 * sum(q.abs().sum() for q in self.params if q.dim() > 1)
        return loss

    def _batch_loss(self, xb, yb):
        if self.cfg.multilabel:
            return self._batch_loss_ml(xb, yb)
        logits = self.net(xb)
        T = self.spec.temperature
        if self.spec.prior and self.spec.blend:
            # pLasso: fit to the blended response (y + eta*prior)/(1+eta).
            # No separate CE term -- the true label enters through the blend.
            t = self.teacher(xb)
            q = F.softmax(t / T, dim=1)
            onehot = F.one_hot(yb, logits.shape[1]).float()
            eta_s = (self.eta_vec[yb] if self.eta_vec is not None
                     else torch.full((len(yb),), self.spec.eta, device=logits.device))
            eta_s = eta_s.unsqueeze(1)
            target = (onehot + eta_s * q) / (1.0 + eta_s)
            loss = -(target * F.log_softmax(logits / T, dim=1)).sum(1).mean() * (T * T)
        else:
            loss = F.cross_entropy(logits, yb)
            if self.spec.prior:
                t = self.teacher(xb)
                if self.spec.not_true:
                    reg = _not_true_distill_per_sample(logits, t, yb, T)
                else:
                    reg = _soft_distill_per_sample(logits, t, T)
                w = self.eta_vec[yb] if self.eta_vec is not None else self.spec.eta
                loss = loss + (w * reg).mean()
        if self.spec.prox:
            prox = sum(((p - w0) ** 2).sum() for p, w0 in zip(self.params, self.w0))
            loss = loss + 0.5 * self.mu * prox
        if self.spec.l1 > 0:                       # pLasso L1 on weight matrices
            loss = loss + self.spec.l1 * sum(p.abs().sum()
                                             for p in self.params if p.dim() > 1)
        return loss

    def train(self, mu=0.0, steps=None, c_global=None, c_local=None, lr=None):
        """Run local training. Returns (new_state, new_c_local, control_used)."""
        self.mu = mu
        steps = self.cfg.local_steps if steps is None else steps
        lr = self.cfg.lr if lr is None else lr
        rho = self.spec.sam_rho                       # SAM only for methods that enable it
        if rho > 0 and self.cfg.sam_rho is not None:  # per-task override (does NOT turn SAM on)
            rho = self.cfg.sam_rho
        # local momentum destabilizes SCAFFOLD's control-variate correction
        mom = 0.0 if c_global is not None else self.cfg.momentum
        vel = [torch.zeros_like(p) for p in self.params] if mom > 0 else None
        n_inner = 0

        for _ in range(steps):
            perm = torch.randperm(self.num, device=self.device)
            for h in range(self.num // self.cfg.batchsize):
                idx = perm[h * self.cfg.batchsize:(h + 1) * self.cfg.batchsize]
                xb, yb = self.x[idx], self.y[idx]
                if self.cfg.augment and xb.dim() == 4:
                    xb = _augment_batch(xb)

                if rho > 0:                       # SAM: ascend then re-grad
                    g = torch.autograd.grad(self._batch_loss(xb, yb), self.params)
                    norm = torch.sqrt(sum((gi ** 2).sum() for gi in g)) + 1e-12
                    eps = [rho * gi / norm for gi in g]
                    with torch.no_grad():
                        for p, e in zip(self.params, eps):
                            p.add_(e)
                    grads = torch.autograd.grad(self._batch_loss(xb, yb), self.params)
                    with torch.no_grad():
                        for p, e in zip(self.params, eps):
                            p.sub_(e)             # restore
                else:
                    grads = torch.autograd.grad(self._batch_loss(xb, yb), self.params)

                with torch.no_grad():
                    for j, (p, g) in enumerate(zip(self.params, grads)):
                        step = g
                        if c_global is not None:  # SCAFFOLD drift correction
                            step = step + (c_global[j] - c_local[j])
                        if vel is not None:       # local SGD momentum
                            vel[j].mul_(mom).add_(step)
                            step = vel[j]
                        p.sub_(lr * step)
                n_inner += 1

        new_state = _clone_state(self.net.state_dict())

        new_c_local = None
        if c_global is not None:                  # SCAFFOLD Option-II control update
            new_c_local = []
            coef = 1.0 / (max(n_inner, 1) * max(lr, 1e-8))
            for j, (p, w0) in enumerate(zip(self.params, self.w0)):
                new_c_local.append(c_local[j] - c_global[j] + coef * (w0 - p.detach()))
        return new_state, new_c_local, n_inner


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(state, client: Client, cfg: TrainConfig):
    """Top-1 accuracy (single-label) or macro mAP (multi-label)."""
    dev = torch.device(cfg.device)
    net = build_model(cfg.model, **cfg.model_kw).to(dev)
    net.load_state_dict(state)
    net.eval()
    n = client.number
    if cfg.multilabel:
        probs, tgts = [], []
        for i in range(0, n, cfg.eval_batch):
            xb = client.x[i:i + cfg.eval_batch].to(dev).float()
            probs.append(torch.sigmoid(net(xb)).cpu())
            tgts.append(client.y[i:i + cfg.eval_batch].cpu())
        import numpy as _np
        from sklearn.metrics import average_precision_score
        P = torch.nan_to_num(torch.cat(probs), nan=0.0).numpy()  # divergence-safe
        Y = torch.cat(tgts).numpy()
        aps = []
        for c in range(Y.shape[1]):
            if Y[:, c].sum() > 0:               # AP undefined for empty class
                aps.append(average_precision_score(Y[:, c], P[:, c]))
        return float(_np.mean(aps)) if aps else 0.0
    correct = 0
    for i in range(0, n, cfg.eval_batch):
        xb = client.x[i:i + cfg.eval_batch].to(dev).float()
        yb = client.y[i:i + cfg.eval_batch].to(dev).long()
        correct += (net(xb).argmax(1) == yb).sum().item()
    return correct / n


@torch.no_grad()
def global_loss(state, client: Client, cfg: TrainConfig):
    dev = torch.device(cfg.device)
    net = build_model(cfg.model, **cfg.model_kw).to(dev)
    net.load_state_dict(state)
    net.eval()
    tot, n = 0.0, client.number
    for i in range(0, n, cfg.eval_batch):
        xb = client.x[i:i + cfg.eval_batch].to(dev).float()
        if cfg.multilabel:
            yb = client.y[i:i + cfg.eval_batch].to(dev).float()
            tot += F.binary_cross_entropy_with_logits(net(xb), yb, reduction="sum").item()
            continue
        yb = client.y[i:i + cfg.eval_batch].to(dev).long()
        tot += F.cross_entropy(net(xb), yb, reduction="sum").item()
    return tot / n


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _select(n_clients, select_clients, straggler_rate, round_idx):
    rng = np.random.default_rng(round_idx)
    sel = rng.choice(n_clients, select_clients, replace=False)
    n_active = round(select_clients * (1 - straggler_rate))
    active = set(rng.choice(sel, n_active, replace=False).tolist())
    return [(int(i), int(i) in active) for i in sel], rng


@dataclass
class _Adaptive:
    """FedProx dynamic coefficient: grow on regress, shrink when stable."""
    value: float = 0.0
    step: float = 0.1
    patience: int = 5
    _good: int = field(default=0)

    def update(self, prev_loss, cur_loss):
        if prev_loss >= cur_loss:
            self._good += 1
        else:
            self.value += self.step
            self._good = 0
        if self._good >= self.patience and self.value > 0:
            self.value -= self.step
            self._good = 0


def _new_history():
    return {"loss": [], "train_acc": [], "test_acc": [], "coef": []}


def _init_global_state(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    net = build_model(cfg.model, **cfg.model_kw)
    return _clone_state(net.state_dict())


def _round_lr(cfg: TrainConfig, r, rounds):
    if cfg.lr_schedule == "cosine":
        eta_min = 0.01 * cfg.lr            # floor: never exactly 0 (breaks SCAFFOLD)
        cos = 0.5 * (1 + math.cos(math.pi * r / max(rounds - 1, 1)))
        return eta_min + (cfg.lr - eta_min) * cos
    return cfg.lr


def _head_key(state):
    for k in ("head.weight", "linear.weight", "classifier.3.weight",
              "net.fc.weight", "fc.weight"):
        if k in state:
            return k
    cand = [k for k, v in state.items() if k.endswith(".weight") and v.dim() == 2]
    return cand[-1] if cand else None


@torch.no_grad()
def _flag_head(states, part_idx, client_list, fedavg_state):
    """FLAG: replace the FedAvg classifier head with a label-adaptive aggregation---
    each class row is averaged across clients weighted by their positive counts for
    that class (so clients that actually hold a class drive its classifier)."""
    key = _head_key(fedavg_state)
    if key is None:
        return fedavg_state
    dev = fedavg_state[key].device
    C = fedavg_state[key].shape[0]
    counts = torch.stack([client_list[ci].y.float().reshape(-1, C).sum(0).to(dev)
                          for ci in part_idx])          # (P, C) per-class positives
    w = counts / counts.sum(0, keepdim=True).clamp_min(1e-8)   # per-class weights
    empty = counts.sum(0) == 0                          # classes absent from all -> uniform
    if empty.any():
        w[:, empty] = 1.0 / len(part_idx)
    out = dict(fedavg_state)
    W = torch.zeros_like(fedavg_state[key])
    for p, sd in enumerate(states):
        W += w[p].unsqueeze(1) * sd[key].to(dev)
    out[key] = W
    bkey = key[:-len("weight")] + "bias"
    if bkey in fedavg_state:
        B = torch.zeros_like(fedavg_state[bkey])
        for p, sd in enumerate(states):
            B += w[p] * sd[bkey].to(dev)
        out[bkey] = B
    return out


@torch.no_grad()
def _spreadout(state, strength, steps=10, lr=0.1):
    """FedAwS: push the classifier's per-class weight vectors apart (server-side)."""
    key = next((k for k in ("head.weight", "linear.weight",
                            "classifier.3.weight", "net.fc.weight", "fc.weight")
                if k in state), None)
    if key is None:                              # fall back to the last 2D weight tensor
        cand = [k for k, v in state.items() if k.endswith(".weight") and v.dim() == 2]
        key = cand[-1] if cand else None
    if key is None:
        return
    W = state[key].clone()
    for _ in range(steps):
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = Wn @ Wn.t()
        sim.fill_diagonal_(0.0)
        # gradient of sum sim^2 w.r.t. W (push rows apart)
        grad = 4.0 * (sim @ Wn) / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        W = W - lr * strength * grad
    state[key] = W


# --------------------------------------------------------------------------- #
# Unified driver
# --------------------------------------------------------------------------- #
def federated(spec: MethodSpec, rounds, client_list, global_client, test_client,
              cfg: TrainConfig, init_state=None, on_round=None):
    dev = torch.device(cfg.device)
    src = init_state if init_state is not None else _init_global_state(cfg)
    state = {k: v.to(dev) for k, v in src.items()}     # resume from checkpoint if given
    hist = _new_history()
    mu = _Adaptive(value=spec.mu0) if (spec.prox and spec.adaptive_mu) else None

    # SCAFFOLD control variates (one per trainable parameter tensor)
    c_global = c_locals = None
    if spec.scaffold:
        probe = build_model(cfg.model, **cfg.model_kw)
        trainable = [p for p in probe.parameters() if p.requires_grad]
        c_global = [torch.zeros_like(p).to(dev) for p in trainable]
        c_locals = {i: [torch.zeros_like(p).to(dev) for p in trainable]
                    for i in range(len(client_list))}

    for r in range(rounds):
        cur_loss = global_loss(state, global_client, cfg)
        hist["loss"].append(cur_loss)
        hist["train_acc"].append(evaluate(state, global_client, cfg))
        hist["test_acc"].append(evaluate(state, test_client, cfg))
        # adaptive mu uses the _Adaptive schedule; otherwise a FIXED mu = spec.mu0
        mu_val = mu.value if mu else (spec.mu0 if spec.prox else 0.0)
        hist["coef"].append(mu_val if spec.prox else spec.eta)
        if on_round is not None:                  # diagnostics hook (no-op by default)
            on_round(r, state)

        roster, rng = _select(len(client_list), cfg.select_clients,
                              cfg.straggler_rate, r)
        states, weights, part_idx = [], [], []
        delta_c_sum = None
        for ci, is_active in roster:
            if not is_active and spec.excludes_stragglers:
                continue
            steps = (cfg.local_steps if is_active
                     else int(rng.integers(1, cfg.local_steps)))
            trainer = LocalTrainer(client_list[ci], state, spec, cfg)
            cg = c_global if spec.scaffold else None
            cl = c_locals[ci] if spec.scaffold else None
            new_state, new_cl, _ = trainer.train(mu=mu_val, steps=steps,
                                                 c_global=cg, c_local=cl,
                                                 lr=_round_lr(cfg, r, rounds))
            states.append(new_state)
            weights.append(client_list[ci].number)
            part_idx.append(ci)
            if spec.scaffold:
                if delta_c_sum is None:
                    delta_c_sum = [torch.zeros_like(p) for p in new_cl]
                for j in range(len(new_cl)):
                    delta_c_sum[j] += (new_cl[j] - c_locals[ci][j])
                c_locals[ci] = new_cl

        if states:
            state = _avg_states(states, weights)
            if spec.flag:                          # FLAG: label-adaptive head aggregation
                state = _flag_head(states, part_idx, client_list, state)
            if spec.spreadout > 0:                 # FedAwS server-side step
                _spreadout(state, spec.spreadout)
            if spec.scaffold and delta_c_sum is not None:
                m = len(client_list)
                for j in range(len(c_global)):
                    c_global[j] += delta_c_sum[j] / m

        if r > 0 and mu:
            mu.update(hist["loss"][r - 1], cur_loss)

    # final evaluation of the model after the last aggregation
    hist["loss"].append(global_loss(state, global_client, cfg))
    hist["train_acc"].append(evaluate(state, global_client, cfg))
    hist["test_acc"].append(evaluate(state, test_client, cfg))
    hist["coef"].append(hist["coef"][-1] if hist["coef"] else 0.0)

    return state, hist


# --------------------------------------------------------------------------- #
# Thin named wrappers (back-compatible API)
# --------------------------------------------------------------------------- #
def run_method(name, rounds, client_list, global_client, test_client, cfg):
    return federated(METHODS[name], rounds, client_list, global_client, test_client, cfg)


def Global(rounds, global_client, test_client, cfg: TrainConfig):
    """Centralised training: one 'client' holding all the data, no stragglers."""
    spec = MethodSpec("Global")
    single = TrainConfig(**{**cfg.__dict__, "select_clients": 1, "straggler_rate": 0.0})
    return federated(spec, rounds, [global_client], global_client, test_client, single)


def FedAvg(rounds, cl, gc, tc, cfg):     return run_method("FedAvg", rounds, cl, gc, tc, cfg)
def FedProx(rounds, cl, gc, tc, cfg):    return run_method("FedProx", rounds, cl, gc, tc, cfg)
def SCAFFOLD(rounds, cl, gc, tc, cfg):   return run_method("SCAFFOLD", rounds, cl, gc, tc, cfg)
def FedNTD(rounds, cl, gc, tc, cfg):     return run_method("FedNTD", rounds, cl, gc, tc, cfg)
def FedSAM(rounds, cl, gc, tc, cfg):     return run_method("FedSAM", rounds, cl, gc, tc, cfg)
def FedPrior(rounds, cl, gc, tc, cfg):   return run_method("FedPrior", rounds, cl, gc, tc, cfg)
