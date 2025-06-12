# rtd_lite_regularizer.py
"""RTD‑Lite regularizer for Difusco.

This module implements an RTD‑Lite loss term that can be plugged into any
training loop.  It assumes you already have:

* `dist` – a symmetric distance matrix **on the same device** as the model
  tensors; shape ``(N, N)`` for single graphs or ``(B, N, N)`` for a batch.
* `tour_edges` – list/LongTensor of edges (i, j) belonging to the ground‑truth
  tour for **each** graph in the batch.
* `logits_t` – raw edge logits (or scores) output by Difusco’s reverse step;
  shape ``(B, N, N)`` (upper‑triangle values matter).

The code follows the *“BIG‑sentinel”* scheme discussed in our design review:
– present‑in‑tour edges keep their true length ``d_ij``
– absent edges get a large constant ``BIG = big_mult * dist.max()``

No additional normalisation is performed, as the two matrices are on the same
physical scale.

References
~~~~~~~~~~
RTD‑Lite paper (Alg. 2)  → https://arxiv.org/abs/2503.11910
Difusco code structure  → https://github.com/Edward-Sun/DIFUSCO
Prim‑MST background     → https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/

Copyright (c) 2025.
"""
from __future__ import annotations

from typing import List, Tuple, Union
import torch
from torch import Tensor

__all__ = [
    "build_weights_ground_truth",
    "build_weights_prediction",
    "rtd_lite_loss",
    "rtd_lite_regularizer",
]

################################################################################
# Helper: build weight matrices
################################################################################

def build_weights_ground_truth(
    tour_edges: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
    dist: Tensor,
    big_mult: float = 10.0,
) -> Tensor:
    """Return ``W_A`` with ``d_ij`` on tour edges, BIG elsewhere.

    Parameters
    ----------
    tour_edges : list or list‑of‑lists
        – If a *single* graph: list of ``(i, j)`` tuples.
        – If a *batch*: list of lists, one per graph.
    dist : torch.Tensor
        Symmetric distance matrix. Shape ``(N, N)`` or ``(B, N, N)``.
    big_mult : float
        Multiplier for the sentinel BIG value. 10× works well in practice.
    """
    if dist.dim() == 2:  # single graph
        big = big_mult * dist.max()
        WA = torch.full_like(dist, big)
        idx = tuple(torch.tensor(tour_edges, device=dist.device).T)
        WA[idx] = dist[idx]
        WA[idx[::-1]] = dist[idx]  # ensure symmetry (j, i)
        torch.diagonal(WA).fill_(0.0)
        return WA

    # Batch case -------------------------------------------------------------
    B, N, _ = dist.shape
    big = big_mult * dist.max()
    WA = torch.full_like(dist, big)
    for b in range(B):
        edges = tour_edges[b]
        if len(edges) == 0:
            continue
        idx = tuple(torch.tensor(edges, device=dist.device).T)
        WA[b, idx[0], idx[1]] = dist[b, idx[0], idx[1]]
        WA[b, idx[1], idx[0]] = dist[b, idx[0], idx[1]]  # symmetry
        torch.diagonal(WA[b]).fill_(0.0)
    return WA


def build_weights_prediction(logits: Tensor, dist: Tensor, big_value: float) -> Tensor:
    """Return ``W_B`` according to BIG‑sentinel strategy.

    ``p = sigmoid(logits)`` maps to [0, 1].  High p ⇒ weight ~ ``d_ij``; low p ⇒
    weight ~ BIG. The operation is differentiable w.r.t logits.
    """
    p = torch.softmax(logits, dim=1)
    p = p[:,1]
    res = (1.0 - p) * big_value + p * dist

    return res

################################################################################
# Core: RTD‑Lite loss (Algorithm 2)
################################################################################


def _prim_mst_total(weights: Tensor) -> Tensor:
    """Return *scalar* total weight of the MST for **one** graph.

    Implementation: O(N²) dense Prim, but uses vectorised tensor ops so it can
    run on GPU. Suitable for N ≲ 1 000. If you need larger graphs, consider a
    sparse MST implementation.
    """
    N = weights.size(0)
    device = weights.device

    selected = torch.zeros(N, dtype=torch.bool, device=device)
    selected[0] = True  # start from vertex 0
    mst_weight = torch.tensor(0.0, device=device)

    for _ in range(N - 1):
        # Consider edges (u, v) where u is selected, v is not.
        mask = selected.unsqueeze(1) & ~selected.unsqueeze(0)
        cand_weights = weights.clone()
        cand_weights[~mask] = float("inf")
        # find minimal candidate
        min_val, idx = cand_weights.view(-1).min(dim=0)
        mst_weight += min_val
        # decode index to (u, v)
        u = idx // N
        v = idx % N
        selected[v] = True
    return mst_weight


def prim_mst_total(weights: Tensor) -> Tensor:
    """Vectorised wrapper handling batch or single graph."""
    if weights.dim() == 2:
        return _prim_mst_total(weights)
    # Batch
    B = weights.size(0)
    totals = [
        _prim_mst_total(weights[b]) for b in range(B)
    ]
    return torch.stack(totals)


@torch.jit.ignore
def rtd_lite_loss(WA: Tensor, WB: Tensor) -> Tensor:
    """Compute RTD‑Lite loss for **batch** or **single** graphs.

    Follows Alg. 2: RTDL = sum(MST(WA)) − sum(MST(min(WA, WB))).
    Differentiable w.r.t `WB` (hence logits) but **not** w.r.t `WA`.
    """
    WC = torch.minimum(WA, WB)  # element‑wise → differentiable
    mst_A = prim_mst_total(WA.detach())  # detach to freeze GT
    mst_C = prim_mst_total(WC)
    return (mst_A - mst_C).mean()  # if batch → mean per graph

################################################################################
# Wrapper: ready‑to‑use regularizer
################################################################################

class RTDLiteRegularizer(torch.nn.Module):
    """Plug‑and‑play RTD‑Lite regularizer with linear warm‑up schedule."""

    def __init__(
        self,
        big_mult: float = 10.0,
        max_coef: float = 0.15,
        warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.big_mult = big_mult
        self.max_coef = max_coef
        self.warmup_steps = warmup_steps
        self.register_buffer("_global_step", torch.tensor(1, dtype=torch.long))

    @property
    def coef(self) -> float:
        step = self._global_step.item()
        if step >= self.warmup_steps:
            return self.max_coef
        return self.max_coef * step / self.warmup_steps

    def forward(
        self,
        logits: Tensor,
        tour_edges: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
        dist: Tensor,
    ) -> Tensor:
        """Return scaled RTD‑Lite loss ready to add to main objective."""
        BIG = self.big_mult * dist.max()
        WA = build_weights_ground_truth(tour_edges, dist, self.big_mult)
        WB = build_weights_prediction(logits, dist, BIG)
        l1   = (WA - WB).abs().mean()
        # add a fallback term so gradient is non-null
        topo_loss = rtd_lite_loss(WA, WB) + 1e-3 * l1
        scaled = self.coef * topo_loss
        # update step counter ----------------------------------------------
        self._global_step += 1
        return scaled

################################################################################
# Example usage inside Difusco Trainer
################################################################################

if __name__ == "__main__":
    # dummy example for 1 graph with 5 nodes -------------------------------
    N = 5
    B = 2
    dist = torch.rand(B, N, N)
    dist = (dist + dist.transpose(1, 2)) / 2
    dist.diagonal(dim1=1, dim2=2).zero_()
    tour_edges = []
    for _ in range(B):
        edges = [(i, i + 1) for i in range(N - 1)]
        edges.append((N - 1, 0))
        tour_edges.append(edges)

    logits = torch.randn(B, 2, N, N, requires_grad=True)
    logits.retain_grad()

    reg = RTDLiteRegularizer(max_coef=0.5, big_mult=2)
    loss = reg(logits, tour_edges, dist)
    loss.backward()
    print("Topo loss:", loss.item())
    print("Gradients non‑zero:", logits.grad.abs().sum() > 0)
