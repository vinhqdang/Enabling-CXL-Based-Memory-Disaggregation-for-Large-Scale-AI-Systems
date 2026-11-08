# Resident-layer placement for weight streaming: design notes (v3)

## Problem
A model is a cyclic sequence of L units (transformer layers; optionally embedding/head units).
Unit i has compute time c_i (GPU, for the current iteration size) and transfer time
f_i = a + s_i / B over the host-to-GPU path (pinned host DRAM, or a CXL expander mapped as host
memory: the same host-mediated DMA path). A set R of units is resident in GPU memory; the
other m = L - |R| units are streamed every iteration through a ring of k equal slots.
One copy engine (transfers serial, in streaming order); one compute stream (units in order).

Event recurrences (j = global index of streamed transfers, p(j) its unit):
  X_j = max(X_{j-1}, Y_{p(j-k)}) + f_j         (link free and slot freed by compute of j-k)
  Y_n = max(Y_{n-1}, X_j if unit n is streamed) + c_n
This is a max-plus linear timed event graph; its steady period P(R,k) is the maximum
circuit mean (weight / number of iterations spanned). Every circuit through a compute node
is a tiling of w consecutive iterations by
  * single units (weight c_i), and
  * segments that start after streamed unit p(j) and end at streamed unit p(j+k+t), t >= 0,
    with weight f_{j+k} + ... + f_{j+k+t} + c_{p(j+k+t)}.
plus the pure link circuit (weight sum of f over streamed units).

## Results to prove (uniform units c, f)
* LB: P >= max(L c, m f) for every placement.
* T1 (compute-bound per unit): f <= c and k >= 2  =>  P = L c for every placement.
* T2 (balanced optimal): f > c, k >= 3, residents spread evenly (gap counts between consecutive
  streamed units differ by at most one, Bresenham)  =>  P = max(L c, m f) (optimal).
  Proof: segment gain over single units is (q-k+1)(f-c) - (k-2)c - c G with G >= q r/m - 1
  for balanced gaps; sum over a w-iteration tiling <= w max(0, m f - L c) + n_seg (2c - (k-1) f).
* T3 (contiguous exact): f > c, k >= 2: P = max(L c, m f, r c + 2c + (m-k) f).
  Penalty ratio contiguous / balanced < 2, approaches 2 when r c ~ m f and k f, c << r c.
* P4 (two slots are not enough): k = 2 can exceed the bound by up to n_seg (2c - f) for c < f < 2c.
* C5 (memory budget of M units for residents + slots): k = 3, r = M - 3, balanced is optimal
  when f > c; P* = max(L c, (L - M + 3) f).
Numerical checks (scratch, 3000 random instances): T2 no violation for k >= 3; T3 exact;
balanced optimal by brute force for L <= 12 also with k = 2.

## Non-uniform units
Exact evaluator + placement search (compute-weighted balancing, local search, brute force for
small L). Case: transformer with a large LM-head/embedding unit.

## Measurement (real GPU)
Tesla T4 on Colab, pinned host memory, PCIe Gen3 x16 (~12 GB/s, the bandwidth class of
published CXL Type-3 expanders, 10-20 GB/s). Real layer kernels (GEMMs, SDPA) with real model
shapes and random weights. Measure c_i, f_i, then measured periods for placements
(contiguous-first, contiguous-last, balanced, random), slots k in 1..4, prefill T and decode B.
Compare with the max-plus prediction from measured c_i, f_i.
