# Submission status

Manuscript: "CAMP: Graph-Aware Prefetching and Pinning for LLM Inference over Host-Mediated CXL
Memory" (first submitted as "CAMP: Content-Aware Memory Prefetching for High-Performance
CXL-Based Inference").

| Date | Venue | Outcome |
|------|-------|---------|
| 2026-Q1 | Array (Elsevier), ARRAY-D-26-00328 | Rejected after two review rounds (2026-05-09). |
| 2026-08-25 | Journal of Systems Architecture (Elsevier), JSA-D-26-01708 | Rejected (decision 2026-10). Two reviewers; comments are answered in `submission/response_to_reviewers.pdf`. |
| 2026-10 | Journal of Parallel and Distributed Computing (Elsevier) | Revised manuscript prepared for submission through the Elsevier transfer offer; see `submission/README.md`. |

## What changed after the JSA reviews

* The simulator was rebuilt (`camp_sim/`): host-mediated CXL path with explicit initiator,
  bandwidth bound, per-copy and synchronisation costs and a staged variant; roofline compute
  model for prefill, decode and mixed batches; measurement noise; real model configurations from
  7B to 405B parameters.
* The pipeline logic was validated on a real GPU (`results/hw/`, `validation/hw_validate.py`).
* The pinning algorithm was reformulated and its relation to the 0-1 knapsack stated exactly;
  frequency/LFU/hot-cold baselines and exhaustive-optimum comparisons were added.
* The compute-time estimate is obtained at runtime and its sensitivity is quantified.
* Related work was cut to the directly relevant material.
