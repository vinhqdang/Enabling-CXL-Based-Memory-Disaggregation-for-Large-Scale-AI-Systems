# Submission status

Manuscript: "Streaming LLM Weights from CXL Memory: Regimes, Placement and Pipeline-Aware Pinning"
(first submitted as "CAMP: Content-Aware Memory Prefetching for High-Performance CXL-Based Inference").

| Date | Venue | Outcome |
|------|-------|---------|
| 2026-01-26 | Array (Elsevier), ARRAY-D-26-00328 | Submitted as "CAMP: Content-Aware Memory Prefetching for High-Performance CXL-Based Inference"; revised once (R1); rejected 2026-08-23. |
| 2026-08-25 | Journal of Systems Architecture (Elsevier), JSA-D-26-01708 | Rejected (decision 2026-10). Two reviewers; the points were addressed by rebuilding the study, not by a rebuttal. |
| 2026-10-06 | Journal of Parallel and Distributed Computing (Elsevier) | Submitted as a new manuscript (title: "Streaming LLM Weights from CXL Memory: Regimes, Placement and Pipeline-Aware Pinning"). Classifications chosen: Heterogeneous Computing System, Resource Allocation, Scheduling In Computing, Optimization, Scalability. Package: `submission/`. Backup venue if rejected: Microprocessors and Microsystems. |
| 2026-10-21 | JPDC, JPDC-D-26-01936 | Rejected without review (Associate Editor in Chief). Reason: results premature; the work is about CXL but no CXL was used. Editor encourages improving the evaluation to represent the actual setup and resubmitting. |
| 2026-10-21 | Venue choice | Array is excluded: it already rejected an earlier version (ARRAY-D-26-00328R1) although Elsevier's transfer list offers it. Remaining transfer options: Machine Learning with Applications (scope is ML applications, weak fit). Backup: Microprocessors and Microsystems. Not yet submitted anywhere. |
| 2026-10-22 | Microprocessors and Microsystems | Package prepared with three authors (Dang, Le, Dinh); journal scope is embedded hardware, so desk-rejection for scope is a risk. Not yet submitted. |

## What changed after the JSA reviews

* The simulator was rebuilt (`camp_sim/`): host-mediated CXL path with explicit initiator,
  bandwidth bound, per-copy and synchronisation costs and a staged variant; roofline compute
  model for prefill, decode and mixed batches; measurement noise; real model configurations from
  7B to 405B parameters.
* The link and overlap logic was checked on a real GPU (`results/hw/`, `validation/hw_validate.py`); the check is mixed and reported as such.
* The pinning algorithm was reformulated and its relation to the 0-1 knapsack stated exactly;
  baselines were strengthened (equal budget, tuned lookahead, tuned FlexGen-style split) and an attribution study and exhaustive-optimum comparison added. The stronger baselines removed the claim of a large gain over FlexGen-style placement on dense models; the paper now reports this.
* The compute-time estimate is obtained at runtime and its sensitivity is quantified.
* Related work was cut to the directly relevant material.
* After the JPDC desk rejection the manuscript was reframed as a simulation study and extended with a pool-sizing rule for several GPUs per expander (E12) and a consistency check against ITME's published weight-streaming result (E13).
