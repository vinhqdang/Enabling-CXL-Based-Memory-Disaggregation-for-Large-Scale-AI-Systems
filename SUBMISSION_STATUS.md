# Submission status

Manuscript: "Streaming LLM Weights from CXL Memory: Regimes, Placement and Pipeline-Aware Pinning"
(first submitted as "CAMP: Content-Aware Memory Prefetching for High-Performance CXL-Based Inference").

| Date | Venue | Outcome |
|------|-------|---------|
| 2026-Q1 | Array (Elsevier), ARRAY-D-26-00328 | Rejected after two review rounds (2026-05-09). |
| 2026-08-25 | Journal of Systems Architecture (Elsevier), JSA-D-26-01708 | Rejected (decision 2026-10). Two reviewers; the points were addressed by rebuilding the study, not by a rebuttal. |
| 2026-10-06 | Journal of Parallel and Distributed Computing (Elsevier) | Submitted as a new manuscript (title: "Streaming LLM Weights from CXL Memory: Regimes, Placement and Pipeline-Aware Pinning"). Classifications chosen: Heterogeneous Computing System, Resource Allocation, Scheduling In Computing, Optimization, Scalability. Package: `submission/`. Backup venue if rejected: Microprocessors and Microsystems. |
| 2026-10-21 | JPDC, JPDC-D-26-01936 | Rejected without review (Associate Editor in Chief). Reason: results premature; the work is about CXL but no CXL was used. Editor encourages improving the evaluation to represent the actual setup and resubmitting. |
| 2026-10-21 | Array (Elsevier transfer offer, guaranteed peer review) | Prepared, not yet submitted. Manuscript reframed as a simulation study (title, abstract, new "Nature of the evidence" paragraph, "Why simulation" note); cover letter, README and title/abstract file updated in `submission/`. Next step: author selects Array on the transfer page and uploads `submission/` files. Backup: Machine Learning with Applications. |

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
| 2026-10-21 | Array (prepared) | Added after a literature update (LIA, ITME, TRACE, DAK): pool-sizing rule for several GPUs per expander (e12) and a consistency check against ITME's published weight-streaming result (e13). Package in `submission/` regenerated. |
