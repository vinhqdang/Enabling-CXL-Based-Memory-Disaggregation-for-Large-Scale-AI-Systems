# Submission Status

Tracks the submission history of "CAMP: Content-Aware Memory Prefetching for
High-Performance CXL-Based Inference".

## History

| Date | Venue | Outcome |
|------|-------|---------|
| ~2026-Q1 | Array (Elsevier), Ms. No. ARRAY-D-26-00328 | **Rejected** after two review rounds (2026-05-09). Reviewer #1 suspected fabricated/unreliable experiments; Reviewer #2 flagged untraceable claims and an implausible perfectly-linear throughput curve. See `manuscript/response_r1.txt` for the round-1 response letter. |
| 2026-08-25 | **Journal of Systems Architecture (JSA)**, Elsevier | **Submitted.** |

## What changed before the JSA submission

- Hardened the discrete-event simulator: modeled the CXL link and GPU compute engine as genuinely shared, contended resources (`simpy.Resource`) so bandwidth/compute saturation effects emerge from simulation instead of being asserted; fixed a topological-sort bug that scrambled execution order; fixed batch-size-ignoring compute formulas.
- Regenerated every reported number from the hardened simulator; rewrote the manuscript so every claim traces to a specific figure/JSON value.
- Ran an independent 5-seat simulated peer-review panel (Journal-Fit, Methodology, Domain, Perspective, Devil's Advocate) and remediated every corroborated finding (a structural reuse-frequency tie in the ablation study, a mislabeled baseline, an arithmetic error, overclaiming language, an undisclosed workload-iteration decision).
- Broadened the evaluation for an engineering-practice venue: real-model validation on `distilgpt2` with real GPU-timed compute, model-scale sensitivity (12-80 layers), hyperparameter sensitivity ($\gamma$, bandwidth, tenant count), and a genuine open-loop/Poisson-arrival continuous-batching evaluation replacing a synchronized closed-loop stress test.
- Reformatted for JSA: numbered citation style, single-author byline (Quang-Vinh Dang, British University Vietnam), abstract trimmed to fit JSA's submission-form 200-word limit, added a Highlights file, added the mandatory Generative AI declaration.

## Full technical history

See project memory `project_camp_array_rejection.md` (Claude Code session memory) for the complete bug-by-bug and reviewer-finding-by-finding remediation log.
