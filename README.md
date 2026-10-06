# Streaming LLM weights from CXL memory: regimes, placement and pipeline-aware pinning

Code, data and manuscript for the paper *Streaming LLM Weights from CXL Memory: Regimes, Placement
and Pipeline-Aware Pinning* (Quang-Vinh Dang, British University Vietnam).

The paper studies inference of models whose weights exceed GPU memory when part of the weights
lives in a CXL Type-3 memory expander and is streamed into HBM by host-initiated DMA. It proposes a
runtime policy (CAMP: work-conserving prefetch, stall-cost-aware pinning, online cost model),
and evaluates it with an event-driven simulator whose link and overlap logic is checked on a real GPU.
The main findings are a regime map and lower bound, the importance of interleaving the resident
layers, and a policy that ties a tuned FlexGen-style per-layer split on dense models and is ahead on
non-uniform ones (the paper reports the negative results too).

## Layout

| Path | Content |
|------|---------|
| `camp_sim/` | Simulator: `hw.py` (GPU and host-mediated link model), `cost.py` (roofline ground truth and the runtime cost model), `models.py` (Llama-2/3, Gemma-2 configs, traces), `sim.py` (event engine), `policies.py` (baselines, CAMP prefetcher, SCP pinning) |
| `experiments/` | `run_all.py` (experiments E1-E11), `validate_hw.py`, `plots.py`, `make_tables.py`, `make_numbers.py` (every number quoted in the text), `fig_architecture.py` |
| `validation/hw_validate.py` | Real-GPU micro-benchmark (copy cost, GEMM roofline, streaming loop) |
| `results/` | Raw JSON results of every experiment and the measured T4 data (`results/hw/`) |
| `tests/test_engine.py` | Closed-form checks of the event engine |
| `manuscript/` | LaTeX source, figures, generated tables, compiled `main.pdf` |
| `submission/` | Cover letter, highlights, submission checklist |
| `legacy/` | First implementation and manuscript version, superseded (not used in the paper) |

## Reproducing the results

```bash
pip install -r requirements.txt
python tests/test_engine.py                 # engine vs. closed-form pipeline times
python experiments/run_all.py               # all simulator experiments (~15 min on four cores; run the e1..e11 arguments in parallel)
python experiments/validate_hw.py           # simulator vs. measured T4 data in results/hw
python experiments/plots.py && python experiments/fig_architecture.py
python experiments/make_tables.py && python experiments/make_numbers.py
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The hardware micro-benchmark needs a CUDA GPU with PyTorch, for example
`colab run --gpu T4 --timeout 2400 validation/hw_validate.py` (output between `HWJSON_BEGIN` and
`HWJSON_END`; the measured file used in the paper is `results/hw/hw_T4_v2.json`).

## What is and is not claimed

* The CXL link is simulated. Its bandwidth and fixed costs are anchored on published device
  measurements and swept over a wide range; no CXL hardware was measured.
* The link and overlap logic (copy path, queueing, overlap of transfers with compute) was compared
  with a Tesla T4 using pinned host DRAM over PCIe as a stand-in for the CXL pool (**no CXL
  device was measured**): 30 configurations, 7.6 % mean absolute error overall,
  0.6 % in link-bound configurations, up to 25 % where compute and copy are
  comparable (cause not isolated). A closed-form expression with the same measured inputs has
  6.4 % mean error, so the check validates inputs and overlap semantics, not the extra
  machinery of the simulator.
* Compute times of the H100-class device come from a roofline model, not from measurements.

See Section 7 of the manuscript for the full list of limitations.

## License

MIT, see `LICENSE`.
