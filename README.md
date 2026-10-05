# CAMP: graph-aware prefetching and pinning for LLM inference over host-mediated CXL memory

Code, data and manuscript for the paper *CAMP: Graph-Aware Prefetching and Pinning for LLM
Inference over Host-Mediated CXL Memory* (Quang-Vinh Dang, British University Vietnam).

The paper studies inference of models whose weights exceed GPU memory when part of the weights
lives in a CXL Type-3 memory expander and is streamed into HBM by host-initiated DMA. It proposes a
runtime policy (online-calibrated cost model, work-conserving prefetch, stall-cost-aware pinning)
and evaluates it with an event-driven simulator whose pipeline logic is validated on a real GPU.

## Layout

| Path | Content |
|------|---------|
| `camp_sim/` | Simulator: `hw.py` (GPU and host-mediated link model), `cost.py` (roofline ground truth and the runtime cost model), `models.py` (Llama-2/3, Gemma-2 configs, traces), `sim.py` (event engine), `policies.py` (baselines, CAMP prefetcher, SCP pinning) |
| `experiments/` | `run_all.py` (experiments E1-E9), `validate_hw.py`, `plots.py`, `make_tables.py`, `fig_architecture.py` |
| `validation/hw_validate.py` | Real-GPU micro-benchmark (copy cost, GEMM roofline, streaming loop) |
| `results/` | Raw JSON results of every experiment and the measured T4 data (`results/hw/`) |
| `tests/test_engine.py` | Closed-form checks of the event engine |
| `manuscript/` | LaTeX source, figures, generated tables, compiled `main.pdf` |
| `submission/` | Cover letter, response to reviewers, highlights |
| `legacy/` | First implementation and manuscript version, superseded (not used in the paper) |

## Reproducing the results

```bash
pip install -r requirements.txt
python tests/test_engine.py                 # engine vs. closed-form pipeline times
python experiments/run_all.py               # all simulator experiments (~5 min on one core)
python experiments/validate_hw.py           # simulator vs. measured T4 data in results/hw
python experiments/plots.py && python experiments/fig_architecture.py
python experiments/make_tables.py
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The hardware micro-benchmark needs a CUDA GPU with PyTorch, for example
`colab run --gpu T4 --timeout 2400 validation/hw_validate.py` (output between `HWJSON_BEGIN` and
`HWJSON_END`; the measured file used in the paper is `results/hw/hw_T4_v2.json`).

## What is and is not claimed

* The CXL link is simulated. Its bandwidth and fixed costs are anchored on published device
  measurements and swept over a wide range; no CXL hardware was measured.
* The pipeline logic (copy path, queueing, overlap of transfers with compute) was compared with a
  Tesla T4 using pinned host memory as a stand-in for the CXL pool: 30 configurations, 7.6 % mean
  absolute error, 4.0 % in link-bound configurations, up to 25 % where GPU clock/power effects
  matter (not modelled).
* Compute times of the H100-class device come from a roofline model, not from measurements.

See Section 7 of the manuscript for the full list of limitations.

## License

MIT, see `LICENSE`.
