"""Write paper/tab_workloads.tex from results/summary/stream_table.csv."""
import collections
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results/summary/stream_table.csv")))
        if r["group"] == "main" and r["exp"] != "allres" and r.get("pred_cal_ms")]
LOW = ("full_B", "full_C15", "full_C20", "full_E", "full_F", "full_Sweep")
groups = collections.defaultdict(list)
for r in rows:
    head = "F512" in r["source"] or "Fdec" in r["source"]
    lock = 450 if any(s in r["source"] for s in LOW) else 570
    groups[(r["model"] + (" +head" if head else ""), r["workload"], lock)].append(r)
NAMES = {"llama3.2-3b": "3B", "llama3.2-3b +head": "3B + head", "llama3.1-8b-16L": "8B (16 layers)"}


def key(k):
    kind, size = k[1].split(":")[:2]
    return (k[0], kind != "pre", int(size))


lines = []
for k in sorted(groups, key=key):
    v = groups[k]
    fc = st.mean(float(x["f_over_c"]) for x in v)
    err = [abs(float(x["period_ms"]) / float(x["pred_cal_ms"]) - 1) * 100 for x in v]
    kind, size = k[1].split(":")[:2]
    wl = ("prefill $T=%s$" % f"{int(size):,}".replace(",", "\\,")) if kind == "pre" else "decode $B=%s$" % size
    reg = "copy-bound ($f\\ge 2c$)" if fc >= 2 else ("band ($c<f<2c$)" if fc > 1.02 else "compute-bound ($f\\le c$)")
    lines.append(f"{NAMES[k[0]]} & {wl} & {k[2]} & {fc:.2f} & {reg} & {len(v)} & {st.mean(err):.2f} & {max(err):.2f}\\\\")
tab = r"""\begin{table*}[t]
\centering
\caption{Measured workloads (Tesla T4). Clock: locked SM clock in MHz. $f/c$: mean ratio of copy to compute time per layer. Config.: streaming configurations measured (all-resident runs excluded). Error: mean and maximum absolute error of the calibrated model, in \%.}
\label{tab:workloads}
\small
\begin{tabular}{llrrlrrr}
\toprule
Model & Workload & Clock & $f/c$ & Regime & Config. & Mean err. & Max err.\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(ROOT, "paper/tab_workloads.tex"), "w").write(tab)
print("wrote paper/tab_workloads.tex", len(lines), "rows")
