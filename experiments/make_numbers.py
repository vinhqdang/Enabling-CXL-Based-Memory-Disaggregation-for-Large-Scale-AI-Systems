"""Compute every number quoted in the manuscript text and write manuscript/numbers.tex (one macro per number)."""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(n):
    return json.load(open(os.path.join(RES, n + ".json")))


def pref(res, p):
    for k, v in res.items():
        if k.startswith(p):
            return v
    raise KeyError(p)


N = {}


def put(name, val, fmt="{:.2f}"):
    N[name] = fmt.format(val) if not isinstance(val, str) else val


def rng(prefix, xs, fmt="{:.2f}"):
    put(prefix + "Min", min(xs), fmt)
    put(prefix + "Max", max(xs), fmt)
    put(prefix + "Med", statistics.median(xs), fmt)


# ------------------------------------------------------------------ E1 regime map
e1 = load("e1_regime_map")
cell = {(c["T"], c["bw"]): c for c in e1["cells"]}
for T, nm in ((1, "One"), (64, "SixtyFour"), (256, "TwoFiftySix"), (1024, "Kilo"), (4096, "FourK"), (16384, "SixteenK"), (65536, "SixtyFiveK")):
    put("regSlow" + nm + "Eighteen", cell[(T, 18)]["camp"] / cell[(T, 18)]["ideal"], "{:.1f}" if cell[(T, 18)]["camp"] / cell[(T, 18)]["ideal"] > 3 else "{:.2f}")
    put("regSlow" + nm + "FortyFive", cell[(T, 45)]["camp"] / cell[(T, 45)]["ideal"], "{:.1f}" if cell[(T, 45)]["camp"] / cell[(T, 45)]["ideal"] > 3 else "{:.2f}")
rng("regDemandOverCamp", [c["demand"] / c["camp"] for c in e1["cells"]])
rng("regCampOverLb", [c["camp"] / c["lb"] for c in e1["cells"]], "{:.3f}")

# ------------------------------------------------------------------ E2 main
e2 = load("e2_main")
rows = []
for r in e2:
    res = r["results"]
    c2 = pref(res, "CAMP-v2")["mean"]
    rows.append(dict(r=r, c2=c2, dem=pref(res, "Demand")["mean"], fg=pref(res, "FlexGen")["mean"], hc=pref(res, "Hot/Cold")["mean"],
                     st=pref(res, "Stride")["mean"], v1=pref(res, "CAMP-v1")["mean"], sk=pref(res, "Static-k")["mean"],
                     rea=pref(res, "Reactive")["mean"], ag=pref(res, "Aggressive")["mean"], unc=pref(res, "CAMP-v2 (uncal")["mean"], ex=pref(res, "CAMP-v2 (exact")["mean"],
                     c2ci=pref(res, "CAMP-v2")["ci"]))
put("nMainRows", len(rows), "{:d}")
put("nMainConfigs", len(rows) // 2, "{:d}")
rng("demOverCamp", [x["dem"] / x["c2"] for x in rows])
rng("skOverCamp", [x["sk"] / x["c2"] for x in rows])
rng("fgOverCamp", [x["fg"] / x["c2"] for x in rows])
rng("hcOverCamp", [x["hc"] / x["c2"] for x in rows])
rng("stOverCamp", [x["st"] / x["c2"] for x in rows])
rng("vOneOverCamp", [x["v1"] / x["c2"] for x in rows])
ext = [min(x["dem"], x["sk"], x["fg"], x["hc"], x["v1"]) / x["c2"] for x in rows]
rng("extOverCamp", ext)
put("nWinExt", sum(1 for e in ext if e > 1.01), "{:d}")
put("nTieExt", sum(1 for e in ext if 0.99 <= e <= 1.01), "{:d}")
put("nLoseExt", sum(1 for e in ext if e < 0.99), "{:d}")
put("worstLossExt", (1 / min(ext) - 1) * 100, "{:.1f}")
rng("detLb", [x["r"]["det_ratio_lb"] for x in rows], "{:.3f}")
rng("noisyLb", [x["c2"] / max(x["r"]["lb"], x["r"]["ideal"]) for x in rows], "{:.3f}")
grp = {g: [x for x in rows if x["r"]["group"] == g] for g in ("main", "constrained")}
for g, xs in grp.items():
    rng(g + "DemOverCamp", [x["dem"] / x["c2"] for x in xs])
    rng(g + "ExtOverCamp", [min(x["dem"], x["sk"], x["fg"], x["hc"], x["v1"]) / x["c2"] for x in xs])
# claim-level numbers
x70 = [x for x in rows if x["r"]["model"] == "llama3-70b" and x["r"]["workload"].startswith("prefill B=4 S=4k") and x["r"]["bw"] == 18.0][0]
put("seventyPreFourkDem", x70["dem"] / x70["c2"])
put("seventyPreFourkFgMs", x70["fg"] * 1e3, "{:.0f}")
put("seventyPreFourkCampMs", x70["c2"] * 1e3, "{:.0f}")
x70 = [x for x in rows if x["r"]["model"] == "llama3-70b" and x["r"]["workload"].startswith("decode") and x["r"]["bw"] == 18.0][0]
put("seventyDecDem", x70["dem"] / x70["c2"])
put("seventyDecCampMs", x70["c2"] * 1e3, "{:.0f}")
put("seventyDecLbMs", x70["r"]["lb"] * 1e3, "{:.0f}")
x405 = [x for x in rows if x["r"]["model"] == "llama3-405b" and x["r"]["workload"].startswith("prefill") and x["r"]["bw"] == 45.0][0]
put("fourOhFiveFortyFiveHcOver", x405["hc"] / x405["c2"])
put("fourOhFiveFortyFiveFgOver", x405["fg"] / x405["c2"])
put("fourOhFiveFortyFiveV1Over", x405["v1"] / x405["c2"])
# calibration irrelevant?
put("uncalMaxPct", max(abs(x["unc"] / x["c2"] - 1) for x in rows) * 100, "{:.1f}")
put("exactMaxPct", max(abs(x["ex"] / x["c2"] - 1) for x in rows) * 100, "{:.1f}")
put("ciMaxPct", max(x["c2ci"] / x["c2"] for x in rows) * 100, "{:.1f}")
# FlexGen vs CAMP, where FlexGen wins
fgw = [x for x in rows if x["fg"] < x["c2"] * 0.99]
put("nFgWins", len(fgw), "{:d}")
put("fgWinMaxPct", max((x["c2"] / x["fg"] - 1) * 100 for x in fgw) if fgw else 0, "{:.1f}")
fgl = [x for x in rows if x["fg"] > x["c2"] * 1.01]
put("nFgLoses", len(fgl), "{:d}")
put("fgLoseMaxPct", max((x["fg"] / x["c2"] - 1) * 100 for x in fgl), "{:.0f}")
stl = [x for x in rows if x["st"] > x["c2"] * 1.01]
put("nStLoses", len(stl), "{:d}")
put("stLoseMaxPct", max((x["st"] / x["c2"] - 1) * 100 for x in rows), "{:.0f}")

put("reaOverSkMax", max(abs(x["rea"] / x["sk"] - 1) for x in rows) * 100, "{:.1f}")
put("agOverSkMax", max(abs(x["ag"] / x["sk"] - 1) for x in rows) * 100, "{:.1f}")
put("agOverSkMin", min(x["ag"] / x["sk"] for x in rows), "{:.2f}")
put("agOverSkMaxRatio", max(x["ag"] / x["sk"] for x in rows), "{:.2f}")

# ------------------------------------------------------------------ E3 pin strategies
e3 = load("e3_pin_strategies")
ratios = {}
for key, nm in (("none", "None"), ("prefix", "Prefix"), ("frequency (conv", "FreqConv"), ("frequency (equal", "FreqEq"), ("stride", "Stride"), ("knapsack", "Dp"), ("marginal", "Greedy"), ("SCP (planner", "ScpPlan"), ("SCP (plan+", "Scp")):
    ratios[nm] = [pref(r["res"], key)["mean"] / max(r["lb"], r["ideal"]) for r in e3]
for nm, xs in ratios.items():
    put("pinMean" + nm, statistics.mean(xs), "{:.3f}")
    put("pinMax" + nm, max(xs), "{:.2f}")
put("pinRandMeanOverScp", statistics.mean(r["random_draws"]["mean"] / pref(r["res"], "SCP (plan+")["mean"] for r in e3), "{:.2f}")
put("pinRandMaxOverScp", max(r["random_draws"]["mean"] / pref(r["res"], "SCP (plan+")["mean"] for r in e3), "{:.2f}")
# SCP regret vs best simple rule
reg = []
for r in e3:
    simple = min(pref(r["res"], k)["mean"] for k in ("none", "prefix", "frequency (conv", "frequency (equal", "stride", "knapsack"))
    reg.append(pref(r["res"], "SCP (plan+")["mean"] / simple - 1)
put("scpRegretMaxPct", max(reg) * 100, "{:.1f}")
put("scpRegretMedPct", statistics.median(reg) * 100, "{:.1f}")
sa = []
for r in e3:
    simple = min(pref(r["res"], k)["mean"] for k in ("none", "prefix", "frequency (conv", "frequency (equal", "stride", "knapsack"))
    sa.append(simple / pref(r["res"], "SCP (plan+")["mean"])
put("scpGainOverBestSimpleMax", (max(sa) - 1) * 100, "{:.1f}")
put("nScpBeatsSimple", sum(1 for v in sa if v > 1.01), "{:d}")
put("nPinCfg", len(e3), "{:d}")
# link utilisation
u_stride = [pref(r["res"], "stride")["util"] for r in e3]
u_freq = [pref(r["res"], "frequency (equal")["util"] for r in e3]
u_pref = [pref(r["res"], "prefix")["util"] for r in e3]
put("utilPrefixMin", min(u_pref), "{:.2f}")
put("utilFreqMin", min(u_freq), "{:.2f}")
put("utilStrideMin", min(u_stride), "{:.2f}")
gp = load("e3_optimality_gap")
put("gapScpMax", max(r["scp_gap_pct"] for r in gp), "{:.1f}")
put("gapScpMean", statistics.mean(r["scp_gap_pct"] for r in gp), "{:.2f}")
put("gapStrideMax", max(r["stride_gap_pct"] for r in gp), "{:.1f}")
put("gapFreqMax", max(r["freq_gap_pct"] for r in gp), "{:.1f}")
put("gapSimMax", max(r["sim_gap_pct"] for r in gp), "{:.1f}")
put("nGap", len(gp), "{:d}")

# ------------------------------------------------------------------ E4 eviction & heterogeneity
ev = load("e4_reuse_heterogeneity")["eviction_uniform"]
lru = [r for r in ev if r["evict"] == "lru" and r["prefetch"] == "demand"][0]
bel = [r for r in ev if r["evict"] == "belady" and r["prefetch"] == "demand"][0]
put("evictLruMs", lru["mean"] * 1e3, "{:.0f}")
put("evictBeladyMs", bel["mean"] * 1e3, "{:.0f}")
put("evictBeladyHit", bel["hit_rate"] * 100, "{:.0f}")
het = load("e4_reuse_heterogeneity")["heterogeneous"]
for c, nm in zip(het, ("Gemma", "Spec", "Enc")):
    scp = pref(c["res"], "CAMP prefetch + SCP")["mean"]
    put(f"het{nm}Dem", pref(c["res"], "demand + LRU")["mean"] / scp)
    put(f"het{nm}Lfu", pref(c["res"], "demand + LFU")["mean"] / scp)
    put(f"het{nm}Fg", pref(c["res"], "FlexGen")["mean"] / scp)
    put(f"het{nm}Stride", pref(c["res"], "CAMP prefetch + stride")["mean"] / scp)
    put(f"het{nm}Freq", pref(c["res"], "CAMP prefetch + frequency (equal")["mean"] / scp)
    put(f"het{nm}Prefix", pref(c["res"], "CAMP prefetch + prefix")["mean"] / scp)
    put(f"het{nm}Dp", pref(c["res"], "CAMP prefetch + knapsack")["mean"] / scp)
    put(f"het{nm}ScpS", scp, "{:.2f}")

# ------------------------------------------------------------------ E5 cost model
e5 = load("e5_cost_model")
for c, nm in zip(e5["cases"], ("A", "B", "C")):
    put(f"cmMape{nm}Gen", c["estimator"]["static"]["before"] * 100, "{:.1f}")
    put(f"cmMape{nm}Cal", c["estimator"]["online"]["after"] * 100, "{:.1f}")
    lat = [c["ref"][k][0] for k in ("generic", "calibrated", "exact")]
    put(f"cmSpread{nm}", (max(lat) / min(lat) - 1) * 100, "{:.1f}")
    put(f"cmNoPinOver{nm}", c["ref"]["no_pins"][0] / c["ref"]["exact"][0])
# grid: bias sweep, static estimator
for c, nm in zip(e5["cases"], ("A", "B", "C")):
    g0 = [g for g in c["grid"] if g["sigma"] == 0.0 and not g["online"]]
    best = min(g["mean"] for g in g0)
    worst = max(g["mean"] for g in g0)
    put(f"cmBiasWorst{nm}", (worst / best - 1) * 100, "{:.1f}")
    g1 = [g for g in c["grid"] if g["online"]]
    put(f"cmBiasOnlineWorst{nm}", (max(g["mean"] for g in g1) / min(g["mean"] for g in g1) - 1) * 100, "{:.1f}")
shp = {(r["evaluate"], r["probe"], r["variant"]): r for r in e5["shape"]}
put("shapeMapeGeneric", max(r["mape"] for r in e5["shape"] if r["variant"].startswith("generic")) * 100, "{:.0f}")
put("shapeMapeCross", max(r["mape"] for r in e5["shape"] if r["probe"] != r["evaluate"]) * 100, "{:.0f}")
put("shapeLatSpreadPct", max(r["mean"] / [x for x in e5["shape"] if x["evaluate"] == r["evaluate"] and x["variant"].startswith("oracle")][0]["mean"] - 1 for r in e5["shape"]) * 100, "{:.1f}")

# ------------------------------------------------------------------ E6 batching / KV
e6 = load("e6_batch_scaling")
dr = [r for r in e6["decode_rows"] if r["bw"] == 18.0 and r["feasible"]]
put("kvMaxB", max(r["B"] for r in dr), "{:d}")
put("kvInfeasB", min(r["B"] for r in e6["decode_rows"] if not r["feasible"]), "{:d}")
put("kvSlowOne", dr[0]["camp"] / dr[0]["ideal"], "{:.0f}")
put("kvSlowMax", max(r["camp"] / r["ideal"] for r in dr), "{:.0f}")
put("kvTputLast", dr[-1]["B"] / dr[-1]["camp"], "{:.0f}")
put("kvIdealTputLast", dr[-1]["B"] / dr[-1]["ideal"], "{:.0f}")
put("kvCacheLastPct", dr[-1]["cache_ratio"] * 100, "{:.0f}")
put("kvCacheFirstPct", dr[0]["cache_ratio"] * 100, "{:.0f}")
tr = [r for r in e6["rows"] if r["bw"] == 18.0]
put("tputFlat", tr[0]["T"] / tr[0]["camp"], "{:.2f}")

# ------------------------------------------------------------------ E7 scale
e7 = load("e7_model_scale")
for r in e7:
    pass
sc = {}
for r in e7:
    sc[(r["model"], r["workload"], r["bw"])] = r
def sl(m, w, bw, key="CAMP-v2"):
    r = sc[(m, w, bw)]
    return pref(r["results"], key)["mean"] / r["ideal"]
put("scaleDec405Eighteen", sl("llama3-405b", "decode B=16 ctx=2k", 18.0), "{:.0f}")
put("scaleDec405FortyFive", sl("llama3-405b", "decode B=16 ctx=2k", 45.0), "{:.0f}")
put("scaleDec405Ms", pref(sc[("llama3-405b", "decode B=16 ctx=2k", 18.0)]["results"], "CAMP-v2")["mean"], "{:.1f}")
put("scalePre405Eighteen", sl("llama3-405b", "prefill B=4 S=2k", 18.0))
put("scalePre405FortyFive", sl("llama3-405b", "prefill B=4 S=2k", 45.0))

# ------------------------------------------------------------------ E8 link
e8 = load("e8_link_sensitivity")
bwrows = [r for r in e8["bw"] if r["workload"] == "decode B=16"]
put("linkDecCamp8", pref(bwrows[0], "CAMP-v2")["mean"] if False else bwrows[0]["CAMP-v2"], "{:.2f}")
put("linkDecOverLbMax", max(r["CAMP-v2"] / max(r["lb"], r["ideal"]) for r in bwrows), "{:.3f}")
pre = [r for r in e8["bw"] if r["workload"].startswith("prefill")]
put("linkPreOverLbMax", max(r["CAMP-v2"] / max(r["lb"], r["ideal"]) for r in pre), "{:.2f}")
put("linkPreOverLbAt64", [r for r in pre if r["bw"] == 64][0]["CAMP-v2"] / max([r for r in pre if r["bw"] == 64][0]["lb"], [r for r in pre if r["bw"] == 64][0]["ideal"]), "{:.3f}")
ca = [r for r in e8["cache"]]
put("cacheOverLbMax", max(r["CAMP-v2"] / max(r["lb"], r["ideal"]) for r in ca), "{:.2f}")
staged = [r for r in e8["path"] if r["mode"] == "staged"]
direct = [r for r in e8["path"] if r["mode"] == "direct"]
pairs = [("cxl_x8_staged", "cxl_x8_measured"), ("x16_staged", "cxl_x16_expected")]
sr = [[x for x in e8["path"] if x["workload"] == r["workload"] and x["link"] == s_][0]["CAMP-v2"] / r["CAMP-v2"] for r in e8["path"] if r["mode"] == "direct" for s_, d_ in pairs if d_ == r["link"]]
put("stagedOverDirectMin", min(sr), "{:.1f}")
put("stagedOverDirectMax", max(sr), "{:.1f}")

# ------------------------------------------------------------------ E9 robustness
e9 = load("e9_robustness")
put("plannerMape", e9["planner_mape"], "{:.2f}")
put("plannerMedian", e9["planner_median"], "{:.2f}")
put("plannerMax", e9["planner_max"], "{:.1f}")
put("nPlanner", len(e9["planner"]), "{:d}")
hi = [r for r in e9["noise"] if r["jitter"] == 0.5]
for r in hi:
    k = "Dec" if r["workload"].startswith("decode") else "Pre"
    put(f"noiseHi{k}CampOverFg", r["CAMP-v2"]["mean"] / r["FlexGen-style"]["mean"], "{:.3f}")
    put(f"noiseHi{k}CampOverDem", r["CAMP-v2"]["mean"] / r["Demand+LRU"]["mean"], "{:.2f}")
lo = [r for r in e9["noise"] if r["jitter"] == 0.0]
for r in lo:
    k = "Dec" if r["workload"].startswith("decode") else "Pre"
    put(f"noiseZero{k}CampOverFg", r["CAMP-v2"]["mean"] / r["FlexGen-style"]["mean"], "{:.3f}")
base = {}
for r in e9["linkbias"]:
    if r["bw_factor"] == 1.0 and r["lat_factor"] == 1.0:
        base[r["workload"]] = r["mean"]
put("linkBiasMaxPct", max(abs(r["mean"] / base[r["workload"]] - 1) for r in e9["linkbias"]) * 100, "{:.1f}")

# ------------------------------------------------------------------ E10 attribution
e10 = load("e10_attribution")
att = {}
for r in e10:
    ks = list(r["steps"].keys())
    att[(r["model"], r["workload"], r["bw"])] = (r, ks)
def step(m, w, bw, i):
    r, ks = att[(m, w, bw)]
    return r["steps"][ks[i]]["mean"]
for key, nm in ((("llama3-70b", "prefill B=4 S=4k", 18.0), "SevenFourk"), (("llama3-70b", "prefill B=4 S=2k", 18.0), "SevenTwok"), (("llama2-13b", "prefill B=4 S=4k", 18.0), "Thirteen")):
    s0 = step(*key, 0)
    for i in range(1, 7):
        put(f"att{nm}{i}", step(*key, i) / s0)
    put(f"att{nm}Zero", s0 * 1e3, "{:.0f}")
    r, ks = att[key]
    put(f"att{nm}Fg", r["steps"][ks[7]]["mean"] / s0)
    put(f"att{nm}Six", r["steps"][ks[6]]["mean"] * 1e3, "{:.0f}")
# conventional baseline vs CAMP across all attribution rows
put("attConvOverScpMax", max(step(k[0], k[1], k[2], 0) / step(k[0], k[1], k[2], 6) for k in att), "{:.2f}")
put("attConvOverScpMin", min(step(k[0], k[1], k[2], 0) / step(k[0], k[1], k[2], 6) for k in att), "{:.2f}")
put("attEqBudgetWorseMax", max(step(k[0], k[1], k[2], 1) / step(k[0], k[1], k[2], 0) for k in att), "{:.2f}")
put("attStrideOverScpMax", max(step(k[0], k[1], k[2], 4) / step(k[0], k[1], k[2], 6) for k in att), "{:.2f}")
put("attFiveOverSixMax", max(step(k[0], k[1], k[2], 5) / step(k[0], k[1], k[2], 6) for k in att), "{:.2f}")

# ------------------------------------------------------------------ E11 alternatives
e11 = load("e11_alternatives")
def alt(m, q, w):
    return [r for r in e11 if r["model"] == m and r["quant"].startswith(q) and r["workload"].startswith(w)][0]
a = alt("llama3-70b", "FP16", "decode")
put("altSeventyFpDec", a["cxl18"] * 1e3, "{:.0f}")
a8 = alt("llama3-70b", "FP8", "decode")
put("altSeventyEightDec", a8["cxl18"] * 1e3, "{:.0f}")
put("altSeventyEightDecRatio", a["cxl18"] / a8["cxl18"], "{:.1f}")
put("altSeventyEightDecHost", a8["hostdram52"] * 1e3, "{:.0f}")
a4 = alt("llama3-70b", "INT4", "decode")
put("altSeventyFourFits", "yes" if a4["fits_one_gpu"] else "no")
a405 = alt("llama3-405b", "INT4", "decode")
put("altFourOhFiveFourDec", a405["cxl18"], "{:.1f}")
put("altFourOhFiveFourWeights", a405["weight_gb"], "{:.0f}")
put("altSeventyFpHostDec", a["hostdram52"] * 1e3, "{:.0f}")
put("altSeventyFpCxl45Dec", a["cxl45"] * 1e3, "{:.0f}")
p = alt("llama3-70b", "FP16", "prefill")
put("altSeventyFpPre", p["cxl18"] * 1e3, "{:.0f}")
put("altSeventyTwoGpuPre", p["two_gpu_latency_optimistic"] * 1e3, "{:.0f}")


# ------------------------------------------------------------------ extra: scale, link, utilisation, re-pin cost
for bw, nm in ((18.0, "Eighteen"), (45.0, "FortyFive")):
    r = sc[("llama2-13b", "decode B=16 ctx=2k", bw)]
    put(f"thirteenDec{nm}Cache", r["cache_ratio"] * 100, "{:.0f}")
    put(f"thirteenDec{nm}CampMs", pref(r["results"], "CAMP-v2")["mean"] * 1e3, "{:.0f}")
    put(f"thirteenDec{nm}FgMs", pref(r["results"], "FlexGen")["mean"] * 1e3, "{:.0f}")
    put(f"thirteenDec{nm}LbMs", r["lb"] * 1e3, "{:.0f}")
    put(f"thirteenDec{nm}DemMs", pref(r["results"], "Demand")["mean"] * 1e3, "{:.0f}")
put("thirteenDecCampOverFg", sc[("llama2-13b", "decode B=16 ctx=2k", 18.0)]["results"]["CAMP-v2"]["mean"] / pref(sc[("llama2-13b", "decode B=16 ctx=2k", 18.0)]["results"], "FlexGen")["mean"], "{:.2f}")
put("fgBelowLbMax", max(1 - pref(r["results"], "FlexGen")["mean"] / r["lb"] for r in e7 if r["lb"] > 1.5 * r["ideal"] or r["workload"].startswith("decode")) * 100, "{:.1f}")
pre8 = [r for r in e8["bw"] if r["workload"].startswith("prefill")]
hcr = [pref(r, "Hot/Cold") / r["CAMP-v2"] for r in pre8]
fgr = [pref(r, "FlexGen") / r["CAMP-v2"] for r in pre8]
put("linkPreHcOverMin", min(hcr), "{:.2f}")
put("linkPreHcOverMax", max(hcr), "{:.2f}")
put("linkPreFgOverMin", min(fgr), "{:.3f}")
put("linkPreFgOverMax", max(fgr), "{:.3f}")
dec8 = [r for r in e8["bw"] if r["workload"] == "decode B=16"]
put("linkDecHcOverMax", max(pref(r, "Hot/Cold") / r["CAMP-v2"] for r in dec8), "{:.3f}")
put("linkDecFgOverMax", max(pref(r, "FlexGen") / r["CAMP-v2"] for r in dec8), "{:.3f}")
cp = [r for r in e8["cache"] if r["workload"].startswith("prefill")]
put("cachePreHcOverLbMax", max(pref(r, "Hot/Cold") / max(r["lb"], r["ideal"]) for r in cp), "{:.2f}")
put("cacheFgOverLbMax", max(pref(r, "FlexGen") / max(r["lb"], r["ideal"]) for r in e8["cache"]), "{:.3f}")
r = [x for x in e3 if x["model"] == "llama2-13b" and x["workload"] == "prefill B=4 S=4k" and x["bw"] == 18.0][0]
for key, nm in (("prefix", "Prefix"), ("frequency (equal", "Freq"), ("stride", "Stride"), ("SCP (plan+", "Scp"), ("knapsack", "Dp")):
    put(f"utilThirteenPre{nm}", pref(r["res"], key)["util"] * 100, "{:.0f}")
    put(f"msThirteenPre{nm}", pref(r["res"], key)["mean"] * 1e3, "{:.0f}")
put("msThirteenPreNone", pref(r["res"], "none")["mean"] * 1e3, "{:.0f}")
# re-pin cost and break-even (one-off move of the pinned bytes over the link)
for mn, wl, tag in (("llama3-70b", "decode B=16 ctx=2k", "SeventyDec"), ("llama3-70b", "prefill B=4 S=2k", "SeventyPre"), ("llama2-13b", "decode B=16 ctx=2k", "ThirteenDec")):
    row = [x for x in e2 if x["model"] == mn and x["workload"] == wl and x["bw"] == 18.0][0]
    gb = row["info"]["pin_gb"]
    rp = gb / 18.0
    p3 = [x for x in e3 if x["model"] == mn and x["workload"] == wl.replace(" ctx=2k", "") and x["bw"] == 18.0][0]
    gain = pref(p3["res"], "none")["mean"] - pref(row["results"], "CAMP-v2")["mean"]
    put(f"repin{tag}Gb", gb, "{:.0f}")
    put(f"repin{tag}S", rp, "{:.1f}")
    put(f"repin{tag}Iters", rp / gain, "{:.1f}")

# ------------------------------------------------------------------ hardware
hv = load("hw_validation")
put("hwPinnedBw", hv["pinned_fit"]["bw_GBps"], "{:.1f}")
put("hwPageableBw", hv["pageable_fit"]["bw_GBps"], "{:.1f}")
put("hwRatio", hv["pageable_fit"]["bw_GBps"] / hv["pinned_fit"]["bw_GBps"], "{:.2f}")
put("hwLatUs", hv["pinned_fit"]["latency_us"], "{:.1f}")
put("hwMape", hv["mape"], "{:.1f}")
put("hwMedian", hv["median"], "{:.1f}")
put("hwMax", hv["max"], "{:.0f}")
put("hwCfMape", hv["cf_mape"], "{:.1f}")
put("hwCfMax", hv["cf_max"], "{:.0f}")
put("hwPinnedMape", hv["pinned_mape"], "{:.1f}")
put("hwUnpinnedMape", hv["unpinned_mape"], "{:.1f}")
put("hwN", len(hv["rows"]), "{:d}")
lb = [r for r in hv["rows"] if r["depth"] > 0 and r["n_pinned"] == 0 and r["T"] <= 1024]
put("hwLinkBoundMape", statistics.mean(abs(r["err_pct"]) for r in lb), "{:.1f}")
put("hwLinkBoundMax", max(abs(r["err_pct"]) for r in lb), "{:.1f}")
put("hwStaging", hv["implied_staging_gbs"], "{:.1f}")
dp = [r for r in hv["rows"] if r["depth"] == 0]
put("hwDemandMax", max(abs(r["err_pct"]) for r in dp), "{:.0f}")
put("hwDemandSign", "over" if statistics.mean(r["err_pct"] for r in dp) > 0 else "under")
big = [r for r in hv["rows"] if r["T"] >= 2048 and r["depth"] > 0]
put("hwBigMean", statistics.mean(r["err_pct"] for r in big), "{:+.0f}")

out = ["% generated by experiments/numbers.py -- do not edit"]
DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
for k, v in sorted(N.items()):
    name = "n" + k[0].upper() + k[1:]
    name = "".join(DIG.get(ch, ch) for ch in name)
    out.append(f"\\newcommand{{\\{name}}}{{{v}}}")
open(os.path.join(ROOT, "manuscript", "numbers.tex"), "w").write("\n".join(out) + "\n")
json.dump(N, open(os.path.join(RES, "numbers.json"), "w"), indent=1)
print(len(N), "numbers")
