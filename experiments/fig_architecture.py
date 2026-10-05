import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.4, 4.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

def box(x, y, w, h, text, fc="#f2f2f2", fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(p, q, text="", color="#222", style="<->", ty=1.0, ls="-", fs=7.8, tx=0.0, va="bottom"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.4, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha="center", va=va, fontsize=fs, color=color)

ax.add_patch(FancyBboxPatch((2, 3), 58, 44, boxstyle="round,pad=0.3,rounding_size=1.4", fc="#fbfbf2", ec="#888", lw=1.1, ls="--"))
ax.text(4, 5.2, "Host server", fontsize=9.5, fontweight="bold", va="center")
ax.add_patch(FancyBboxPatch((72, 3), 26, 44, boxstyle="round,pad=0.3,rounding_size=1.4", fc="#f4faf4", ec="#6a6", lw=1.1))
ax.text(74, 5.2, "GPU (PCIe Gen5 x16)", fontsize=9.5, fontweight="bold", va="center")

# top row
box(4, 31, 16, 9, "Host DRAM\n(bounce bufs,\n\"staged\" mode)", fc="#ececec")
box(31, 31, 25, 9, "CPU + CAMP runtime\n(cost model, pin plan,\nprefetch controller)", fc="#dbe9f6")
box(75, 31, 21, 9, "HBM 80 GB\npinned + streamed\nlayers, KV cache", fc="#d5ecd5", bold=True)
# bottom row
box(4, 12, 16, 10, "CXL Type-3\nmemory expander\n(DDR5)\nweight pool", fc="#f9dcc4", bold=True, fs=8)
box(31, 12, 25, 10, "Root complex\n(host bridge: PCIe / CXL)", fc="#e8e8e8")
box(75, 12, 21, 10, "Copy engine (DMA)\n+ SMs (kernels)", fc="#e6f3e6")

arrow((20, 17), (31, 17), "CXL.mem\n10-45 GB/s", color="#b5651d", ty=0.8, fs=7.2)
arrow((56, 17), (75, 17), "PCIe DMA\ncudaMemcpyAsync", color="#2a7", style="->", ty=0.8)
arrow((43.5, 31), (43.5, 22), "load/store\n(NUMA node)", tx=7.5, ty=-1.0, va="center")
arrow((31, 35.5), (20, 35.5), "staging\ncopy", ls=":", color="#777", ty=0.8, fs=7.2)
arrow((56, 35.5), (75, 35.5), "launch copies/kernels,\nevents", color="#335", style="->", ty=0.8)
arrow((85.5, 22), (85.5, 31), "", style="->", color="#2a7")
ax.text(50, 0.4, "The GPU never addresses the Type-3 device: the host initiates every transfer, and the end-to-end rate is "
        "bounded by min(CXL read bandwidth, PCIe bandwidth).", ha="center", fontsize=7.6, style="italic")
plt.savefig("manuscript/camp_architecture.png", dpi=220, bbox_inches="tight")
