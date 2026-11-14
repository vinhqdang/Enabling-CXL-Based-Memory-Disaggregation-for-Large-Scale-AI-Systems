"""Build the IEEE Access version: one .tex file (inputs and numbers inlined) in the ieeeaccess class.

Usage: python3 paper/make_ieee.py [outdir]   (default: submission_ieee/)
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "submission_ieee")
BUILD = os.path.join(OUT, "build")
CLS = os.path.join(HERE, "ieee")

src = open(os.path.join(HERE, "main.tex")).read()
for _ in range(3):
    src = re.sub(r"\\input\{([^}]+\.tex)\}[^\n]*", lambda m: open(os.path.join(HERE, m.group(1))).read().rstrip("\n"), src)
assert "\\input{" not in src

# ---------------------------------------------------------------- body
body = src[src.index("\\section{Introduction}"):src.index("\\section*{CRediT")]

# abbreviations defined at first use in the text (IEEE Access requirement)
FIRST = [
    ("A large language model whose weights do not fit in GPU memory",
     "A large language model (LLM) whose weights do not fit in graphics processing unit (GPU) memory"),
    ("page-locked DRAM", "page-locked dynamic random-access memory (DRAM)"),
    ("host-initiated DMA path", "host-initiated direct memory access (DMA) path"),
    ("about 12~GB/s for PCIe Gen3", "about 12~GB/s for Peripheral Component Interconnect Express (PCIe) Gen3"),
    ("cache over GPU, CPU and disk", "cache over GPU, central processing unit (CPU) and disk"),
    ("and overlaps I/O with computation", "and overlaps input/output (I/O) with computation"),
    ("between GPU, CPU and NVMe memory", "between GPU, CPU and non-volatile memory express (NVMe) memory"),
    ("40 streaming multiprocessors", "40 streaming multiprocessors (SMs)"),
    ("and CUDA \\nCudaVersion", "and the Compute Unified Device Architecture (CUDA) \\nCudaVersion"),
    ("with random FP16 weights", "with random 16-bit floating-point (FP16) weights"),
]
for a, b in FIRST:
    assert body.count(a) >= 1, a
    body = body.replace(a, b, 1)
# check that CPU/NVMe are not used before their definition
for ab in ("CPU", "NVMe", "PCIe", "DRAM", "DMA", "GPU", "SMs", "FP16", "CUDA", "I/O"):
    d = body.find("(" + ab + ")")
    assert d >= 0 and body.find(ab) == d + 1, (ab, d, body.find(ab), body[body.find(ab) - 80:body.find(ab) + 10])
body = body.replace("Hugging Face Accelerate \\texttt", "Hugging Face Accelerate \\texttt")
body = body.replace("HF Accelerate", "Hugging Face Accelerate")
# equation too wide for the IEEE column
GAIN = "\\mathrm{gain} = (t+1)f + c - c\\,(q+G) = (q-k+1)(f-c) - (k-2)\\,c - c\\,G. \\label{eq:gain}\n\\end{equation}"
assert GAIN in body
body = body.replace("\\begin{equation}\n" + GAIN,
    "\\begin{align}\n\\mathrm{gain} &= (t+1)f + c - c\\,(q+G) \\nonumber\\\\\n &= (q-k+1)(f-c) - (k-2)\\,c - c\\,G. \\label{eq:gain}\n\\end{align}")
W1 = "\\text{weight} &\\le vLc+(f-c)(vm-1)-N\\bigl((k-1)f-c\\bigr)-c\\,r\\,(v-1)\\\\\n&\\le vLc+(f-c)(vm-1)-\\bigl((k-1)f-c\\bigr)-c\\,r\\,(v-1)\\\\"
assert W1 in body
body = body.replace(W1, "\\text{weight} &\\le vLc+(f-c)(vm-1)\\\\\n&\\quad-N\\bigl((k-1)f-c\\bigr)-c\\,r\\,(v-1)\\\\\n&\\le vLc+(f-c)(vm-1)\\\\\n&\\quad-\\bigl((k-1)f-c\\bigr)-c\\,r\\,(v-1)\\\\")
# first paragraph starts with a drop cap
body = body.replace("\\label{sec:intro}\n\nA large", "\\label{sec:intro}\n\n\\PARstart{A}{} large", 1)

PREAMBLE = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{xurl}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}

\newcommand{\Res}{R}
\newcommand{\per}{P}
""" + open(os.path.join(HERE, "numbers.tex")).read() + r"""
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2017.DOI}

\title{Where Should Resident Layers Go? Optimal Placement for Weight Streaming in Offloaded Large Language Model Inference}
\author{\uppercase{Quang-Vinh Dang}\authorrefmark{1},
\uppercase{Dat Le}\authorrefmark{2}, AND \uppercase{Minh Ngoc Dinh}\authorrefmark{3}}

\address[1]{British University Vietnam, Hung Yen, Vietnam (e-mail: vinh.dq4@buv.edu.vn; ORCID: 0000-0002-3877-8024)}
\address[2]{Department of Science, Technology, and International Projects, Ho Chi Minh City University of Economics and Finance, Ho Chi Minh City 700000, Vietnam (e-mail: datla@uef.edu.vn; ORCID: 0009-0005-5038-8570)}
\address[3]{Millennia Education, Ho Chi Minh City, Vietnam (e-mail: minh.dinh@maeducation.com; ORCID: 0000-0003-4026-6068)}
\tfootnote{This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.}

\markboth
{Dang \headeretal: Optimal Placement for Weight Streaming in Offloaded LLM Inference}
{Dang \headeretal: Optimal Placement for Weight Streaming in Offloaded LLM Inference}

\corresp{Corresponding author: Quang-Vinh Dang (e-mail: vinh.dq4@buv.edu.vn).}

\begin{abstract}
""" + open(os.path.join(HERE, "ieee", "abstract_ieee.tex")).read().strip() + r"""
\end{abstract}

\begin{keywords}
Compute Express Link, graphics processing units, large language model inference, layer placement, max-plus algebra, memory hierarchy, prefetching, weight offloading.
\end{keywords}

\titlepgskip=-21pt

\maketitle

"""

BACK = r"""
\section*{Data Availability}
The model code, the GPU runtime, the raw measurements and the scripts that produce every number, table and figure are available at \url{https://github.com/vinhqdang/Enabling-CXL-Based-Memory-Disaggregation-for-Large-Scale-AI-Systems}.

\section*{Acknowledgment}
""" + open(os.path.join(HERE, "ieee", "acknowledgment.tex")).read().strip() + r"""

\bibliographystyle{IEEEtran}
\bibliography{refs}

""" + open(os.path.join(HERE, "ieee", "biographies.tex")).read().strip() + r"""

\EOD

\end{document}
"""

tex = PREAMBLE + body + BACK
# IEEE class loads hyperref itself; elsarticle-only commands must not remain
for bad in ("\\sep", "frontmatter", "\\ead", "\\cortext", "elsarticle"):
    assert bad not in tex, bad

shutil.rmtree(BUILD, ignore_errors=True)
os.makedirs(os.path.join(BUILD, "figs"))
for f in glob.glob(os.path.join(CLS, "*")):
    if os.path.basename(f) not in ("abstract_ieee.tex", "acknowledgment.tex", "biographies.tex"):
        shutil.copy(f, BUILD)
for f in glob.glob(os.path.join(HERE, "figs", "*.pdf")):
    if os.path.basename(f)[:-4] in tex:
        shutil.copy(f, os.path.join(BUILD, "figs"))
shutil.copy(os.path.join(HERE, "refs.bib"), BUILD)
open(os.path.join(BUILD, "manuscript.tex"), "w").write(tex)


def run(cmd):
    subprocess.run(cmd, cwd=BUILD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"])
run(["bibtex", "manuscript"])
run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"])
run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"])
log = open(os.path.join(BUILD, "manuscript.log"), errors="replace").read()
errors = re.findall(r"^! .*", log, flags=re.M)
undefined = re.findall(r"(?:Reference|Citation) `[^']+' .*undefined", log)
pages = re.search(r"Output written on manuscript.pdf \((\d+) pages", log)
print("errors:", errors[:5], "undefined:", undefined[:5], "pages:", pages.group(1) if pages else "?")

# source archive: everything needed to compile, plus the bibliography output
zpath = os.path.join(OUT, "manuscript_source.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(BUILD):
        for f in files:
            if f.endswith((".aux", ".log", ".blg", ".out", ".pdf")) and root == BUILD:
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, BUILD))
shutil.copy(os.path.join(BUILD, "manuscript.pdf"), os.path.join(OUT, "manuscript.pdf"))
print("wrote", zpath, "and manuscript.pdf")
