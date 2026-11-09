"""Build the anonymised submission source: one .tex file (inputs, numbers and bibliography inlined) and PNG figures."""
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "submission_fgcs")
BUILD = os.path.join(OUT, "build_anon")
FIGS = ["fig_accuracy", "fig_penalty", "fig_timeline", "fig_slots", "fig_nonuniform"]

src = open(os.path.join(HERE, "main.tex")).read()
for _ in range(3):
    src = re.sub(r"\\input\{([^}]+\.tex)\}[^\n]*", lambda m: open(os.path.join(HERE, m.group(1))).read().rstrip("\n"), src)
assert "\\input{" not in src

# anonymise: authors, affiliations, contributions, repository link
src = "\n".join(l for l in src.split("\n") if not re.match(r"\\(author|ead|cortext|address)\b", l))
src = re.sub(r"\\section\*\{CRediT authorship contribution statement\}\n.*?\n\n", "", src, flags=re.S)
src = re.sub(r"available at \\url\{[^}]*\}", "available in a public repository (link withheld for anonymous review)", src)
assert "Dang" not in src and "vinhqdang" not in src and "buv" not in src.lower()

# bibliography inlined
bbl = open(os.path.join(HERE, "main.bbl")).read()
src = re.sub(r"\\bibliographystyle\{[^}]*\}\n\\bibliography\{[^}]*\}", lambda m: bbl, src)
assert "\\bibliography{" not in src

# PNG figures in the root of the archive
for i, f in enumerate(FIGS, 1):
    src = src.replace("figs/%s.pdf" % f, "Figure_%d.png" % i)
assert "figs/" not in src

shutil.rmtree(BUILD, ignore_errors=True)
os.makedirs(BUILD)
open(os.path.join(BUILD, "manuscript.tex"), "w").write(src)
for i, f in enumerate(FIGS, 1):
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", os.path.join(HERE, "figs", f + ".pdf"),
                    os.path.join(BUILD, "Figure_%d" % i)], check=True, stderr=subprocess.DEVNULL)
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], cwd=BUILD, stdout=subprocess.DEVNULL)
log = open(os.path.join(BUILD, "manuscript.log")).read()
assert "Undefined control" not in log and "Citation" not in log.split("LaTeX Warning")[-1][:0] + ""
zpath = os.path.join(OUT, "manuscript_source.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(BUILD, "manuscript.tex"), "manuscript.tex")
    for i in range(1, len(FIGS) + 1):
        z.write(os.path.join(BUILD, "Figure_%d.png" % i), "Figure_%d.png" % i)
shutil.copy(os.path.join(BUILD, "manuscript.pdf"), os.path.join(OUT, "manuscript.pdf"))
print("wrote", zpath, "and manuscript.pdf;", "undefined refs:", log.count("undefined"))
