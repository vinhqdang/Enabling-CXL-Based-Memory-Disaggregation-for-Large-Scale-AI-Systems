# IEEE Access submission package

Built by `python3 paper/make_ieee.py` from the same sources as the main manuscript (`paper/`).

| File | Purpose |
|---|---|
| `manuscript.pdf` | PDF compiled from the LaTeX source (15 pages, ieeeaccess template) |
| `manuscript_source.zip` | LaTeX source: `manuscript.tex`, `manuscript.bbl`, `refs.bib`, `figs/*.pdf`, class, style and font files |
| `cover_letter.pdf` / `.tex` | Cover letter |
| `title_abstract_keywords.txt` | Title, abstract (240 words), 8 keywords, author data for the submission form |

## Checklist against the IEEE Access guidelines

- [x] Double-column IEEE Access template, PDF and LaTeX source with identical content
- [x] All three authors with affiliations, e-mail and ORCID (not anonymised)
- [x] Abstract 150-250 words, no abbreviations or references; 8 keywords in alphabetical order
- [x] Abbreviations defined at first use in the text
- [x] Generative AI use disclosed in the Acknowledgment
- [x] Author biographies below the references
- [x] Under 20 pages; package well under 40 MB
- [ ] Biographies of Dat Le and Minh Ngoc Dinh: only affiliation so far; extend in `paper/ieee/biographies.tex`
- [ ] Author photos (optional): replace `IEEEbiographynophoto` with `IEEEbiography` and add photos
- [ ] ORCID records public for all authors
- [ ] Manuscript type in the portal: Research Article
