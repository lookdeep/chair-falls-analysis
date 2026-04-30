# IEEE Format Conversion — J-BHI Submission

The pandoc build pipeline produces submission-ready content (manuscript text, supplement, cover letter, IEEE-numbered references) but emits single-column LaTeX and single-column DOCX. J-BHI requires IEEE single-spaced double-column PDF with embedded figures/tables. The final visual layout is one of the two paths below.

## Path A — Overleaf with IEEEtran (recommended)

1. Create new Overleaf project from the IEEE Conference / Journal template (search "IEEEtran" in Overleaf's gallery).
2. Replace the template's `main.tex` with `paper/arxiv/manuscript.tex` (already pandoc-rendered with IEEE CSL).
3. Add `paper/arxiv/references.bib` to the project.
4. Replace the template's preamble lines with:
   ```latex
   \documentclass[journal]{IEEEtran}
   \usepackage{cite}
   \usepackage{graphicx}
   \usepackage{amsmath}
   \usepackage{booktabs}
   ```
5. Wrap the abstract with `\begin{abstract} ... \end{abstract}` and Index Terms with `\begin{IEEEkeywords} ... \end{IEEEkeywords}`. The pandoc output already separates these; minor manual adjustment.
6. Add IEEEtran-specific author block:
   ```latex
   \author{Paolo~Gabriel, Peter~Rehani, Zack~Drumm, Tyler~Troy, Tiffany~Wyatt, Narinder~Singh
   \thanks{All authors are with LookDeep Health.}}
   ```
7. Replace `\section{Section Name}` headings as-is — IEEEtran auto-numbers.
8. Compile (Overleaf has pdflatex installed). Iterate on page count toward 7 pages.

## Path B — IEEE Word Template

1. Download `JBHI-Word-Template.docx` from IEEE's author center.
2. Open `paper/build/manuscript.docx` (pandoc output) in Word.
3. Copy paragraph-by-paragraph into the IEEE Word template, using the template's pre-defined styles (Title, Author, Abstract, Section Heading, etc.).
4. Re-link figures and re-import tables (pandoc's grid tables don't always survive transfer cleanly).
5. Word's word count and built-in IEEE template will produce the double-column layout.

## Word Count Target

- Current main text: 3,247 words (Introduction → Conclusions, paragraphs only, no tables / figures / references).
- J-BHI ≤7 printed pages ≈ 5,000–6,000 words + figures/tables.
- Substantial headroom for figures/tables. No further trimming needed.

## Submission Bundle (5 files)

1. `paper/build/manuscript.docx` (or PDF after Path A/B)
2. `paper/build/cover_letter.docx`
3. `paper/build/supplement.docx` (consolidated supplementary material)
4. STROBE checklist (`paper/build/strobe_supplement.docx`) — optional but disclosed
5. References (`.bib` if Path A, embedded if Path B)
