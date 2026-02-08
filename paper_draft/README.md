# Paper Draft

This directory contains the LaTeX source code for the research paper.

## Structure
- `main.tex`: The primary document file.
- `references.bib`: Bibliography file.
- `sections/`: Contains individual LaTeX files for each section.
- `commands/`: Contains custom macros and formatting definitions.
- `figures/`: Place figures here (if any).
- `tables/`: Place complex table definitions here (if any).

## Compilation

To compile the paper, ensure you have a LaTeX distribution (like TeX Live) installed.

Run the following commands:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The output will be `main.pdf`.
