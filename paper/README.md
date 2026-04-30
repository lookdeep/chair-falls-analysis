# Manuscript Workflow

`paper/manuscript.md` is the canonical manuscript source.

Local build outputs:

- `paper/arxiv/manuscript.tex` plus `paper/arxiv/assets/` for TeX/arXiv packaging
- `paper/peer/chair_fall_paper_peer_edit.docx` for collaborator review

These are generated build artifacts, not co-masters. Review comments should still be reconciled back into `paper/manuscript.md`.

Build commands:

```bash
just paper-build
uv run --project . python scripts/build_dual_track_paper.py
```

Reconcile an exported Google Doc back to the canonical manuscript:

```bash
just paper-reconcile DOCX="/absolute/path/to/peer_edit_export.docx"
```

That command creates local comparison artifacts under `paper/reconcile/`:

- `exported.raw.md`
- `exported.normalized.md`
- `canonical.normalized.md`
- `canonical_vs_exported.diff`
- `summary.md`

Use those files to port accepted edits manually into `paper/manuscript.md`, then rerun `just paper-build`.

Optional local seed refresh:

```bash
uv run --project . python scripts/build_dual_track_paper.py --refresh-seed
```

`--refresh-seed` only works when a local `docs/chair_fall_paper_medrxiv.docx` seed exists. That helper is optional and is not part of the public release surface.

Citation policy:

- `paper/arxiv/references.bib` is the maintained bibliography source of truth
- `paper/manuscript.md` uses Pandoc citekeys rather than hand-numbered references
- `paper/csl/american-medical-association.csl` defines the AMA-style output used for both TeX and peer-review DOCX
- The `# References` block in `paper/manuscript.md` is only a citeproc placeholder; do not hand-edit bibliography text there

Figure policy:

- Main-text figures are staged as lowercase PNG files in `paper/assets/`
- The manuscript carries six main-text figures in order of appearance
- `paper/arxiv/assets/` is generated locally at build time rather than tracked as source
- Generator IDs differ from manuscript numbering: `fig7_operational_signals` stages as manuscript Figure 4, `fig4_mechanism_by_posture` stages as manuscript Figure 5, and `fig5_chair_occupancy_daypart` stages as manuscript Figure 6
- `fig6_context_schematic` is retained as Appendix Figure A1; no per-event patient frames remain in the manuscript bundle
- The interactive SVG simulator is excluded from the manuscript bundle
