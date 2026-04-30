# chair-falls-analysis

Public release of the analysis code, manuscript source, and curated aggregate data bundle for:

> **Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring**
> Paolo Gabriel, Peter Rehani, Zack Drumm, Tyler Troy, Tiffany Wyatt, Narinder Singh
> Submitted to *IEEE Journal of Biomedical and Health Informatics* (Sensor Informatics).
> Preprint: [arXiv:2603.22785](https://arxiv.org/abs/2603.22785)

Project page: <https://lookdeep.github.io/chair-falls-analysis/>

## What's in here

- `paper/manuscript.md` — canonical manuscript source (markdown)
- `paper/supplement.md` — supplementary material (full sensitivity panel, label-quality benchmark, multimodal-LLM benchmark, mechanism-coded observation cohort, furniture-origin analyses)
- `paper/cover_letter.md` — J-BHI cover letter
- `paper/build/` — pre-rendered .docx artifacts (manuscript, supplement, cover letter, STROBE supplement)
- `paper/csl/ieee.csl` — IEEE citation style
- `paper/metadata.yaml` — pandoc build metadata
- `paper/arxiv/references.bib` — bibliography
- `paper/assets/*.png` — main-text figures
- `data/public/falls-observations-v3-consensus.csv` — adjudicated fall annotations (de-identified, aggregated)
- `data/public/derived/` — frozen aggregate derived tables used by the manuscript and figures (run id: `consensus_v3_refresh_20260320T215557Z`)
- `data/public/gemini/` — multimodal-LLM benchmark precomputed outputs
- `figures/` — manuscript figure source files (PNG + SVG)
- `src/`, `scripts/`, `sql/`, `tests/` — analysis code
- `docs/strobe_checklist.md`, `docs/strobe_flow_diagram.md` — STROBE-compliant cohort flow
- `docs/reproduce_this_paper.md` — reproducibility walkthrough
- `docs/artifact_map.md`, `docs/analysis_base_data_dictionary.md` — data dictionary

## Quick start

```bash
uv venv
uv sync --project .
cp .env.example .env
just paper-build   # rebuilds manuscript .tex and peer-edit .docx from paper/manuscript.md
```

Frozen outputs in `data/public/derived/` are sufficient to regenerate every numeric value reported in the manuscript without rerunning the upstream pipeline. Individual-level video and patient-level tables cannot be released under the existing Data Use Agreement.

## Reproducibility

The full pipeline (upstream extract → transform → QA → modeling) is shipped here for transparency, but raw video and patient-level intermediate tables are not. The `data/public/` bundle is the closed-form input to every downstream report; figures and tables in the manuscript trace deterministically to those CSVs at frozen run id `consensus_v3_refresh_20260320T215557Z`.

See `docs/reproduce_this_paper.md` for the step-by-step path from CSVs → manuscript .docx.

## Citation

Citation block will be added when the paper is accepted. For now, please cite the preprint:

```bibtex
@article{gabriel2026chairfalls,
  author = {Gabriel, Paolo and Rehani, Peter and Drumm, Zack and Troy, Tyler and Wyatt, Tiffany and Singh, Narinder},
  title  = {Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring},
  year   = {2026},
  eprint = {2603.22785},
  archivePrefix = {arXiv},
  doi    = {10.48550/arXiv.2603.22785}
}
```

## License

CC0 1.0 Universal (Public Domain Dedication). See [LICENSE](LICENSE).

## Data and ethics

All video data was de-identified upstream via facial blurring and removal of direct identifiers; only AI-derived per-hour state estimates and aggregated counts enter the analytic dataset. Because no identifiable private information was obtained or analyzed, the study does not constitute human subjects research under 45 CFR 46.102(e)(1)(ii) and HIPAA Safe Harbor de-identification (45 CFR 164.514(b)) applies. No IRB review was required.

## Acknowledgements

This work was conducted under internal research operations at LookDeep Health. No external grant funding supported the study.
