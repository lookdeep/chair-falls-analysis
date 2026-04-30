# Cover Letter — J-BHI Submission

**Journal:** IEEE Journal of Biomedical and Health Informatics (J-BHI)
**Manuscript title:** Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring
**Target section:** Sensor Informatics
**Authors:** Paolo Gabriel, Peter Rehani, Zack Drumm, Tyler Troy, Tiffany Wyatt, Narinder Singh
**Corresponding author:** Paolo Gabriel (paolo@lookdeep.health), LookDeep Health
**Submission type:** Research Article

---

Dear Editor,

We submit for your consideration a sensor-informatics methodology paper on continuous video-based patient monitoring. Hospital fall surveillance is conventionally reported per occupied bed-day — a denominator that conflates bed time, chair time, transfers, and ambulation. We replace that blended denominator with probability-weighted position-specific exposure-hours derived from per-hour computer-vision state estimates and propagate classifier label uncertainty into the downstream rate inference via systematic misclassification-sensitivity scenarios. The manuscript characterizes the resulting denominator framework on a retrospective deployment spanning 11 hospitals (40 adjudicated falls; 320.5 chair-hours; 5,121.4 bed-hours) and identifies improved position-label calibration as the primary prerequisite for confirmatory inference in continuous-monitoring settings. The contribution is methodological — a reusable pipeline for translating probabilistic per-hour sensor states into rate inferences with explicit uncertainty propagation — applicable beyond fall surveillance to any deployment where state-label calibration is imperfect.

**Why J-BHI Sensor Informatics.** The work fits J-BHI's Sensor Informatics scope: the core problem is acquisition-to-inference for continuous biomedical sensor streams (computer-vision-derived per-hour state estimates), with measurable label-quality bounds, in a constrained hospital deployment context. The contribution is original IT content in measurement methodology — denominator construction, probabilistic event allocation, and uncertainty propagation — not clinical validation alone.

**Novelty statement.** To our knowledge, this is the first published evaluation of (i) probability-weighted position-specific exposure-hour denominators derived from a deployed CV pipeline at multi-hospital scale, and (ii) a misclassification-sensitivity framework that propagates classifier-level label uncertainty (macro F1 = 0.528; ECE = 0.450) into estimate-level inferential bounds for continuous-monitoring rate estimation.

**Related preprint.** A preprint of this manuscript was posted to arXiv prior to peer review (arXiv:2603.22785; doi:10.48550/arXiv.2603.22785). Relative to the preprint, this submission reframes the contribution toward sensor-informatics methodology, designates HC3 robust standard errors as the primary uncertainty estimator, and consolidates auxiliary analyses (label-quality benchmark, mechanism taxonomy, multimodal-LLM benchmark, furniture-origin analyses) into supplementary material.

**No duplicate submission.** This manuscript is not under consideration at any other journal, and no overlapping content has been submitted elsewhere.

**Ethics determination.** Video data was de-identified prior to analysis via upstream facial blurring and removal of direct identifiers; the analytic dataset contains only AI-derived per-hour state estimates and aggregated counts. Because no identifiable private information was obtained or analyzed, this study does not constitute human subjects research under 45 CFR 46.102(e)(1)(ii) and HIPAA Safe Harbor de-identification (45 CFR 164.514(b)) applies. No IRB review was required. Outcomes did not influence patient care.

**Conflict of interest.** All authors are current or former employees of LookDeep Health, which developed, operates, and commercially deploys the AI monitoring system evaluated. The same team defined the adjudicated ground truth, built the measurement instrument, conducted the analysis, and interpreted the results. This is disclosed in the manuscript Conflict of Interest section.

**Open Access decision.** Open Access ($1,950 APC). Confirmed at submission.

**Suggested reviewers.** None nominated; we defer reviewer selection to the editorial team.

**Funding.** No external grant funding supported this study; the work was conducted under internal research operations at LookDeep Health.

**Data and code availability.** Analysis code, aggregate output tables, and the curated public data bundle are openly available in the project's public GitHub repository (https://github.com/lookdeep/chair-falls-risk), with a permanent DOI archive deposited to Zenodo at publication. Individual-level video and patient-level tables cannot be shared under the Data Use Agreement between LookDeep Health and the participating health system.

We believe this work advances sensor-informatics methodology for continuous-monitoring deployments and look forward to the editorial team's consideration.

Sincerely,

Paolo Gabriel
paolo@lookdeep.health
ORCID: 0000-0002-5464-769X
LookDeep Health

---

**Outstanding administrative items before final ScholarOne upload (for Paolo to resolve):**

1. **ORCIDs from co-authors** — required at submission for all authors. Currently committed: PG (0000-0002-5464-769X). Need: PR, ZD, TT, TW, NS.
2. **Public GitHub repo URL** — used in Data Availability and this cover letter as `https://github.com/lookdeep/chair-falls-risk`. Confirm this is the final URL or update to the actual repo before upload (the lookdeep org currently has no public chair-falls repo; a public release branch or new repo is needed).
3. **IEEE format conversion** — see `paper/build/IEEE-format-conversion.md`. The pandoc-rendered .docx and .tex are submission-ready in content; double-column IEEEtran formatting is a final visual layout pass best done in Overleaf or with the IEEE Word template.
