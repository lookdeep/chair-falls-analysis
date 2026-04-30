set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

list:
    @just --list

setup:
    uv venv
    uv sync --project . --extra dev

lint:
    uv run --project . ruff check src scripts tests

test-smoke:
    uv run --project . pytest -m smoke

freeze-real-benchmark RUN_ID="consensus_v3_refresh_20260320T215557Z" FIXTURE_VERSION="frozen_v1" SOURCE_PROJECT_ROOT="" SOURCE_CONSENSUS_PATH="" OVERWRITE="false":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(freeze --source-run-id "{{RUN_ID}}" --fixture-version "{{FIXTURE_VERSION}}")
    if [ -n "{{SOURCE_PROJECT_ROOT}}" ]; then args+=(--source-project-root "{{SOURCE_PROJECT_ROOT}}"); fi
    if [ -n "{{SOURCE_CONSENSUS_PATH}}" ]; then args+=(--source-consensus-path "{{SOURCE_CONSENSUS_PATH}}"); fi
    if [ "{{OVERWRITE}}" = "true" ]; then args+=(--overwrite); fi
    uv run --project . python scripts/real_benchmark.py "${args[@]}"

replay-real-benchmark FIXTURE_VERSION="frozen_v1" RUN_ID="" OVERWRITE="false":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(replay --fixture-version "{{FIXTURE_VERSION}}")
    if [ -n "{{RUN_ID}}" ]; then args+=(--run-id "{{RUN_ID}}"); fi
    if [ "{{OVERWRITE}}" = "true" ]; then args+=(--overwrite); fi
    uv run --project . python scripts/real_benchmark.py "${args[@]}"

test-real-benchmark FIXTURE_VERSION="frozen_v1":
    REAL_BENCHMARK_FIXTURE_DIR="${REAL_BENCHMARK_FIXTURE_DIR:-data/fixtures/real_benchmark/{{FIXTURE_VERSION}}}" \
      uv run --project . pytest -m real_benchmark --tb=short -q

prep-dry-run:
    uv run --project . python scripts/run_all.py --dry-run

report-falls-descriptive RUN_ID="":
    if [ -n "{{RUN_ID}}" ]; then \
      uv run --project . python scripts/compile_falls_descriptive.py --run-id "{{RUN_ID}}"; \
    else \
      uv run --project . python scripts/compile_falls_descriptive.py; \
    fi

report-full-cohort-descriptive RUN_ID="":
    if [ -n "{{RUN_ID}}" ]; then \
      uv run --project . python scripts/compile_full_cohort_descriptive.py --run-id "{{RUN_ID}}"; \
    else \
      uv run --project . python scripts/compile_full_cohort_descriptive.py; \
    fi

report-chair-bed-inference RUN_ID="":
    if [ -n "{{RUN_ID}}" ]; then \
      uv run --project . python scripts/compile_chair_bed_inference_report.py --run-id "{{RUN_ID}}"; \
    else \
      uv run --project . python scripts/compile_chair_bed_inference_report.py; \
    fi

report-label-eval-shareholder RUN_ID="":
    if [ -n "{{RUN_ID}}" ]; then \
      uv run --project . python scripts/compile_label_eval_shareholder_report.py --run-id "{{RUN_ID}}"; \
    else \
      uv run --project . python scripts/compile_label_eval_shareholder_report.py; \
    fi

paper-build RUN_ID="consensus_v3_refresh_20260320T215557Z" REFRESH_SEED="false" REGENERATE_FIGURES="false" VERIFY_TEX="true" EMIT_PDF="false":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--run-id "{{RUN_ID}}")
    if [ "{{REFRESH_SEED}}" = "true" ]; then args+=(--refresh-seed); fi
    if [ "{{REGENERATE_FIGURES}}" = "true" ]; then args+=(--regenerate-figures); fi
    if [ "{{VERIFY_TEX}}" = "false" ]; then args+=(--skip-tex-verify); fi
    if [ "{{EMIT_PDF}}" = "true" ]; then args+=(--emit-pdf); fi
    uv run --project . python scripts/build_dual_track_paper.py "${args[@]}"

paper-reconcile DOCX="" OUTPUT_DIR="" OVERWRITE="false":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{DOCX}}" ]; then
      echo "DOCX=<path-to-exported-google-doc.docx> is required"
      exit 1
    fi
    args=(--docx "{{DOCX}}")
    if [ -n "{{OUTPUT_DIR}}" ]; then args+=(--output-dir "{{OUTPUT_DIR}}"); fi
    if [ "{{OVERWRITE}}" = "true" ]; then args+=(--overwrite); fi
    uv run --project . python scripts/reconcile_peer_edit_docx.py "${args[@]}"

shadow-ablation RUN_ID="" GATE_1="true" GATE_2="true":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--no-dry-run)
    if [ -n "{{RUN_ID}}" ]; then args+=(--run-id "{{RUN_ID}}"); fi
    if [ "{{GATE_1}}" = "true" ]; then args+=(--gate-1-pass); fi
    if [ "{{GATE_2}}" = "true" ]; then args+=(--gate-2-pass); fi
    uv run --project . python scripts/run_all.py "${args[@]}"
    rid="${RUN_ID}"
    if [ -z "$rid" ]; then
      rid=$(ls outputs/manifests/run_manifest_*.yaml 2>/dev/null | sort | tail -n 1 | sed 's#.*run_manifest_##; s#.yaml##')
    fi
    uv run --project . python scripts/compile_label_eval_shareholder_report.py --run-id "$rid"

compile-dashboard RUN_ID="":
    if [ -n "{{RUN_ID}}" ]; then       uv run --project . python scripts/compile_chair_fall_dashboard.py --run-id "{{RUN_ID}}";     else       uv run --project . python scripts/compile_chair_fall_dashboard.py;     fi

report-compare-descriptive SCOPE="full_cohort" BASELINE_RUN_ID="" CANDIDATE_RUN_ID="":
    if [ -n "{{BASELINE_RUN_ID}}" ] && [ -n "{{CANDIDATE_RUN_ID}}" ]; then \
      uv run --project . python scripts/compare_descriptive_outputs.py --scope "{{SCOPE}}" --baseline-run-id "{{BASELINE_RUN_ID}}" --candidate-run-id "{{CANDIDATE_RUN_ID}}"; \
    else \
      uv run --project . python scripts/compare_descriptive_outputs.py --scope "{{SCOPE}}"; \
    fi

model RUN_ID="":
    uv run python scripts/run_modeling.py {{if RUN_ID != "" {"--run-id " + RUN_ID} else {""}}}

manifest-validate:
    @RUN_ID=$$(ls outputs/manifests/run_manifest_*.yaml 2>/dev/null | tail -n 1 | sed 's#.*run_manifest_##; s#.yaml##') ; \
    test -n "$$RUN_ID" ; \
    test -f "outputs/manifests/run_manifest_$$RUN_ID.yaml" ; \
    echo "Validated outputs/manifests/run_manifest_$$RUN_ID.yaml"

# ---------------------------------------------------------------------------
# Fresh end-to-end pipeline run + status check
# ---------------------------------------------------------------------------

# Wipe all previous run artifacts so the next run starts clean.
clean:
    rm -rf data/raw/* data/staged/* outputs/qa/* outputs/audit/* outputs/manifests/*
    rm -f outputs/*.html
    @echo "Cleaned data/raw, data/staged, and outputs."

# Run the full extract → transform → QA → manifest pipeline (live BQ).
run RUN_ID="" GATE_1="" GATE_2="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--no-dry-run)
    if [ -n "{{RUN_ID}}" ]; then args+=(--run-id "{{RUN_ID}}"); fi
    if [ "{{GATE_1}}" = "true" ]; then args+=(--gate-1-pass); fi
    if [ "{{GATE_2}}" = "true" ]; then args+=(--gate-2-pass); fi
    uv run --project . python scripts/run_all.py "${args[@]}"

# Compile all HTML reports from the latest (or given) run.
# Chair-bed inference report is skipped when run mode is descriptive_only
# (its artifacts only exist in inferential_ready mode).
reports RUN_ID="":
    #!/usr/bin/env bash
    set -euo pipefail
    rid_flag=""
    if [ -n "{{RUN_ID}}" ]; then rid_flag="--run-id {{RUN_ID}}"; fi
    uv run --project . python scripts/compile_falls_descriptive.py $rid_flag
    uv run --project . python scripts/compile_full_cohort_descriptive.py $rid_flag
    if [ -n "{{RUN_ID}}" ]; then
      GATE_FILE="outputs/qa/gate_preflight_{{RUN_ID}}.json"
    else
      GATE_FILE=$(ls outputs/qa/gate_preflight_*.json 2>/dev/null | sort | tail -n 1 || true)
    fi
    RUN_MODE=$(python3 -c "import json; g=json.load(open('$GATE_FILE')); print(g.get('effective_run_mode','descriptive_only'))" 2>/dev/null || echo "descriptive_only")
    if [ "$RUN_MODE" = "inferential_ready" ]; then
      uv run --project . python scripts/compile_chair_bed_inference_report.py $rid_flag
    else
      echo "Skipping chair-bed inference report (run_mode=$RUN_MODE — requires inferential_ready)."
    fi
    uv run --project . python scripts/compile_label_eval_shareholder_report.py $rid_flag
    echo "All reports compiled."

# Lint, test, validate manifest, and summarise output health.
status:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "===== LINT ====="
    uv run --project . ruff check src scripts tests && echo "OK" || echo "FAIL"
    echo ""
    echo "===== SMOKE TESTS ====="
    uv run --project . pytest -m smoke --tb=short -q && echo "OK" || echo "FAIL"
    echo ""
    echo "===== FULL TESTS ====="
    uv run --project . pytest --tb=short -q && echo "OK" || echo "FAIL"
    echo ""
    echo "===== MANIFEST ====="
    MANIFEST=$(ls outputs/manifests/run_manifest_*.yaml 2>/dev/null | grep -v smoke_pipeline | sort | tail -n 1 || true)
    if [ -n "$MANIFEST" ]; then
      echo "Latest manifest: $MANIFEST"
      RUN_ID=$(echo "$MANIFEST" | sed 's#.*run_manifest_##; s#.yaml##')
      echo "Run ID: $RUN_ID"
    else
      echo "No manifest found."
    fi
    echo ""
    echo "===== OUTPUT ARTIFACTS ====="
    echo "  raw dirs:     $(ls -d data/raw/*/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "  staged dirs:  $(ls -d data/staged/*/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "  qa files:     $(ls outputs/qa/* 2>/dev/null | wc -l | tr -d ' ')"
    echo "  audit files:  $(ls outputs/audit/* 2>/dev/null | wc -l | tr -d ' ')"
    echo "  manifests:    $(ls outputs/manifests/* 2>/dev/null | wc -l | tr -d ' ')"
    echo "  HTML reports: $(ls outputs/*.html 2>/dev/null | wc -l | tr -d ' ')"
    echo ""
    echo "===== GATE STATUS ====="
    GATE=$(ls outputs/qa/gate_preflight_*.json 2>/dev/null | grep -v smoke_pipeline | sort | tail -n 1 || true)
    if [ -n "$GATE" ]; then
      python3 -c "import json,sys; g=json.load(open('$GATE')); print(f'  gate_1_pass: {g.get(\"gate_1_pass\",\"?\")}'); print(f'  gate_2_pass: {g.get(\"gate_2_pass\",\"?\")}'); print(f'  run_mode:    {g.get(\"effective_run_mode\",\"?\")}')"
    else
      echo "  No gate preflight found."
    fi

# Build Word submission package for ACI ScholarOne upload.
# Requires pandoc 3.x. Outputs: paper/build/{manuscript,cover_letter,strobe_supplement}.docx
build-submission:
    mkdir -p paper/build
    cd paper && pandoc manuscript.md \
      --from markdown \
      --to docx \
      --citeproc \
      --metadata-file metadata.yaml \
      --bibliography arxiv/references.bib \
      --csl csl/american-medical-association.csl \
      --resource-path . \
      -o build/manuscript.docx
    cd paper && pandoc cover_letter.md \
      --from markdown \
      --to docx \
      -o build/cover_letter.docx
    cd paper && pandoc ../docs/strobe_checklist.md \
      --from markdown \
      --to docx \
      -o build/strobe_supplement.docx
    @echo "Submission package written to paper/build/"

# Clean → run pipeline → compile reports → status check.  The full loop.
fresh RUN_ID="":
    #!/usr/bin/env bash
    set -euo pipefail
    just clean
    rid_flag=""
    if [ -n "{{RUN_ID}}" ]; then rid_flag="--run-id {{RUN_ID}}"; fi
    uv run --project . python scripts/run_all.py --no-dry-run --gate-1-pass --gate-2-pass $rid_flag
    just reports "{{RUN_ID}}"
    just status
