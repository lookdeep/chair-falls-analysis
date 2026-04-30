# ruff: noqa: E402

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.qa import _consensus_study_window_reconciliation_rows


def test_consensus_study_window_reconciliation_rows_report_full_file_and_study_window_counts(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "data" / "public"
    public_dir.mkdir(parents=True)
    consensus_path = public_dir / "falls-observations-v3-consensus.csv"
    consensus_path.write_text(
        "\n".join(
            [
                "event_key,last_furniture,furniture_departure_time,prefall_location,fall_time_consensus,response_time_consensus,fall_tags",
                "100|2024-07-31|23:59,,,,23:59:10,,slip",
                "101|2024-08-01|00:01,,,,00:01:10,00:01:40,slip",
                "102|2025-06-01|12:00,,,,, ,no_fall",
                "103|2025-12-31|23:58,,,,23:58:10,23:58:40,support_bed",
                "104|2026-01-01|00:01,,,,00:01:10,00:01:40,slip",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = Settings(
        project_root=tmp_path,
        run_id="test",
        fall_labels_consensus_csv_path=Path("data/public/falls-observations-v3-consensus.csv"),
        study_start_date=date(2024, 8, 1),
        study_end_date=date(2025, 12, 31),
    )

    rows = _consensus_study_window_reconciliation_rows(settings)
    row_map = {row["metric"]: row["value"] for row in rows}

    assert row_map["consensus_source_unique_monitors_full_file"] == 5
    assert row_map["consensus_included_fall_unique_monitors_full_file"] == 4
    assert row_map["consensus_source_rows_study_window"] == 3
    assert row_map["consensus_source_unique_monitors_study_window"] == 3
    assert row_map["consensus_included_fall_rows_study_window"] == 2
    assert row_map["consensus_included_fall_unique_monitors_study_window"] == 2
