# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.config import Settings
from ld_chair_falls.qa import write_falls_descriptive_tables


def test_write_falls_descriptive_tables_canonicalizes_site_aliases(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, run_id="site_alias_fixture")
    raw_tables = {
        "fall_events_source": pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T10:00:00Z",
                    "2026-01-02T10:00:00Z",
                    "2026-01-03T10:00:00Z",
                    "2026-01-04T10:00:00Z",
                ],
                "timestamp_local": [
                    "2026-01-01T04:00:00Z",
                    "2026-01-02T04:00:00Z",
                    "2026-01-03T04:00:00Z",
                    "2026-01-04T04:00:00Z",
                ],
                "hospital_id": [5, 5, 5, 5],
                "division_id": [37, 37, 37, 20],
                "hospital_system_name": ["Baptist", "Baptist", "Baptist", "Baptist"],
                "hospital_name": ["DeSoto", "Desoto", "DeSoto", "CC"],
                "patient_id": [1001, 1002, 1003, 1004],
                "monitor_id": [2001, 2002, 2003, 2004],
            }
        )
    }

    write_falls_descriptive_tables(settings, raw_tables=raw_tables)

    falls_by_site = pd.read_csv(tmp_path / "outputs" / "qa" / "falls_by_site_site_alias_fixture.csv")
    aliases = pd.read_csv(tmp_path / "outputs" / "qa" / "falls_site_name_aliases_site_alias_fixture.csv")

    de_soto_rows = falls_by_site.loc[falls_by_site["division_id"] == 37]
    assert len(de_soto_rows.index) == 1
    assert int(de_soto_rows.iloc[0]["falls"]) == 3
    assert de_soto_rows.iloc[0]["hospital_name"] == "DeSoto"

    assert not aliases.empty
    assert "Desoto" in aliases["alias_hospital_name"].tolist()
    assert "DeSoto" in aliases["canonical_hospital_name"].tolist()
