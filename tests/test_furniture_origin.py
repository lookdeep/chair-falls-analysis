# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ld_chair_falls.label_eval import (
    _classify_origin_chain,
    _furniture_origin_chain_summary,
    _post_departure_latency_table,
    _tag_origin_chain_crosstab,
)
from ld_chair_falls.modeling import _reclassify_chair_origin_falls

# ── _classify_origin_chain tests ──


class TestClassifyOriginChain:
    def test_direct_chair(self):
        """Test that prefall_location='chair' returns 'direct_chair'."""
        assert _classify_origin_chain("chair", "") == "direct_chair"
        assert _classify_origin_chain("Chair", None) == "direct_chair"
        assert _classify_origin_chain("CHAIR", "bed") == "direct_chair"

    def test_direct_bed(self):
        """Test that prefall_location='bed' returns 'direct_bed'."""
        assert _classify_origin_chain("bed", "") == "direct_bed"
        assert _classify_origin_chain("Bed", "chair") == "direct_bed"
        assert _classify_origin_chain("BED", None) == "direct_bed"

    def test_chair_origin_room(self):
        """Test that room/no_patient + chair furniture returns 'chair_origin_room'."""
        assert _classify_origin_chain("room", "chair") == "chair_origin_room"
        assert _classify_origin_chain("no_patient", "chair") == "chair_origin_room"
        assert _classify_origin_chain("Room", "Chair") == "chair_origin_room"

    def test_bed_origin_room(self):
        """Test that room/no_patient + bed furniture returns 'bed_origin_room'."""
        assert _classify_origin_chain("room", "bed") == "bed_origin_room"
        assert _classify_origin_chain("no_patient", "bed") == "bed_origin_room"
        assert _classify_origin_chain("NO_PATIENT", "Bed") == "bed_origin_room"

    def test_floor_origin_room(self):
        """Test that room/no_patient + floor furniture returns 'floor_origin_room'."""
        assert _classify_origin_chain("room", "floor") == "floor_origin_room"
        assert _classify_origin_chain("no_patient", "floor") == "floor_origin_room"

    def test_unknown_origin_room(self):
        """Test that room/no_patient + empty/unknown furniture returns 'unknown_origin_room'."""
        assert _classify_origin_chain("room", "") == "unknown_origin_room"
        assert _classify_origin_chain("room", "other_furniture") == "unknown_origin_room"
        assert _classify_origin_chain("no_patient", None) == "unknown_origin_room"
        assert _classify_origin_chain("room", "   ") == "unknown_origin_room"

    def test_other(self):
        """Test that unrecognized prefall_location returns 'other'."""
        assert _classify_origin_chain("bathroom", "") == "other"
        assert _classify_origin_chain("", "") == "other"
        assert _classify_origin_chain("kitchen", "chair") == "other"
        assert _classify_origin_chain("hallway", "bed") == "other"


# ── _furniture_origin_chain_summary tests ──


class TestFurnitureOriginChainSummary:
    def test_basic_summary(self):
        """Test basic aggregation of furniture_origin_chain counts and percentages."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair", "direct_chair", "direct_bed", "chair_origin_room"],
        })
        result = _furniture_origin_chain_summary(truth)
        assert set(result.columns) == {"furniture_origin_chain", "event_count", "pct_of_events"}
        assert len(result) == 3
        # direct_chair should have 2 events
        chair_row = result.loc[result["furniture_origin_chain"] == "direct_chair"]
        assert int(chair_row["event_count"].iloc[0]) == 2
        assert float(chair_row["pct_of_events"].iloc[0]) == 50.0

    def test_percentages_sum_to_100(self):
        """Test that percentages across all categories sum to 100."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair"] * 3 + ["direct_bed"] * 2 + ["chair_origin_room"] * 5,
        })
        result = _furniture_origin_chain_summary(truth)
        total_pct = result["pct_of_events"].sum()
        assert abs(total_pct - 100.0) < 0.1  # Allow for rounding

    def test_empty_input(self):
        """Test that empty DataFrame returns empty result."""
        result = _furniture_origin_chain_summary(pd.DataFrame())
        assert result.empty
        assert set(result.columns) == {"furniture_origin_chain", "event_count", "pct_of_events"}

    def test_missing_column(self):
        """Test that missing furniture_origin_chain column returns empty result."""
        result = _furniture_origin_chain_summary(pd.DataFrame({"other_col": [1, 2]}))
        assert result.empty

    def test_single_category(self):
        """Test aggregation with single category."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair"] * 5,
        })
        result = _furniture_origin_chain_summary(truth)
        assert len(result) == 1
        assert result.iloc[0]["event_count"] == 5
        assert result.iloc[0]["pct_of_events"] == 100.0


# ── _post_departure_latency_table tests ──


class TestPostDepartureLatencyTable:
    def test_basic_latency(self):
        """Test basic latency aggregation by furniture type."""
        truth = pd.DataFrame({
            "last_furniture": ["bed", "bed", "bed", "chair"],
            "post_departure_latency_seconds": [10.0, 20.0, 30.0, 15.0],
        })
        result = _post_departure_latency_table(truth)
        assert len(result) == 2
        bed_row = result.loc[result["last_furniture"] == "bed"]
        assert float(bed_row["latency_seconds_median"].iloc[0]) == 20.0
        assert float(bed_row["n_events"].iloc[0]) == 3
        assert float(bed_row["latency_seconds_mean"].iloc[0]) == 20.0

    def test_latency_percentiles(self):
        """Test percentile calculations."""
        truth = pd.DataFrame({
            "last_furniture": ["chair"] * 10,
            "post_departure_latency_seconds": list(range(10, 110, 10)),  # 10, 20, ..., 100
        })
        result = _post_departure_latency_table(truth)
        assert len(result) == 1
        row = result.iloc[0]
        # For [10, 20, 30, ..., 100]: p25 should be ~32.5, p75 should be ~77.5
        assert float(row["latency_seconds_p25"]) > 0
        assert float(row["latency_seconds_p75"]) > float(row["latency_seconds_p25"])
        assert float(row["latency_seconds_min"]) == 10.0
        assert float(row["latency_seconds_max"]) == 100.0

    def test_empty_input(self):
        """Test that empty DataFrame returns empty result."""
        result = _post_departure_latency_table(pd.DataFrame())
        assert result.empty

    def test_all_null_latency(self):
        """Test that all-null latency returns empty result."""
        truth = pd.DataFrame({
            "last_furniture": ["bed", "chair"],
            "post_departure_latency_seconds": [None, None],
        })
        result = _post_departure_latency_table(truth)
        assert result.empty

    def test_blank_furniture_excluded(self):
        """Test that empty/blank last_furniture is excluded."""
        truth = pd.DataFrame({
            "last_furniture": ["", "bed", None, "chair"],
            "post_departure_latency_seconds": [10.0, 20.0, 30.0, 15.0],
        })
        result = _post_departure_latency_table(truth)
        # Should only have bed and chair (empty and None excluded)
        furns = set(result["last_furniture"])
        assert "" not in furns
        assert None not in furns

    def test_null_latency_in_group_excluded(self):
        """Test that null latency values within a group are excluded."""
        truth = pd.DataFrame({
            "last_furniture": ["bed", "bed", "bed"],
            "post_departure_latency_seconds": [10.0, None, 20.0],
        })
        result = _post_departure_latency_table(truth)
        bed_row = result.loc[result["last_furniture"] == "bed"]
        assert int(bed_row["n_events"].iloc[0]) == 2  # Only 2 non-null values


# ── _tag_origin_chain_crosstab tests ──


class TestTagOriginChainCrosstab:
    def test_basic_crosstab(self):
        """Test basic crosstab with multiple tags and chains."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair", "direct_chair", "direct_bed"],
            "fall_tags": ["slip, footrest", "tumble", "slip"],
        })
        result = _tag_origin_chain_crosstab(truth)
        assert not result.empty
        assert set(result.columns) == {"furniture_origin_chain", "fall_tag", "count", "pct_within_chain", "pct_global"}
        # direct_chair should have: slip+footrest, tumble (2 tags total, split into slip, footrest, tumble)
        chair_tags = result.loc[result["furniture_origin_chain"] == "direct_chair"]
        assert len(chair_tags) == 3
        assert set(chair_tags["fall_tag"]) == {"slip", "footrest", "tumble"}

    def test_percentages_computation(self):
        """Test that percentages are computed correctly."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair", "direct_chair", "direct_bed"],
            "fall_tags": ["slip", "slip", "slip"],
        })
        result = _tag_origin_chain_crosstab(truth)
        chair_row = result.loc[result["furniture_origin_chain"] == "direct_chair"]
        # 2 slips in direct_chair
        assert float(chair_row.iloc[0]["count"]) == 2
        assert float(chair_row.iloc[0]["pct_within_chain"]) == 100.0
        assert float(chair_row.iloc[0]["pct_global"]) > 0

    def test_empty_input(self):
        """Test that empty DataFrame returns empty result."""
        result = _tag_origin_chain_crosstab(pd.DataFrame())
        assert result.empty

    def test_missing_columns(self):
        """Test that missing required columns returns empty result."""
        result = _tag_origin_chain_crosstab(
            pd.DataFrame({
                "furniture_origin_chain": ["direct_chair"],
                "other_col": ["a"],
            })
        )
        assert result.empty

    def test_empty_tags(self):
        """Test that empty/null tags are excluded."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair", "direct_chair"],
            "fall_tags": ["", None],
        })
        result = _tag_origin_chain_crosstab(truth)
        assert result.empty

    def test_tag_whitespace_trimmed(self):
        """Test that tag whitespace is trimmed and case normalized."""
        truth = pd.DataFrame({
            "furniture_origin_chain": ["direct_chair", "direct_chair"],
            "fall_tags": ["  SLIP  ,  TUMBLE  ", "slip"],
        })
        result = _tag_origin_chain_crosstab(truth)
        # Should have 2 distinct tags: slip and tumble
        assert len(result) == 2
        assert set(result["fall_tag"]) == {"slip", "tumble"}


# ── _reclassify_chair_origin_falls tests ──


class TestReclassifyChairOriginFalls:
    def test_basic_reclassification(self):
        """Test that matching rows with falls_in_hour > 0 are reclassified."""
        df = pd.DataFrame({
            "monitor_id": [100, 100, 200],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 10:00"]),
            "pct_chair": [0.3, 0.5, 0.1],
            "pct_bed": [0.7, 0.5, 0.9],
            "falls_in_hour": [1, 0, 1],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # Row 0 should be reclassified (matched + falls_in_hour > 0)
        assert result.loc[0, "pct_chair"] == 1.0
        assert result.loc[0, "pct_bed"] == 0.0
        # Row 1 should NOT be reclassified (falls_in_hour == 0)
        assert result.loc[1, "pct_chair"] == 0.5
        assert result.loc[1, "pct_bed"] == 0.5
        # Row 2 should NOT be reclassified (different monitor_id)
        assert result.loc[2, "pct_chair"] == 0.1
        assert result.loc[2, "pct_bed"] == 0.9

    def test_multiple_matches_reclassified(self):
        """Test that multiple matching rows are all reclassified."""
        df = pd.DataFrame({
            "monitor_id": [100, 100, 100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00", "2024-01-01 11:00"]),
            "pct_chair": [0.3, 0.4, 0.5],
            "pct_bed": [0.7, 0.6, 0.5],
            "falls_in_hour": [1, 1, 1],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [100, 100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # All three rows should be reclassified
        assert all(result["pct_chair"] == 1.0)
        assert all(result["pct_bed"] == 0.0)

    def test_empty_chair_events(self):
        """Test that empty chair_events leaves df unchanged."""
        df = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
            "pct_chair": [0.3],
            "pct_bed": [0.7],
            "falls_in_hour": [1],
        })
        result = _reclassify_chair_origin_falls(df, pd.DataFrame(columns=["monitor_id", "hour_ts"]))
        assert result.loc[0, "pct_chair"] == 0.3
        assert result.loc[0, "pct_bed"] == 0.7

    def test_no_match_in_analysis_base(self):
        """Test that unmatched chair_events leave df unchanged."""
        df = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
            "pct_chair": [0.3],
            "pct_bed": [0.7],
            "falls_in_hour": [1],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [999],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        assert result.loc[0, "pct_chair"] == 0.3
        assert result.loc[0, "pct_bed"] == 0.7

    def test_zero_falls_not_reclassified(self):
        """Test that rows with falls_in_hour <= 0 are not reclassified even if matched."""
        df = pd.DataFrame({
            "monitor_id": [100, 100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "pct_chair": [0.3, 0.5],
            "pct_bed": [0.7, 0.5],
            "falls_in_hour": [1, 0],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [100, 100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # Row 0 should be reclassified (falls_in_hour=1)
        assert result.loc[0, "pct_chair"] == 1.0
        # Row 1 should NOT be reclassified (falls_in_hour=0)
        assert result.loc[1, "pct_chair"] == 0.5

    def test_missing_columns_handled(self):
        """Test that missing required columns are handled gracefully."""
        df = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
            "pct_chair": [0.3],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # Should return df unchanged (no pct_bed column to modify)
        assert result.loc[0, "pct_chair"] == 0.3

    def test_null_monitor_id_excluded(self):
        """Test that null monitor_id values in chair_events are excluded."""
        df = pd.DataFrame({
            "monitor_id": [100, 200],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
            "pct_chair": [0.3, 0.4],
            "pct_bed": [0.7, 0.6],
            "falls_in_hour": [1, 1],
        })
        chair_events = pd.DataFrame({
            "monitor_id": [None, 100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # Only monitor_id=100 should be reclassified
        assert result.loc[0, "pct_chair"] == 1.0
        assert result.loc[1, "pct_chair"] == 0.4

    def test_timestamp_comparison_works(self):
        """Test that timestamps are properly compared even when types differ."""
        df = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00:00"]),
            "pct_chair": [0.3],
            "pct_bed": [0.7],
            "falls_in_hour": [1],
        })
        # Create chair_events with slightly different timestamp format
        chair_events = pd.DataFrame({
            "monitor_id": [100],
            "hour_ts": pd.to_datetime(["2024-01-01 10:00"]),
        })
        result = _reclassify_chair_origin_falls(df, chair_events)
        # Should still match
        assert result.loc[0, "pct_chair"] == 1.0
        assert result.loc[0, "pct_bed"] == 0.0
