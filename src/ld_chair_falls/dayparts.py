from __future__ import annotations

from typing import Final

import pandas as pd

DAYPART_SPECS: Final[tuple[tuple[str, int, int, str], ...]] = (
    ("window_00_05", 0, 5, "00:00-05:59"),
    ("window_06_08", 6, 8, "06:00-08:59"),
    ("window_09_11", 9, 11, "09:00-11:59"),
    ("window_12_14", 12, 14, "12:00-14:59"),
    ("window_15_17", 15, 17, "15:00-17:59"),
    ("window_18_20", 18, 20, "18:00-20:59"),
    ("window_21_23", 21, 23, "21:00-23:59"),
)

DAYPART_ORDER: Final[list[str]] = [spec[0] for spec in DAYPART_SPECS] + ["unknown"]
DAYPART_DISPLAY: Final[dict[str, str]] = {
    **{spec[0]: spec[3] for spec in DAYPART_SPECS},
    "unknown": "Unknown",
}
DAYPART_SENSITIVITY_LABELS: Final[dict[str, str]] = {
    spec[0]: f"timewindow_{spec[0].removeprefix('window_')}" for spec in DAYPART_SPECS
}


def classify_hour(hour: float | int | None) -> str:
    if pd.isna(hour):
        return "unknown"
    hour_int = int(hour)
    for key, start_hour, end_hour, _display in DAYPART_SPECS:
        if start_hour <= hour_int <= end_hour:
            return key
    return "unknown"
