"""US public holidays observed in New York City.

Hard-coded rather than pulled from a package: the synthetic dataset spans a known
window, and an extra dependency would have to be pinned and vendored into the
serving image for the *is_holiday* feature to be computable at inference time.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

NYC_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2024
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 10, 14),  # Columbus Day
        date(2024, 11, 5),  # Election Day
        date(2024, 11, 11),  # Veterans Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 11, 29),  # Day after Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
        date(2024, 12, 25),  # Christmas Day
        date(2024, 12, 31),  # New Year's Eve
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 10, 13),
        date(2025, 11, 11),
        date(2025, 11, 27),
        date(2025, 11, 28),
        date(2025, 12, 24),
        date(2025, 12, 25),
        date(2025, 12, 31),
    }
)


def is_holiday(timestamps: pd.Series) -> np.ndarray:
    """Boolean mask marking timestamps that fall on an observed NYC holiday."""
    dates = pd.to_datetime(timestamps).dt.date
    return dates.isin(NYC_HOLIDAYS).to_numpy()
