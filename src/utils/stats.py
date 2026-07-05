from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# Two-sided 95% Student's t critical values: t_{0.975, df}
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        return math.nan
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    return 1.96


def mean_t_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, 0.0, 0.0
    std = float(np.std(data, ddof=1))
    se = std / math.sqrt(data.size)
    ci = t_critical_95(int(data.size - 1)) * se
    return mean, std, float(ci)
