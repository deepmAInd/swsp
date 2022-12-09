import sys
from typing import List
import pandas as pd

sys.setrecursionlimit(4000)

intervals = pd.read_csv("./scripts/IBI.csv")[" IBI"].to_numpy() * 1000


# def find_local_maxima(bvp: List[int]) -> List[int]:
#     return [i for i in range(1, len(bvp) - 1) if bvp[i - 1] < bvp[i] > bvp[i + 1]]


# def calc_intervals(bvp: List[int]) -> List[int]:
#     return [(1 / (abs(bvp[i] - bvp[i + 1]) * 64)) * 1000 for i in range(len(bvp) - 1)]


def clean_intervals(intervals: List[int]) -> List[int]:
    return [
        intervals[i]
        for i in range(1, len(intervals))
        if ((intervals[i - 1] + (intervals[i - 1] * 0.2)) >= intervals[i])
        and (200 < intervals[i] < 1200)
    ]


def _sum(intervals: List[int]) -> float:
    return (
        ((intervals[0] - intervals[1]) ** 2) + _sum(intervals[1:])
        if len(intervals) > 1
        else 0
    )


def rmssd(intervals: List[int]) -> float:
    total = _sum(intervals)
    avg = total / (len(intervals) - 1)
    return avg**0.5


intervals = clean_intervals(intervals)
print(rmssd(intervals))
