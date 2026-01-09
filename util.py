

from datetime import datetime


def normalize_timestamp(x):
    x = x - x.iloc[0]
    return datetime(year=2023, month=1, day=1) + x