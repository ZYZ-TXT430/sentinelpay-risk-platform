from __future__ import annotations

from typing import Iterable

import pandas as pd

CATEGORICAL = ["merchant_category", "country", "channel"]
NUMERIC = [
    "amount",
    "hour",
    "account_age_days",
    "velocity_1h",
    "distance_km",
    "merchant_risk",
    "is_new_device",
    "is_weekend",
]
FEATURES = NUMERIC + CATEGORICAL


def validate_input(frame: pd.DataFrame, required: Iterable[str] = FEATURES) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    validate_input(frame)
    output = frame[FEATURES].copy()
    output["log_amount"] = (output["amount"].clip(lower=0) + 1).map(lambda x: __import__("math").log(x))
    output["night_activity"] = ((output["hour"] <= 5) | (output["hour"] >= 23)).astype(int)
    output["velocity_amount_ratio"] = output["velocity_1h"] / (output["amount"] + 1)
    return output
