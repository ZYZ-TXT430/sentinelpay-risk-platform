from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED


def make_transactions(n: int = 80_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a deterministic, realistic-looking payment event stream."""
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="7min")
    merchant_categories = np.array(
        ["grocery", "electronics", "fashion", "travel", "gaming", "digital_goods"]
    )
    countries = np.array(["CN", "US", "GB", "JP", "SG", "AU"])
    channels = np.array(["mobile", "web", "pos", "api"])
    category = rng.choice(merchant_categories, n, p=[.28, .16, .18, .08, .12, .18])
    country = rng.choice(countries, n, p=[.62, .10, .07, .08, .07, .06])
    channel = rng.choice(channels, n, p=[.47, .31, .16, .06])
    amount = np.clip(rng.lognormal(4.0, 1.0, n), 2, 12_000).round(2)
    hour = np.asarray(timestamps.hour)
    account_age = np.clip(rng.gamma(3.2, 120, n).astype(int), 1, 4000)
    velocity = rng.poisson(1.7, n) + (channel == "api") * rng.poisson(1.2, n)
    new_device = rng.binomial(1, np.clip(.08 + 250 / (account_age + 500), .08, .48))
    distance = np.clip(rng.gamma(2.0, 70, n), 0, 900).round(1)
    merchant_risk = rng.beta(2, 9, n)
    weekend = (timestamps.dayofweek >= 5).astype(int)

    # Log-odds includes non-linear behavior and a mild concept drift after March.
    logit = (
        -5.0
        + 0.00011 * amount
        + 0.35 * (amount > 700)
        + 0.55 * (hour <= 5)
        + 0.45 * (hour >= 23)
        + 0.50 * new_device
        + 0.16 * velocity
        + 0.003 * distance
        + 1.8 * merchant_risk
        + 0.45 * (country != "CN")
        + 0.35 * (channel == "api")
        + 0.40 * (category == "digital_goods")
        + 0.25 * weekend
        + 0.18 * (timestamps >= "2024-03-01")
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(n)],
            "timestamp": timestamps,
            "user_id": [f"U{v:05d}" for v in rng.randint(0, max(1000, n // 20), n)],
            "merchant_id": [f"M{v:04d}" for v in rng.randint(0, 800, n)],
            "device_id": [f"D{v:05d}" for v in rng.randint(0, 4000, n)],
            "amount": amount,
            "merchant_category": category,
            "country": country,
            "channel": channel,
            "hour": hour,
            "account_age_days": account_age,
            "velocity_1h": velocity,
            "is_new_device": new_device.astype(bool),
            "distance_km": distance,
            "merchant_risk": merchant_risk.round(4),
            "is_weekend": weekend,
            "is_fraud": fraud,
        }
    )
