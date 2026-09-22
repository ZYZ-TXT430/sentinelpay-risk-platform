import pandas as pd
import pytest

from sentinelpay.features import build_features


def test_feature_builder_adds_behavioral_features():
    row = pd.DataFrame([{
        "amount": 100, "hour": 2, "account_age_days": 30, "velocity_1h": 4,
        "distance_km": 12, "merchant_risk": .2, "is_new_device": True,
        "is_weekend": 0, "merchant_category": "electronics", "country": "CN", "channel": "mobile",
    }])
    features = build_features(row)
    assert "log_amount" in features
    assert features.loc[0, "night_activity"] == 1
    assert features.loc[0, "velocity_amount_ratio"] > 0


def test_feature_builder_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required features"):
        build_features(pd.DataFrame([{"amount": 10}]))
