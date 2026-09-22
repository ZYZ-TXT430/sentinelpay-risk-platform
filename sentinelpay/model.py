from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTIFACT_DIR, RANDOM_SEED
from .features import CATEGORICAL, NUMERIC, build_features


@dataclass
class RiskModel:
    pipeline: Pipeline
    threshold: float
    feature_names: list[str]
    model_version: str = "sentinelpay-2024.04"

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(build_features(rows))[:, 1]

    def explain(self, row: pd.DataFrame, probability: float) -> list[dict[str, Any]]:
        raw = row.iloc[0]
        factors = []
        if raw["amount"] > 700:
            factors.append(("金额显著偏高", float(min(raw["amount"] / 700, 5) / 5)))
        if raw["is_new_device"]:
            factors.append(("新设备交易", 0.72))
        if raw["hour"] <= 5 or raw["hour"] >= 23:
            factors.append(("深夜时段", 0.64))
        if raw["velocity_1h"] >= 5:
            factors.append(("一小时内频次较高", 0.68))
        if raw["country"] != "CN":
            factors.append(("跨境交易", 0.42))
        if raw["merchant_risk"] > 0.35:
            factors.append(("商户历史风险较高", float(raw["merchant_risk"])))
        if not factors:
            factors.append(("未发现明显高风险信号", 1 - probability))
        return [{"factor": name, "contribution": round(value, 3)} for name, value in factors[:4]]


def _make_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC + ["log_amount", "night_activity", "velocity_amount_ratio"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse=False), CATEGORICAL),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=180, max_depth=14, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predicted = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        # Business cost: missed fraud is expensive; blocking a good customer also hurts.
        cost = fn * 80 + fp * 3
        rows.append((cost, threshold, fp, fn, tp, tn))
    cost, threshold, fp, fn, tp, tn = min(rows)
    metrics = {"threshold": float(threshold), "false_positive": int(fp), "false_negative": int(fn),
               "true_positive": int(tp), "true_negative": int(tn), "business_cost": int(cost)}
    return float(threshold), metrics


def train_model(frame: pd.DataFrame) -> tuple[RiskModel, dict[str, Any]]:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    first = int(len(frame) * 0.70)
    second = int(len(frame) * 0.85)
    train, valid, test = frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]
    pipeline = _make_pipeline()
    pipeline.fit(build_features(train), train["is_fraud"])
    valid_scores = pipeline.predict_proba(build_features(valid))[:, 1]
    threshold, threshold_metrics = choose_threshold(valid["is_fraud"], valid_scores)
    test_scores = pipeline.predict_proba(build_features(test))[:, 1]
    predictions = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(test["is_fraud"], predictions, labels=[0, 1]).ravel()
    metrics = {
        "split": {"train": len(train), "validation": len(valid), "test": len(test)},
        "fraud_rate_test": round(float(test["is_fraud"].mean()), 4),
        "roc_auc": round(float(roc_auc_score(test["is_fraud"], test_scores)), 4),
        "pr_auc": round(float(average_precision_score(test["is_fraud"], test_scores)), 4),
        "threshold": round(threshold, 2),
        "recall": round(float(tp / max(tp + fn, 1)), 4),
        "precision": round(float(tp / max(tp + fp, 1)), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "threshold_selection": threshold_metrics,
    }
    model = RiskModel(pipeline, threshold, list(build_features(frame).columns))
    return model, metrics


def save_model(model: RiskModel, metrics: dict[str, Any]) -> None:
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model() -> RiskModel:
    return joblib.load(ARTIFACT_DIR / "model.joblib")
