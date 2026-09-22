from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ARTIFACT_DIR
from .data import make_transactions
from .model import load_model

app = FastAPI(title="SentinelPay Risk API", version="0.1.0")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"


class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    merchant_category: str
    country: str = "CN"
    channel: str = "mobile"
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0)
    velocity_1h: int = Field(ge=0)
    is_new_device: bool = False
    distance_km: float = Field(default=0, ge=0)
    merchant_risk: float = Field(default=0.1, ge=0, le=1)
    is_weekend: int = Field(default=0, ge=0, le=1)
    user_id: str = "anonymous"
    device_id: str = "unknown"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/v1/risk/score")
def score(transaction: Transaction) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact missing. Run the training pipeline first.")
    model = load_model()
    row = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
    scored = pd.DataFrame([row])
    probability = float(model.score(scored)[0])
    action = "BLOCK" if probability >= model.threshold else ("REVIEW" if probability >= model.threshold * 0.65 else "ALLOW")
    return {
        "risk_score": round(probability, 4),
        "risk_level": "high" if action == "BLOCK" else ("medium" if action == "REVIEW" else "low"),
        "recommended_action": action,
        "threshold": model.threshold,
        "model_version": model.model_version,
        "explanations": model.explain(scored, probability),
    }
