from __future__ import annotations

import argparse
import json

from .config import DATA_DIR, REPORT_DIR
from .data import make_transactions
from .model import save_model, train_model


def run(n_transactions: int = 80_000) -> dict:
    frame = make_transactions(n_transactions)
    try:
        frame.to_parquet(DATA_DIR / "transactions.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        frame.to_csv(DATA_DIR / "transactions.csv", index=False)
    model, metrics = train_model(frame)
    save_model(model, metrics)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentinelPay risk model")
    parser.add_argument("--n-transactions", type=int, default=80_000)
    args = parser.parse_args()
    run(args.n_transactions)
