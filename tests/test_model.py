from sentinelpay.data import make_transactions
from sentinelpay.model import train_model


def test_training_returns_business_metrics():
    frame = make_transactions(3000)
    model, metrics = train_model(frame)
    assert 0 < metrics["pr_auc"] <= 1
    assert 0 < model.threshold < 1
    assert metrics["split"]["train"] < len(frame)


def test_model_scores_probability():
    frame = make_transactions(1500)
    model, _ = train_model(frame)
    result = model.score(frame.iloc[:5])
    assert len(result) == 5
    assert ((result >= 0) & (result <= 1)).all()
