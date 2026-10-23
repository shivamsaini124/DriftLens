from src.data import load_dataset
from src.model import train_model, evaluate_model


def test_train_and_evaluate_model():
    X_train, X_test, y_train, y_test = load_dataset()

    model = train_model(X_train, y_train, random_state=42)
    metrics = evaluate_model(model, X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1", "f1_score"}
    assert expected_keys.issubset(metrics.keys())

    # Baseline performance check on held-out test data
    assert metrics["accuracy"] > 0.90
    assert metrics["f1"] > 0.90
    assert metrics["precision"] > 0.90
    assert metrics["recall"] > 0.90
