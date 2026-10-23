from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def train_model(X_train, y_train, random_state: int = 42):
    """Train the baseline Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    predictions = model.predict(X_test)
    f1_val = float(f1_score(y_test, predictions, zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": f1_val,
        "f1_score": f1_val,
    }
