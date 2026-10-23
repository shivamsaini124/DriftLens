
from src.data import load_dataset
from src.model import train_model, evaluate_model

X_train, X_test, y_train, y_test = load_dataset()

model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

print("Baseline model metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
