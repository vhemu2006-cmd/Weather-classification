"""Lab 3 - baseline evaluation: metrics, confusion matrix, CV, feature importance."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import (MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_STATE,
                        TARGET_ENCODED, TEST_SIZE, classification_metrics)

DATA = PROCESSED_DIR / "processed_weather.csv"
MODEL = MODELS_DIR / "random_forest_model.pkl"


def evaluate():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET_ENCODED])
    y = df[TARGET_ENCODED]
    model = joblib.load(MODEL)
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    # Recreate the training split.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = model.predict(X_test)

    metrics = classification_metrics(y_test, y_pred)
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        "\n".join(f"{k}: {v:.6f}" for k, v in metrics.items()), encoding="utf-8"
    )

    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix - Random Forest (Daily Summary)")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 3-fold CV on a smaller forest of the same family (keeps runtime reasonable).
    cv_model = clone(model).set_params(n_estimators=20, n_jobs=1)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(cv_model, X, y, cv=skf, scoring="accuracy")
    (OUTPUTS_DIR / "cross_validation_results.txt").write_text(
        "Validation Scores: " + str(scores.tolist()) + "\n"
        "Average Validation Accuracy: " + str(scores.mean()), encoding="utf-8"
    )

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values(ascending=False).to_csv(
        OUTPUTS_DIR / "feature_importance.csv", header=["importance"]
    )
    return metrics["accuracy"], scores.mean()


if __name__ == "__main__":
    accuracy, cv_accuracy = evaluate()
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Average Validation Accuracy: {cv_accuracy:.6f}")
