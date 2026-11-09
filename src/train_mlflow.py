"""Lab 4 - Random Forest training with MLflow experiment tracking."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import (ARTIFACTS_DIR, CATEGORICAL, MLFLOW_EXPERIMENT, MODELS_DIR,
                        NUMERIC, PROCESSED_DIR, RF_PARAMS, classification_metrics,
                        load_xy_arrays)


def top_k_accuracy(y_true, y_proba, classes, k=3):
    topk = classes[np.argsort(y_proba, axis=1)[:, -k:]]
    return float(np.mean([t in row for t, row in zip(y_true, topk)]))


def train_and_track(run_name="RandomForest_Baseline", params=None):
    params = dict(RF_PARAMS if params is None else params)
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    X_train, X_test, y_train, y_test = load_xy_arrays()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = classification_metrics(y_test, y_pred)
        metrics["top3_accuracy"] = top_k_accuracy(y_test, y_proba, model.classes_, 3)
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: Accuracy = {metrics['accuracy']:.4f} | "
              f"F1 (weighted) = {metrics['f1_score']:.4f}")

        ARTIFACTS_DIR.mkdir(exist_ok=True)

        # Top-20 most important features
        ohe_path = MODELS_DIR / "ohe.pkl"
        if ohe_path.exists():
            ohe = joblib.load(ohe_path)
            names = np.array(NUMERIC + list(ohe.get_feature_names_out(CATEGORICAL)))
        else:
            names = np.array([f"f{i}" for i in range(X_train.shape[1])])
        order = np.argsort(model.feature_importances_)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(names[order][::-1], model.feature_importances_[order][::-1])
        ax.set_title(f"Top-20 Feature Importances - {run_name}")
        fi_path = ARTIFACTS_DIR / "feature_importance.png"
        fig.savefig(fi_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(fi_path), artifact_path="plots")

        meta = PROCESSED_DIR / "dataset_metadata.json"
        if meta.exists():
            mlflow.log_artifact(str(meta), artifact_path="metadata")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        joblib.dump(model, MODELS_DIR / "random_forest_model.pkl", compress=3)
        print(f"Run '{run_name}' successfully tracked!")


if __name__ == "__main__":
    # Baseline run
    train_and_track(run_name="RandomForest")

    # Optional extra experiment (uncomment to compare runs in the MLflow UI):
    # train_and_track(
    #     run_name="RandomForest_Shallow",
    #     params={"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10,
    #             "random_state": 42, "n_jobs": -1},
    # )
