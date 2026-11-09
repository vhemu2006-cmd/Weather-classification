
"""Lab 12 - Class-wise evaluation of V1 and V2 on evolved data."""

import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from monitoring_common import (
    EVOLVED_LABELED_PATH,
    TARGET,
    load_production_loader,
)

from weather_common import (
    clean_dataframe,
    load_config,
    project_path,
)

V2_MODEL = project_path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR = project_path("models/retrained_preprocessor_v2.pkl")
V2_ENCODER = project_path("models/retrained_label_encoder_v2.pkl")

OUTPUT_DIR = project_path("outputs/retraining/classwise")


def evaluate_model(name, model, preprocessor, encoder, X, y, config):

    X_clean = clean_dataframe(X.copy(), config)
    X_transformed = preprocessor.transform(X_clean)

    y_true = encoder.transform(y)
    y_pred = model.predict(X_transformed)

    labels = list(range(len(encoder.classes_)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / f"{name.lower()}_classification_report.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_,
    )

    cm_path = OUTPUT_DIR / f"{name.lower()}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    print(f"\n{name} CLASS-WISE RESULTS")
    print("-" * 60)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

    # Show 10 classes with the lowest F1 score among supported classes
    class_results = report_df.loc[
        report_df.index.isin(encoder.classes_)
    ].copy()

    class_results = class_results[
        class_results["support"] > 0
    ].sort_values("f1-score")

    print("\n10 LOWEST F1-SCORE CLASSES")
    print(class_results.head(10).to_string())

    print(f"\nClassification report: {report_path}")
    print(f"Confusion matrix: {cm_path}")

    return report_df


def main():

    print("=" * 70)
    print("V1 vs V2 CLASS-WISE EVALUATION")
    print("=" * 70)

    config = load_config()

    # Load evolved labelled dataset
    evolved = pd.read_csv(EVOLVED_LABELED_PATH)

    X = evolved.drop(columns=[TARGET])
    y = evolved[TARGET].to_numpy()

    # Load V1
    loader = load_production_loader()

    # Load V2
    model_v2 = joblib.load(V2_MODEL)
    preprocessor_v2 = joblib.load(V2_PREPROCESSOR)
    encoder_v2 = joblib.load(V2_ENCODER)

    print(f"\nEvolved dataset shape: {evolved.shape}")
    print(f"Number of classes: {len(loader.label_encoder.classes_)}")

    # Evaluate V1
    report_v1 = evaluate_model(
        "V1",
        loader.model,
        loader.preprocessor,
        loader.label_encoder,
        X,
        y,
        config,
    )

    # Evaluate V2
    report_v2 = evaluate_model(
        "V2",
        model_v2,
        preprocessor_v2,
        encoder_v2,
        X,
        y,
        config,
    )

    # Compare per-class F1 scores
    comparison = pd.DataFrame({
        "V1_F1": report_v1.loc[
            loader.label_encoder.classes_, "f1-score"
        ],
        "V2_F1": report_v2.loc[
            encoder_v2.classes_, "f1-score"
        ],
    })

    comparison["F1_Improvement"] = (
        comparison["V2_F1"] - comparison["V1_F1"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "v1_v2_classwise_comparison.csv"
    )

    print("\n" + "=" * 70)
    print("CLASS-WISE COMPARISON COMPLETED")
    print("=" * 70)

    print("\nTOP 10 IMPROVED CLASSES")
    print(
        comparison.sort_values(
            "F1_Improvement", ascending=False
        ).head(10).to_string()
    )

    print("\nTOP 10 DEGRADED CLASSES")
    print(
        comparison.sort_values(
            "F1_Improvement"
        ).head(10).to_string()
    )


if __name__ == "__main__":
    main()
