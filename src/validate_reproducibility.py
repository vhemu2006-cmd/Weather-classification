"""Lab 4 - verify that two identical training runs give identical scores."""
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import ARTIFACTS_DIR, RF_PARAMS, load_xy_arrays


def run_deterministic_test():
    print("Running Reproducibility Validation...")
    X_train, X_test, y_train, y_test = load_xy_arrays()
    params = dict(RF_PARAMS)

    scores = []
    for _ in range(2):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(f1_score(y_test, model.predict(X_test), average="weighted", zero_division=0))

    is_reproducible = scores[0] == scores[1]
    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_f1_weighted": scores[0],
        "execution_2_f1_weighted": scores[1],
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED",
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "reproducibility_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )

    print(f"Execution 1 F1: {scores[0]:.6f}")
    print(f"Execution 2 F1: {scores[1]:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_deterministic_test()
