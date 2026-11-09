from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import yaml


class ModelLoader:
    """
    Loads the model, preprocessing pipeline and label encoder from the
    verified MLflow deployment bundle in deployment/.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path
        self.deployment_dir = self.project_root / "deployment"

        self.config: dict[str, Any] = {}

        self.model = None
        self.preprocessor = None
        self.label_encoder = None

        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.model_id = None
        self.metadata: dict[str, Any] = {}

    def load_config(self) -> None:
        """Load project configuration from config.yaml."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def load_deployment_bundle(self) -> None:
        """Load model, preprocessor, label encoder and metadata from the bundle."""
        model_path = self.deployment_dir / "model.pkl"
        preprocessor_path = self.deployment_dir / "preprocessor.pkl"
        label_encoder_path = self.deployment_dir / "label_encoder.pkl"
        metadata_path = self.deployment_dir / "model_metadata.json"

        for path in (model_path, preprocessor_path, label_encoder_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Deployment artifact not found: {path}")

        print(f"[INFO] Loading deployment metadata from: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.model_name = self.metadata.get("model_name")
        self.model_version = self.metadata.get("model_version")
        self.run_id = self.metadata.get("run_id")
        self.model_id = self.metadata.get("model_id")
        self.model_uri = str(model_path)

        print(f"[INFO] Model name    : {self.model_name}")
        print(f"[INFO] Model version : {self.model_version}")
        print(f"[INFO] MLflow Run ID : {self.run_id}")
        print(f"[INFO] Model ID      : {self.model_id}")

        print(f"[INFO] Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("[SUCCESS] Model loaded successfully.")

        print(f"[INFO] Loading preprocessor from: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)
        print("[SUCCESS] Preprocessor loaded successfully.")

        print(f"[INFO] Loading label encoder from: {label_encoder_path}")
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"[SUCCESS] Label encoder loaded ({len(self.label_encoder.classes_)} classes).")

    def load(self) -> None:
        """Load configuration and deployment artifacts."""
        print("[INFO] Initializing ModelLoader...")
        self.load_config()
        self.load_deployment_bundle()
        print("[SUCCESS] ModelLoader initialization completed.")

    def is_ready(self) -> bool:
        """Return True when model, preprocessor and label encoder are loaded."""
        return all(x is not None for x in (self.model, self.preprocessor, self.label_encoder))

    @property
    def n_classes(self) -> int | None:
        return None if self.label_encoder is None else int(len(self.label_encoder.classes_))

    def predict(self, features):
        """
        Return (class_ids, labels, max_probabilities, top3) for transformed features.
        top3 is a list (one per row) of [(label, probability), ...].
        """
        probabilities = self.model.predict_proba(features)
        model_classes = self.model.classes_.astype(int)

        best = probabilities.argmax(axis=1)
        class_ids = model_classes[best]
        labels = self.label_encoder.inverse_transform(class_ids)
        max_probabilities = probabilities[np.arange(len(best)), best]

        top_idx = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
        top3 = []
        for row, indices in enumerate(top_idx):
            names = self.label_encoder.inverse_transform(model_classes[indices])
            top3.append([(str(n), float(probabilities[row, i])) for n, i in zip(names, indices)])

        return class_ids, labels, max_probabilities, top3


if __name__ == "__main__":
    loader = ModelLoader()
    try:
        loader.load()
        print("\n========== MODEL LOADER STATUS ==========")
        print(f"Model loaded         : {loader.model is not None}")
        print(f"Preprocessor loaded  : {loader.preprocessor is not None}")
        print(f"Label encoder loaded : {loader.label_encoder is not None}")
        print(f"Model name           : {loader.model_name}")
        print(f"Model version        : {loader.model_version}")
        print(f"MLflow Run ID        : {loader.run_id}")
        print(f"Model ID             : {loader.model_id}")
        print(f"Model path           : {loader.model_uri}")
        print(f"Classes              : {loader.n_classes}")
        print("Ready                :", loader.is_ready())
        print("=========================================\n")
    except Exception as exc:
        print(f"[ERROR] Model loading failed: {exc}")
        raise
