#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "public" / "models" / "faceease_models.json"


def export_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    pipeline = bundle["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    exported: dict[str, Any] = {
        "modelName": bundle["modelName"],
        "target": bundle["target"],
        "featureColumns": bundle["featureColumns"],
        "inputColumns": bundle["inputColumns"],
        "includeTaskType": bundle["includeTaskType"],
        "excludeReviewNeeded": bundle["excludeReviewNeeded"],
        "metrics": bundle["metrics"],
    }

    transformers = dict(preprocessor.named_transformers_)
    feature_scaler = transformers["features"]
    exported["featureScaler"] = {
        "mean": feature_scaler.mean_.tolist(),
        "scale": feature_scaler.scale_.tolist(),
    }

    if bundle["includeTaskType"]:
        task_encoder = transformers["taskType"]
        exported["taskTypeCategories"] = task_encoder.categories_[0].tolist()

    if bundle["modelName"] == "Ridge":
        exported["kind"] = "linear"
        exported["coef"] = model.coef_.tolist()
        exported["intercept"] = float(model.intercept_)
    elif bundle["modelName"] == "RandomForestRegressor":
        exported["kind"] = "randomForest"
        exported["trees"] = [export_tree(estimator.tree_) for estimator in model.estimators_]
    elif bundle["modelName"] == "GradientBoostingRegressor":
        exported["kind"] = "gradientBoosting"
        exported["learningRate"] = float(model.learning_rate)
        exported["initialPrediction"] = float(model.init_.constant_[0][0])
        exported["trees"] = [export_tree(estimator[0].tree_) for estimator in model.estimators_]
    else:
        raise ValueError(f"Unsupported model for browser export: {bundle['modelName']}")

    return exported


def export_tree(tree: Any) -> dict[str, Any]:
    return {
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.reshape(tree.node_count, -1)[:, 0].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained FaceEase sklearn models to browser JSON.")
    parser.add_argument("--task-model", type=Path, default=ROOT / "output" / "faceease_task_model.pkl")
    parser.add_argument("--negative-model", type=Path, default=ROOT / "output" / "faceease_negative_model.pkl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "version": "2026-06-07",
        "taskModel": export_bundle(args.task_model),
        "negativeModel": export_bundle(args.negative_model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported browser models to {args.output}")


if __name__ == "__main__":
    main()
