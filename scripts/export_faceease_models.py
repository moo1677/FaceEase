#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_MODEL = ROOT / "output" / "faceease_score_model.pkl"
DEFAULT_OUTPUT = ROOT / "public" / "models" / "faceease_models.json"


def export_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    pipeline = bundle["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    exported: dict[str, Any] = {
        "modelName": bundle["modelName"],
        "target": bundle["target"],
        "scoreDefinition": bundle["scoreDefinition"],
        "negativePatternDefinition": bundle["negativePatternDefinition"],
        "featureColumns": bundle["featureColumns"],
        "inputColumns": bundle["inputColumns"],
        "numericFeatureColumns": bundle["numericFeatureColumns"],
        "categoricalFeatureColumns": bundle["categoricalFeatureColumns"],
        "includeTaskType": bundle["includeTaskType"],
        "featureAdjustment": bundle.get("featureAdjustment", {}),
        "metrics": bundle["metrics"],
        "selectedMetrics": bundle["selectedMetrics"],
    }

    transformers = dict(preprocessor.named_transformers_)
    if "numeric" in transformers:
        numeric_pipeline = transformers["numeric"]
        scaler = numeric_pipeline.named_steps["scale"]
        exported["numericScaler"] = {
            "columns": bundle["numericFeatureColumns"],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    if "categorical" in transformers:
        categorical_pipeline = transformers["categorical"]
        encoder = categorical_pipeline.named_steps["onehot"]
        exported["categoricalCategories"] = {
            column: categories.tolist()
            for column, categories in zip(bundle["categoricalFeatureColumns"], encoder.categories_)
        }

    if bundle["modelName"] == "Ridge":
        exported["kind"] = "linear"
        exported["coef"] = model.coef_.tolist()
        exported["intercept"] = float(model.intercept_)
    elif bundle["modelName"] == "RandomForestRegressor":
        exported["kind"] = "randomForest"
        exported["trees"] = [export_regression_tree(estimator.tree_) for estimator in model.estimators_]
    elif bundle["modelName"] == "GradientBoostingRegressor":
        exported["kind"] = "gradientBoosting"
        exported["learningRate"] = float(model.learning_rate)
        exported["initialPrediction"] = float(model.init_.constant_[0][0])
        exported["trees"] = [export_regression_tree(estimator[0].tree_) for estimator in model.estimators_]
    else:
        raise ValueError(f"Unsupported model for browser export: {bundle['modelName']}")

    return exported


def export_regression_tree(tree: Any) -> dict[str, Any]:
    return {
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.reshape(tree.node_count, -1)[:, 0].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained FaceEase score model to browser JSON.")
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_model = export_bundle(args.score_model)
    payload = {
        "version": "2026-06-08",
        "scoreModel": score_model,
        "selectedModel": score_model["modelName"],
        "selectedMetrics": score_model["selectedMetrics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported browser score model to {args.output}")


if __name__ == "__main__":
    main()
