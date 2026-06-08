#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_MODEL = ROOT / "output" / "faceease_score_model.pkl"


def clamp_score(value: float) -> float:
    return min(100.0, max(0.0, float(value)))


def summarize_frame_features(frame_features: list[dict[str, float]], feature_columns: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    base_keys = sorted(
        {
            column.removeprefix("mean_").removeprefix("std_")
            for column in feature_columns
            if column.startswith(("mean_", "std_"))
        }
    )
    for key in base_keys:
        values = np.array([row[key] for row in frame_features if key in row], dtype=float)
        summary[f"mean_{key}"] = float(values.mean()) if values.size else 0.0
        summary[f"std_{key}"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def load_model_bundle(score_model_path: Path = DEFAULT_SCORE_MODEL) -> dict[str, Any]:
    return joblib.load(score_model_path)


def predict_from_summary(
    task_type: str,
    summary_features: dict[str, float],
    score_model_path: Path = DEFAULT_SCORE_MODEL,
) -> dict[str, Any]:
    bundle = load_model_bundle(score_model_path)
    row = {column: summary_features.get(column, 0.0) for column in bundle["featureColumns"]}
    if "taskType" in bundle["featureColumns"]:
        row["taskType"] = task_type

    predicted_score = clamp_score(bundle["model"].predict(pd.DataFrame([row]))[0])
    return {
        "taskType": task_type,
        "target": bundle["target"],
        "selectedModel": bundle["modelName"],
        "predictedScore": round(predicted_score, 3),
        "scoreDefinition": bundle["scoreDefinition"],
        "negativePatternDefinition": bundle["negativePatternDefinition"],
        "feedback": generate_feedback(task_type, predicted_score, summary_features),
    }


def predict_from_frame_features(
    task_type: str,
    frame_features: list[dict[str, float]],
    score_model_path: Path = DEFAULT_SCORE_MODEL,
) -> dict[str, Any]:
    bundle = load_model_bundle(score_model_path)
    summary = summarize_frame_features(frame_features, bundle["featureColumns"])
    return predict_from_summary(task_type, summary, score_model_path)


def generate_feedback(
    task_type: str,
    predicted_score: float,
    summary_features: dict[str, float],
) -> list[str]:
    mean = lambda key: float(summary_features.get(f"mean_{key}", 0.0))
    feedback: list[str] = []

    if predicted_score >= 80:
        feedback.append("최종 자연스러움 score가 높게 예측되었습니다.")
    elif predicted_score >= 60:
        feedback.append("최종 자연스러움 score가 보통 이상으로 예측되었습니다.")
    elif predicted_score >= 40:
        feedback.append("최종 자연스러움 score가 다소 낮아 개선 여지가 있습니다.")
    else:
        feedback.append("최종 자연스러움 score가 낮게 예측되었습니다.")

    if task_type in {"awkward", "over_smile"}:
        feedback.append("이 taskType은 어색한표정 또는 과하게웃는표정 부정 패턴 계열입니다.")
    if mean("asymmetry") > 0.07:
        feedback.append("좌우 표정 차이가 커서 자연스러움 score를 낮출 수 있습니다.")
    if mean("mouthTension") > 0.2:
        feedback.append("입 주변 긴장도가 높아 어색한 인상으로 보일 수 있습니다.")
    if mean("smile") > 0.18 and task_type in {"listening", "calm"}:
        feedback.append("미소가 과하게 나타나 요구 과제와 다르게 보일 수 있습니다.")

    return feedback[:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FaceEase score regression inference from summary feature JSON.")
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    result = predict_from_summary(args.task_type, summary, args.score_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
