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
DEFAULT_TASK_MODEL = ROOT / "output" / "faceease_task_model.pkl"
DEFAULT_NEGATIVE_MODEL = ROOT / "output" / "faceease_negative_model.pkl"
POSITIVE_TASKS = {"natural_smile", "listening", "calm"}
FEEDBACK_THRESHOLDS = {
    "natural_smile": {
        "weakSmileMax": 0.04,
        "lowEyeBlinkMax": 0.08,
        "lowEyeBrowMax": 0.045,
        "asymmetryHigh": 0.07,
        "mouthTensionHigh": 0.25,
    },
    "listening": {
        "overSmileHigh": 0.12,
        "mouthTensionHigh": 0.11,
        "lowBlinkMax": 0.06,
        "restlessExpressionStdHigh": 0.04,
        "jawOpenHigh": 0.07,
    },
    "calm": {
        "mouthTensionHigh": 0.13,
        "browActivityHigh": 0.22,
        "asymmetryHigh": 0.05,
        "expressionActivityHigh": 0.13,
        "frameMotionStdHigh": 0.03,
    },
}


def clamp_score(value: float) -> float:
    return min(100.0, max(0.0, float(value)))


def summarize_frame_features(frame_features: list[dict[str, float]], feature_columns: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    base_keys = sorted({column.removeprefix("mean_").removeprefix("std_") for column in feature_columns})
    for key in base_keys:
        values = np.array([row[key] for row in frame_features if key in row], dtype=float)
        summary[f"mean_{key}"] = float(values.mean()) if values.size else 0.0
        summary[f"std_{key}"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def load_model_bundle(task_model_path: Path = DEFAULT_TASK_MODEL, negative_model_path: Path = DEFAULT_NEGATIVE_MODEL):
    return {
        "task": joblib.load(task_model_path),
        "negative": joblib.load(negative_model_path),
    }


def predict_from_summary(
    task_type: str,
    summary_features: dict[str, float],
    task_model_path: Path = DEFAULT_TASK_MODEL,
    negative_model_path: Path = DEFAULT_NEGATIVE_MODEL,
) -> dict[str, Any]:
    bundles = load_model_bundle(task_model_path, negative_model_path)
    task_bundle = bundles["task"]
    negative_bundle = bundles["negative"]

    task_row = {"taskType": task_type}
    task_row.update({column: summary_features.get(column, 0.0) for column in task_bundle["featureColumns"]})
    negative_row = {
        column: summary_features.get(column, 0.0) for column in negative_bundle["featureColumns"]
    }

    predicted_task = clamp_score(task_bundle["model"].predict(pd.DataFrame([task_row]))[0])
    predicted_negative = clamp_score(negative_bundle["model"].predict(pd.DataFrame([negative_row]))[0])

    if task_type in POSITIVE_TASKS:
        final_naturalness = clamp_score(predicted_task * 0.7 + (100.0 - predicted_negative) * 0.3)
    else:
        final_naturalness = None

    feedback = generate_feedback(
        task_type=task_type,
        predicted_task_performance_score=predicted_task,
        predicted_negative_expression_score=predicted_negative,
        final_naturalness_score=final_naturalness,
        summary_features=summary_features,
    )
    return {
        "taskType": task_type,
        "predictedTaskPerformanceScore": round(predicted_task, 3),
        "predictedNegativeExpressionScore": round(predicted_negative, 3),
        "finalNaturalnessScore": round(final_naturalness, 3) if final_naturalness is not None else None,
        "feedback": feedback,
    }


def predict_from_frame_features(
    task_type: str,
    frame_features: list[dict[str, float]],
    task_model_path: Path = DEFAULT_TASK_MODEL,
    negative_model_path: Path = DEFAULT_NEGATIVE_MODEL,
) -> dict[str, Any]:
    task_bundle = joblib.load(task_model_path)
    summary = summarize_frame_features(frame_features, task_bundle["featureColumns"])
    return predict_from_summary(task_type, summary, task_model_path, negative_model_path)


def generate_feedback(
    task_type: str,
    predicted_task_performance_score: float,
    predicted_negative_expression_score: float,
    final_naturalness_score: float | None,
    summary_features: dict[str, float],
) -> list[str]:
    mean = lambda key: float(summary_features.get(f"mean_{key}", 0.0))
    std = lambda key: float(summary_features.get(f"std_{key}", 0.0))
    feedback: list[str] = []

    if (
        task_type in POSITIVE_TASKS
        and final_naturalness_score is not None
        and final_naturalness_score >= 75
        and predicted_task_performance_score >= 75
        and predicted_negative_expression_score <= 35
    ):
        return ["현재 표정은 선택한 과제와 비교적 잘 맞고 부정 표정 신호도 낮게 나타났습니다."]

    if predicted_task_performance_score < 55:
        feedback.append("요청된 표정 과제 수행도가 낮아 목표 표정이 명확하게 전달되지 않을 수 있습니다.")
    if predicted_negative_expression_score > 65:
        feedback.append("어색하거나 과하게 웃는 부정 표정 패턴과 유사도가 높게 나타났습니다.")

    if task_type == "natural_smile":
        thresholds = FEEDBACK_THRESHOLDS["natural_smile"]
        if mean("smile") < thresholds["weakSmileMax"]:
            feedback.append("입꼬리 움직임이 부족해 미소가 약하게 보일 수 있습니다.")
        if mean("blink") < thresholds["lowEyeBlinkMax"] and mean("browActivity") < thresholds["lowEyeBrowMax"]:
            feedback.append("눈가 움직임이 적어 다소 딱딱한 미소로 보일 수 있습니다.")
        if mean("asymmetry") > thresholds["asymmetryHigh"]:
            feedback.append("좌우 미소 균형이 불안정해 어색하게 보일 수 있습니다.")
        if mean("mouthTension") > thresholds["mouthTensionHigh"]:
            feedback.append("입 주변 긴장도가 높아 자연스러운 미소보다 굳은 표정으로 보일 수 있습니다.")
    elif task_type == "listening":
        thresholds = FEEDBACK_THRESHOLDS["listening"]
        if std("expressionActivity") > thresholds["restlessExpressionStdHigh"]:
            feedback.append("표정 변화가 커서 안정적인 경청 표정보다 산만해 보일 수 있습니다.")
        if mean("smile") > thresholds["overSmileHigh"]:
            feedback.append("미소가 과하게 나타나 경청 표정보다는 웃는 표정에 가깝게 보일 수 있습니다.")
        if mean("mouthTension") > thresholds["mouthTensionHigh"]:
            feedback.append("입 주변 긴장도가 높아 다소 굳어 보일 수 있습니다.")
        if mean("blink") < thresholds["lowBlinkMax"]:
            feedback.append("눈가 움직임이 너무 적으면 반응이 부족해 보일 수 있습니다.")
        if mean("jawOpen") > thresholds["jawOpenHigh"]:
            feedback.append("입이 벌어진 시간이 길어 경청 표정보다 말하거나 놀란 표정에 가깝게 보일 수 있습니다.")
    elif task_type == "calm":
        thresholds = FEEDBACK_THRESHOLDS["calm"]
        if mean("mouthTension") > thresholds["mouthTensionHigh"]:
            feedback.append("입 주변 긴장도가 높아 차분하기보다 굳어 보일 수 있습니다.")
        if mean("browActivity") > thresholds["browActivityHigh"]:
            feedback.append("눈썹 움직임이 많아 불안정한 인상으로 보일 수 있습니다.")
        if mean("asymmetry") > thresholds["asymmetryHigh"]:
            feedback.append("좌우 표정 차이가 커서 표정이 어색하게 보일 수 있습니다.")
        if (
            mean("expressionActivity") > thresholds["expressionActivityHigh"]
            or std("frameMotion") > thresholds["frameMotionStdHigh"]
        ):
            feedback.append("표정 변화가 지나치게 크면 차분한 기본 표정과 다르게 보일 수 있습니다.")
    elif task_type in {"awkward", "over_smile"}:
        feedback.append("awkward와 over_smile은 서비스 권장 과제가 아니라 부정 패턴 학습용 taskType입니다.")

    if not feedback and final_naturalness_score is not None:
        feedback.append("현재 표정은 선택한 과제와 비교적 잘 맞고 부정 표정 신호도 낮게 나타났습니다.")
    return feedback[:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FaceEase service inference from summary feature JSON.")
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--task-model", type=Path, default=DEFAULT_TASK_MODEL)
    parser.add_argument("--negative-model", type=Path, default=DEFAULT_NEGATIVE_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    result = predict_from_summary(args.task_type, summary, args.task_model, args.negative_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
