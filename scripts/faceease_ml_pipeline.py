#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data_set"
DEFAULT_LABEL_FILE = ROOT / "Labeling_scaled_reworked_v2.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "output"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

TASK_SUFFIXES = {
    "자연스러운미소": "natural_smile",
    "과하게웃는표정": "over_smile",
    "경청하는표정": "listening",
    "차분한표정": "calm",
    "어색한표정": "awkward",
}
POSITIVE_TASKS = {"natural_smile", "listening", "calm"}
NEGATIVE_TASKS = {"awkward", "over_smile"}

FRAME_FEATURE_KEYS = [
    "expressionActivity",
    "frameMotion",
    "motionAverage",
    "asymmetry",
    "mouthTension",
    "blink",
    "smile",
    "jawOpen",
    "browActivity",
    "baselineNaturalness",
    "baselineRigidity",
]
TRACKED_MOTION_KEYS = [
    "mouthSmileLeft",
    "mouthSmileRight",
    "jawOpen",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "browInnerUp",
    "browDownLeft",
    "browDownRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "cheekSquintLeft",
    "cheekSquintRight",
]
LABEL_COLUMNS = [
    "userId",
    "videoName",
    "taskType",
    "taskPerformanceScore",
    "negativeExpressionScore",
    "finalNaturalnessScore",
    "rigidityScore",
    "reviewNeeded",
]


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def join_key(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_video_name(path: Path) -> dict[str, str]:
    video_name = normalize_text(path.name)
    stem = normalize_text(path.stem)
    if "_" not in stem:
        raise ValueError(f"Cannot parse video name without underscore: {path.name}")

    user_id, expression_name = stem.split("_", 1)
    user_id = normalize_text(user_id)
    expression_name = normalize_text(expression_name).replace(" ", "")
    task_type = TASK_SUFFIXES.get(expression_name)
    if not task_type:
        raise ValueError(f"Unknown task suffix '{expression_name}' in {path.name}")

    return {
        "userId": user_id,
        "videoName": video_name,
        "taskType": task_type,
        "joinKey": join_key(video_name),
    }


def score(blendshapes: dict[str, float], key: str) -> float:
    return float(blendshapes.get(key, 0.0))


def average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def round3(value: float) -> float:
    return round(float(value), 3)


@dataclass
class ExpressionFeatureCalculator:
    previous_blendshape_map: dict[str, float] | None = None
    motion_window: list[float] = field(default_factory=list)

    def calculate(self, blendshapes: dict[str, float]) -> dict[str, float]:
        smile_left = score(blendshapes, "mouthSmileLeft")
        smile_right = score(blendshapes, "mouthSmileRight")
        jaw_open = score(blendshapes, "jawOpen")
        mouth_press = average(
            [score(blendshapes, "mouthPressLeft"), score(blendshapes, "mouthPressRight")]
        )
        mouth_pucker = score(blendshapes, "mouthPucker")
        mouth_funnel = score(blendshapes, "mouthFunnel")
        brow_up = score(blendshapes, "browInnerUp")
        brow_down = average([score(blendshapes, "browDownLeft"), score(blendshapes, "browDownRight")])
        eye_blink_left = score(blendshapes, "eyeBlinkLeft")
        eye_blink_right = score(blendshapes, "eyeBlinkRight")

        frame_motion = self.get_frame_motion(blendshapes)
        self.motion_window.append(frame_motion)
        if len(self.motion_window) > 24:
            self.motion_window.pop(0)

        smile = average([smile_left, smile_right])
        blink = average([eye_blink_left, eye_blink_right])
        mouth_tension = clamp01(mouth_press * 0.75 + mouth_pucker * 0.15 + mouth_funnel * 0.1)
        asymmetry = clamp01(
            abs(smile_left - smile_right) * 0.55
            + abs(eye_blink_left - eye_blink_right) * 0.25
            + abs(score(blendshapes, "browOuterUpLeft") - score(blendshapes, "browOuterUpRight")) * 0.2
        )
        brow_activity = clamp01(brow_up * 0.55 + brow_down * 0.45)
        expression_activity = clamp01(
            smile * 0.3 + jaw_open * 0.2 + brow_activity * 0.2 + blink * 0.12 + mouth_tension * 0.18
        )
        motion_average = average(self.motion_window)

        stillness = 1 - clamp01(motion_average / 0.035)
        tension_signal = clamp01(mouth_tension * 0.65 + brow_down * 0.2 + asymmetry * 0.15)
        baseline_rigidity = round(clamp01(stillness * 0.58 + tension_signal * 0.42) * 100)
        baseline_naturalness = round(
            clamp01(0.72 - baseline_rigidity / 170 + expression_activity * 0.28 - asymmetry * 0.18) * 100
        )

        self.previous_blendshape_map = blendshapes

        return {
            "baselineNaturalness": float(baseline_naturalness),
            "baselineRigidity": float(baseline_rigidity),
            "expressionActivity": round3(expression_activity),
            "frameMotion": round3(frame_motion),
            "motionAverage": round3(motion_average),
            "asymmetry": round3(asymmetry),
            "mouthTension": round3(mouth_tension),
            "blink": round3(blink),
            "smile": round3(smile),
            "jawOpen": round3(jaw_open),
            "browActivity": round3(brow_activity),
        }

    def get_frame_motion(self, blendshapes: dict[str, float]) -> float:
        if self.previous_blendshape_map is None:
            return 0.0

        return average(
            [
                abs(score(blendshapes, key) - score(self.previous_blendshape_map, key))
                for key in TRACKED_MOTION_KEYS
            ]
        )


def ensure_model_file(output_dir: Path, model_path: Path | None) -> Path:
    if model_path:
        return model_path

    target = output_dir / "face_landmarker.task"
    if not target.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe FaceLandmarker model to {target}")
        urllib.request.urlretrieve(MODEL_URL, target)
    return target


def create_landmarker(model_path: Path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def sample_times(video_path: Path, interval_seconds: float) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    duration = frame_count / fps if fps > 0 else 5.0
    if duration <= 0 or not math.isfinite(duration):
        duration = 5.0

    count = max(1, int(math.ceil(duration / interval_seconds)))
    return [round(i * interval_seconds, 3) for i in range(count)]


def read_frame_at(cap: cv2.VideoCapture, seconds: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def detect_blendshapes(landmarker: Any, frame_bgr: np.ndarray) -> dict[str, float] | None:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(image)
    categories = result.face_blendshapes[0] if result.face_blendshapes else None
    if not categories:
        return None
    return {category.category_name: float(category.score) for category in categories}


def summarize_features(frame_features: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in FRAME_FEATURE_KEYS:
        values = np.array([row[key] for row in frame_features if key in row], dtype=float)
        if values.size == 0:
            summary[f"mean_{key}"] = np.nan
            summary[f"std_{key}"] = np.nan
            continue
        summary[f"mean_{key}"] = round(float(values.mean()), 6)
        summary[f"std_{key}"] = round(float(values.std(ddof=0)), 6)
    return summary


def extract_video_summary(
    video_path: Path,
    landmarker: Any,
    interval_seconds: float,
) -> dict[str, Any]:
    meta = parse_video_name(video_path)
    calculator = ExpressionFeatureCalculator()
    frame_features: list[dict[str, float]] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        for seconds in sample_times(video_path, interval_seconds):
            frame = read_frame_at(cap, seconds)
            if frame is None:
                continue
            blendshapes = detect_blendshapes(landmarker, frame)
            if blendshapes is None:
                continue
            frame_features.append(calculator.calculate(blendshapes))
    finally:
        cap.release()

    row: dict[str, Any] = {
        "userId": meta["userId"],
        "videoName": meta["videoName"],
        "taskType": meta["taskType"],
        "sampleCount": len(frame_features),
    }
    row.update(summarize_features(frame_features))
    row["_joinKey"] = meta["joinKey"]
    return row


def build_summary_dataset(data_dir: Path, output_dir: Path, model_path: Path, interval_seconds: float) -> pd.DataFrame:
    video_paths = sorted(data_dir.glob("*.mp4"), key=lambda path: join_key(path.name))
    if not video_paths:
        raise FileNotFoundError(f"No mp4 files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    with create_landmarker(model_path) as landmarker:
        for index, video_path in enumerate(video_paths, start=1):
            print(f"[{index}/{len(video_paths)}] Extracting {normalize_text(video_path.name)}")
            rows.append(extract_video_summary(video_path, landmarker, interval_seconds))

    summary = pd.DataFrame(rows)
    public_summary = summary.drop(columns=["_joinKey"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_summary.to_csv(output_dir / "faceease_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_labels(label_file: Path) -> pd.DataFrame:
    labels = pd.read_excel(label_file, sheet_name="Scaled_Label")
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"Scaled_Label is missing columns: {missing}")

    labels = labels[LABEL_COLUMNS].copy()
    labels["_joinKey"] = labels["videoName"].map(join_key)
    labels["userId"] = labels["userId"].map(normalize_text)
    labels["videoName"] = labels["videoName"].map(normalize_text)
    labels["taskType"] = labels["taskType"].map(normalize_text)
    return labels


def join_labels(summary: pd.DataFrame, labels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    feature_columns = [column for column in summary.columns if column.startswith(("mean_", "std_"))]
    summary_join = summary.drop(columns=["userId", "taskType"], errors="ignore")
    labeled = labels.merge(summary_join, on="_joinKey", how="inner", suffixes=("", "_fromVideo"))

    if labeled.empty:
        raise RuntimeError("No rows joined between summary dataset and Scaled_Label labels.")

    labeled = labeled[
        [
            "userId",
            "videoName",
            "taskType",
            "sampleCount",
            "taskPerformanceScore",
            "negativeExpressionScore",
            "finalNaturalnessScore",
            "rigidityScore",
            "reviewNeeded",
            *feature_columns,
        ]
    ].copy()
    labeled.to_csv(output_dir / "faceease_labeled_dataset.csv", index=False, encoding="utf-8-sig")

    unmatched_summary = sorted(set(summary["_joinKey"]) - set(labels["_joinKey"]))
    unmatched_labels = sorted(set(labels["_joinKey"]) - set(summary["_joinKey"]))
    diagnostics = {
        "summaryRows": int(len(summary)),
        "labelRows": int(len(labels)),
        "joinedRows": int(len(labeled)),
        "unmatchedSummaryVideoNames": unmatched_summary,
        "unmatchedLabelVideoNames": unmatched_labels,
    }
    (output_dir / "faceease_join_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return labeled


def filtered_training_data(labeled: pd.DataFrame, exclude_review_needed: bool) -> pd.DataFrame:
    data = labeled.copy()
    if exclude_review_needed:
        review = data["reviewNeeded"].astype(str).str.upper().str.strip()
        data = data[review.ne("YES")].copy()
    return data.dropna(subset=["taskPerformanceScore", "negativeExpressionScore"]).reset_index(drop=True)


def task_one_hot_kwargs() -> dict[str, Any]:
    try:
        OneHotEncoder(sparse_output=False)
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def build_model_candidates(feature_columns: list[str], include_task_type: bool) -> dict[str, Pipeline]:
    if include_task_type:
        preprocessor = ColumnTransformer(
            [
                ("taskType", OneHotEncoder(**task_one_hot_kwargs()), ["taskType"]),
                ("features", StandardScaler(), feature_columns),
            ]
        )
        input_columns = ["taskType", *feature_columns]
    else:
        preprocessor = ColumnTransformer([("features", StandardScaler(), feature_columns)])
        input_columns = feature_columns

    return {
        "Ridge": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "GradientBoostingRegressor": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }, input_columns


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_candidates(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    include_task_type: bool,
) -> tuple[str, Pipeline, dict[str, Any], list[str]]:
    candidates, input_columns = build_model_candidates(feature_columns, include_task_type)
    groups = data["userId"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise RuntimeError("GroupKFold requires at least two distinct userId groups.")

    n_splits = min(5, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    X = data[input_columns]
    y = data[target_column].astype(float)
    metrics: dict[str, Any] = {}

    for name, model in candidates.items():
        fold_metrics = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = np.clip(model.predict(X.iloc[test_idx]), 0, 100)
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "mae": float(mean_absolute_error(y.iloc[test_idx], predictions)),
                    "rmse": rmse(y.iloc[test_idx], predictions),
                    "r2": float(r2_score(y.iloc[test_idx], predictions)) if len(test_idx) > 1 else float("nan"),
                }
            )

        rmse_values = np.array([row["rmse"] for row in fold_metrics], dtype=float)
        mae_values = np.array([row["mae"] for row in fold_metrics], dtype=float)
        r2_values = np.array([row["r2"] for row in fold_metrics], dtype=float)
        metrics[name] = {
            "folds": fold_metrics,
            "mean_mae": float(np.nanmean(mae_values)),
            "mean_rmse": float(np.nanmean(rmse_values)),
            "std_rmse": float(np.nanstd(rmse_values)),
            "mean_r2": float(np.nanmean(r2_values)),
            "selection_score": float(np.nanmean(rmse_values) + np.nanstd(rmse_values)),
        }

    best_name = min(metrics, key=lambda name: metrics[name]["selection_score"])
    best_model = candidates[best_name]
    best_model.fit(X, y)
    return best_name, best_model, metrics, input_columns


def train_and_save_models(
    labeled: pd.DataFrame,
    output_dir: Path,
    exclude_review_needed: bool,
) -> dict[str, Any]:
    data = filtered_training_data(labeled, exclude_review_needed)
    feature_columns = [column for column in data.columns if column.startswith(("mean_", "std_"))]
    data = data.dropna(subset=feature_columns).reset_index(drop=True)
    if len(data) < 4:
        raise RuntimeError(f"Not enough training rows after filtering: {len(data)}")

    task_name, task_model, task_metrics, task_input_columns = evaluate_candidates(
        data=data,
        target_column="taskPerformanceScore",
        feature_columns=feature_columns,
        include_task_type=True,
    )
    negative_name, negative_model, negative_metrics, negative_input_columns = evaluate_candidates(
        data=data,
        target_column="negativeExpressionScore",
        feature_columns=feature_columns,
        include_task_type=False,
    )

    task_artifact = {
        "model": task_model,
        "modelName": task_name,
        "target": "taskPerformanceScore",
        "featureColumns": feature_columns,
        "inputColumns": task_input_columns,
        "includeTaskType": True,
        "excludeReviewNeeded": exclude_review_needed,
        "metrics": task_metrics,
        "positiveTasks": sorted(POSITIVE_TASKS),
        "negativeTasks": sorted(NEGATIVE_TASKS),
    }
    negative_artifact = {
        "model": negative_model,
        "modelName": negative_name,
        "target": "negativeExpressionScore",
        "featureColumns": feature_columns,
        "inputColumns": negative_input_columns,
        "includeTaskType": False,
        "excludeReviewNeeded": exclude_review_needed,
        "metrics": negative_metrics,
        "positiveTasks": sorted(POSITIVE_TASKS),
        "negativeTasks": sorted(NEGATIVE_TASKS),
    }

    joblib.dump(task_artifact, output_dir / "faceease_task_model.pkl")
    joblib.dump(negative_artifact, output_dir / "faceease_negative_model.pkl")

    report = {
        "trainingRows": int(len(data)),
        "excludedReviewNeeded": bool(exclude_review_needed),
        "taskPerformanceModel": {
            "selectedModel": task_name,
            "metrics": task_metrics,
        },
        "negativeExpressionModel": {
            "selectedModel": negative_name,
            "metrics": negative_metrics,
        },
    }
    (output_dir / "faceease_training_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and train the FaceEase video-level regression dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument(
        "--include-review-needed",
        action="store_true",
        help="Include rows where reviewNeeded is YES. By default they are excluded from training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model_file(args.output_dir, args.model_path)
    summary = build_summary_dataset(args.data_dir, args.output_dir, model_path, args.interval_seconds)
    labels = load_labels(args.label_file)
    labeled = join_labels(summary, labels, args.output_dir)
    report = train_and_save_models(
        labeled=labeled,
        output_dir=args.output_dir,
        exclude_review_needed=not args.include_review_needed,
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
