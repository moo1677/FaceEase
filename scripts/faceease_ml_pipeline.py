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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data_set"
DEFAULT_LABEL_FILE = ROOT / "Labeling.xlsx"
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
NATURALNESS_BINARY_LABELS = ["needs_improvement", "natural"]
NATURALNESS_LEVEL_LABELS = [
    "needs_improvement",
    "slightly_awkward",
    "mostly_natural",
    "very_natural",
]
NATURALNESS_LEVEL_DISPLAY_TEXT = {
    "needs_improvement": "개선 필요",
    "slightly_awkward": "다소 어색함",
    "mostly_natural": "대체로 자연스러움",
    "very_natural": "매우 자연스러움",
}
NEGATIVE_PATTERN_LABELS = [0, 1]
BOUNDARY_AMBIGUOUS_RANGES = ((38.0, 42.0), (58.0, 62.0), (78.0, 82.0))
CALM_BASELINE_TASK_TYPE = "calm"
CALM_BASELINE_SCALE = 0.8
CALM_BASELINE_FEATURE_KEYS = (
    "smile",
    "jawOpen",
    "browActivity",
    "mouthTension",
    "blink",
    "asymmetry",
    "browDown",
)

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
    "score",
]
RAW_LABEL_COLUMNS = ["영상번호", "표정", "avg"]
TARGET_COLUMN = "score"
SCORE_DEFINITION = "score = task performance score - negative pattern penalty"
NEGATIVE_PATTERN_DEFINITION = {
    "negativePatterns": ["awkward", "over_smile"],
    "description": "어색한표정과 과하게웃는표정을 부정점수 보정 기준으로 사용",
}
PRESENTATION_SUMMARY = [
    "Labeling.xlsx의 score 컬럼을 최종 자연스러움 점수로 사용하였다.",
    "score는 과제수행 정도에서 어색한표정과 과하게웃는표정에 따른 부정점수를 보정한 값으로 해석하였다.",
    "본 프로젝트는 복잡한 다중 분류 구조 대신 score 예측 회귀 모델을 중심으로 단순화하였다.",
    "모델 성능은 MAE, RMSE, R²를 통해 평가하였다.",
]
NON_FEATURE_COLUMNS = {
    "userId",
    "videoName",
    "_joinKey",
    "sampleId",
    "timestampIso",
    "reviewNeeded",
    "taskPerformanceScore",
    "negativeExpressionScore",
    "finalNaturalnessScore",
    "rigidityScore",
    "naturalnessBinaryLabel",
    "naturalnessLevel",
    "boundaryAmbiguous",
    "negative_pattern",
    TARGET_COLUMN,
}


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
    calm_baseline: dict[str, float] | None = None
    baseline_scale: float = CALM_BASELINE_SCALE
    previous_blendshape_map: dict[str, float] | None = None
    motion_window: list[float] = field(default_factory=list)

    def calculate(self, blendshapes: dict[str, float]) -> dict[str, float]:
        smile_left = score(blendshapes, "mouthSmileLeft")
        smile_right = score(blendshapes, "mouthSmileRight")
        raw_jaw_open = score(blendshapes, "jawOpen")
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

        raw_smile = average([smile_left, smile_right])
        raw_blink = average([eye_blink_left, eye_blink_right])
        raw_mouth_tension = clamp01(mouth_press * 0.75 + mouth_pucker * 0.15 + mouth_funnel * 0.1)
        raw_asymmetry = clamp01(
            abs(smile_left - smile_right) * 0.55
            + abs(eye_blink_left - eye_blink_right) * 0.25
            + abs(score(blendshapes, "browOuterUpLeft") - score(blendshapes, "browOuterUpRight")) * 0.2
        )
        raw_brow_activity = clamp01(brow_up * 0.55 + brow_down * 0.45)
        smile = self.adjust_for_calm_baseline("smile", raw_smile)
        jaw_open = self.adjust_for_calm_baseline("jawOpen", raw_jaw_open)
        brow_activity = self.adjust_for_calm_baseline("browActivity", raw_brow_activity)
        mouth_tension = self.adjust_for_calm_baseline("mouthTension", raw_mouth_tension)
        blink = self.adjust_for_calm_baseline("blink", raw_blink)
        asymmetry = self.adjust_for_calm_baseline("asymmetry", raw_asymmetry)
        adjusted_brow_down = self.adjust_for_calm_baseline("browDown", brow_down)
        expression_activity = clamp01(
            smile * 0.3 + jaw_open * 0.2 + brow_activity * 0.2 + blink * 0.12 + mouth_tension * 0.18
        )
        motion_average = average(self.motion_window)

        stillness = 1 - clamp01(motion_average / 0.035)
        tension_signal = clamp01(mouth_tension * 0.65 + adjusted_brow_down * 0.2 + asymmetry * 0.15)
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
            "browDown": round3(adjusted_brow_down),
        }

    def adjust_for_calm_baseline(self, key: str, value: float) -> float:
        if not self.calm_baseline:
            return clamp01(value)
        return clamp01(value - self.calm_baseline.get(key, 0.0) * self.baseline_scale)

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


def extract_video_frame_features(
    video_path: Path,
    landmarker: Any,
    interval_seconds: float,
    calm_baseline: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    calculator = ExpressionFeatureCalculator(calm_baseline=calm_baseline)
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

    return frame_features


def summarize_calm_baseline(frame_features: list[dict[str, float]]) -> dict[str, float]:
    baseline: dict[str, float] = {}
    for key in CALM_BASELINE_FEATURE_KEYS:
        values = np.array([row[key] for row in frame_features if key in row], dtype=float)
        baseline[key] = round(float(values.mean()), 6) if values.size else 0.0
    return baseline


def build_calm_baselines(
    video_paths: list[Path],
    landmarker: Any,
    interval_seconds: float,
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    calm_paths = []
    for video_path in video_paths:
        meta = parse_video_name(video_path)
        if meta["taskType"] == CALM_BASELINE_TASK_TYPE:
            calm_paths.append((meta, video_path))

    for index, (meta, video_path) in enumerate(calm_paths, start=1):
        print(f"[baseline {index}/{len(calm_paths)}] Extracting {normalize_text(video_path.name)}")
        raw_features = extract_video_frame_features(video_path, landmarker, interval_seconds)
        if raw_features:
            baselines[meta["userId"]] = summarize_calm_baseline(raw_features)

    return baselines


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
    calm_baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    meta = parse_video_name(video_path)
    frame_features = extract_video_frame_features(video_path, landmarker, interval_seconds, calm_baseline)

    row: dict[str, Any] = {
        "userId": meta["userId"],
        "videoName": meta["videoName"],
        "taskType": meta["taskType"],
        "sampleCount": len(frame_features),
        "calmBaselineScale": CALM_BASELINE_SCALE if calm_baseline else 0.0,
    }
    row.update(summarize_features(frame_features))
    row["_joinKey"] = meta["joinKey"]
    return row


def build_summary_dataset(data_dir: Path, output_dir: Path, model_path: Path, interval_seconds: float) -> pd.DataFrame:
    video_paths = sorted(data_dir.glob("*.mp4"), key=lambda path: join_key(path.name))
    if not video_paths:
        raise FileNotFoundError(f"No mp4 files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with create_landmarker(model_path) as landmarker:
        calm_baselines = build_calm_baselines(video_paths, landmarker, interval_seconds)
        (output_dir / "faceease_calm_baselines.json").write_text(
            json.dumps(json_safe(calm_baselines), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for index, video_path in enumerate(video_paths, start=1):
            meta = parse_video_name(video_path)
            calm_baseline = calm_baselines.get(meta["userId"])
            print(f"[{index}/{len(video_paths)}] Extracting {normalize_text(video_path.name)}")
            rows.append(extract_video_summary(video_path, landmarker, interval_seconds, calm_baseline))

    summary = pd.DataFrame(rows)
    public_summary = summary.drop(columns=["_joinKey"])
    public_summary.to_csv(output_dir / "faceease_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_labels(label_file: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(label_file)
    score_sheet = next(
        (
            sheet_name
            for sheet_name in excel.sheet_names
            if TARGET_COLUMN in pd.read_excel(label_file, sheet_name=sheet_name, nrows=0).columns
        ),
        None,
    )

    if score_sheet:
        labels = pd.read_excel(label_file, sheet_name=score_sheet)
    elif "Raw_Clean" in excel.sheet_names:
        raw = pd.read_excel(label_file, sheet_name="Raw_Clean")
        missing_raw = [column for column in RAW_LABEL_COLUMNS if column not in raw.columns]
        if missing_raw:
            raise ValueError(f"Raw_Clean is missing columns: {missing_raw}")
        labels = convert_raw_clean_labels(raw)
    else:
        raise ValueError(
            f"{label_file} must contain a score column, or a Raw_Clean sheet with {RAW_LABEL_COLUMNS}."
        )

    if TARGET_COLUMN not in labels.columns and "avg" in labels.columns:
        labels = labels.rename(columns={"avg": TARGET_COLUMN})

    missing = [column for column in [TARGET_COLUMN] if column not in labels.columns]
    if missing:
        raise ValueError(f"Labeling.xlsx is missing columns: {missing}")

    labels = labels.copy()
    if "videoName" not in labels.columns and {"영상번호", "표정"}.issubset(labels.columns):
        labels = convert_raw_clean_labels(labels)
    if "userId" in labels.columns:
        labels["userId"] = labels["userId"].map(normalize_text)
    if "videoName" in labels.columns:
        labels["videoName"] = labels["videoName"].map(normalize_text)
        labels["_joinKey"] = labels["videoName"].map(join_key)
    if "taskType" in labels.columns:
        labels["taskType"] = labels["taskType"].map(normalize_text)
    labels[TARGET_COLUMN] = pd.to_numeric(labels[TARGET_COLUMN], errors="coerce")
    return labels


def convert_raw_clean_labels(raw: pd.DataFrame) -> pd.DataFrame:
    score_column = TARGET_COLUMN if TARGET_COLUMN in raw.columns else "avg"
    if score_column not in raw.columns:
        raise ValueError(f"Raw labels must include either '{TARGET_COLUMN}' or 'avg'.")

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        user_id = normalize_text(row["영상번호"])
        expression_name = normalize_text(row["표정"]).replace(" ", "")
        task_type = TASK_SUFFIXES.get(expression_name)
        if not task_type:
            raise ValueError(f"Unknown expression label in Raw_Clean: {expression_name}")

        score_value = float(row[score_column])
        rows.append(
            {
                "userId": user_id,
                "videoName": f"{user_id}_{expression_name}.mp4",
                "taskType": task_type,
                TARGET_COLUMN: score_value,
            }
        )

    return pd.DataFrame(rows)


def join_labels(summary: pd.DataFrame, labels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    feature_columns = [column for column in summary.columns if column.startswith(("mean_", "std_"))]
    summary_join = summary.drop(columns=["userId", "taskType"], errors="ignore")
    labeled = labels.merge(summary_join, on="_joinKey", how="inner", suffixes=("", "_fromVideo"))

    if labeled.empty:
        raise RuntimeError("No rows joined between summary dataset and Labeling.xlsx labels.")

    labeled = labeled[
        [
            "userId",
            "videoName",
            "taskType",
            TARGET_COLUMN,
            "sampleCount",
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
    if exclude_review_needed and "reviewNeeded" in data.columns:
        review = data["reviewNeeded"].astype(str).str.upper().str.strip()
        data = data[review.ne("YES")].copy()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return data


def naturalness_binary_label(final_naturalness_score: float) -> str:
    return "natural" if float(final_naturalness_score) >= 60 else "needs_improvement"


def naturalness_level(final_naturalness_score: float) -> str:
    score_value = float(final_naturalness_score)
    if score_value < 40:
        return "needs_improvement"
    if score_value < 60:
        return "slightly_awkward"
    if score_value < 80:
        return "mostly_natural"
    return "very_natural"


def is_boundary_ambiguous(final_naturalness_score: float) -> bool:
    score_value = float(final_naturalness_score)
    return any(lower <= score_value <= upper for lower, upper in BOUNDARY_AMBIGUOUS_RANGES)


def task_one_hot_kwargs() -> dict[str, Any]:
    try:
        OneHotEncoder(sparse_output=False)
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def build_model_candidates(feature_columns: list[str], include_task_type: bool) -> dict[str, Pipeline]:
    preprocessor, input_columns = build_preprocessor(feature_columns, include_task_type)

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


def build_classifier_candidates(feature_columns: list[str], include_task_type: bool) -> dict[str, Pipeline]:
    preprocessor, input_columns = build_preprocessor(feature_columns, include_task_type)

    return {
        "LogisticRegression": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "GradientBoostingClassifier": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=180,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }, input_columns


def build_preprocessor(
    feature_columns: list[str], include_task_type: bool
) -> tuple[ColumnTransformer, list[str]]:
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

    return preprocessor, input_columns


def infer_feature_columns(data: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS and not column.startswith("_")
    ]
    if not feature_columns:
        raise RuntimeError("No feature columns found for score regression.")
    return feature_columns


def split_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def build_score_preprocessor(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns, categorical_columns = split_feature_columns(data, feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(**task_one_hot_kwargs())),
                    ]
                ),
                categorical_columns,
            )
        )

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if not transformers:
        raise RuntimeError("No usable numeric or categorical feature columns found.")

    return ColumnTransformer(transformers), numeric_columns, categorical_columns


def build_score_model_candidates(data: pd.DataFrame, feature_columns: list[str]) -> dict[str, Pipeline]:
    def preprocessor() -> ColumnTransformer:
        return build_score_preprocessor(data, feature_columns)[0]

    return {
        "Ridge": Pipeline(
            [
                ("preprocess", preprocessor()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocess", preprocessor()),
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
                ("preprocess", preprocessor()),
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
    }


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
) -> tuple[str, Pipeline, dict[str, Any], list[str], list[str], list[str]]:
    del include_task_type
    candidates = build_score_model_candidates(data, feature_columns)
    n_splits = min(5, len(data))
    if n_splits < 2:
        raise RuntimeError(f"At least two training rows are required for cross validation: {len(data)}")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = data[feature_columns]
    y = data[target_column].astype(float)
    metrics: dict[str, Any] = {}

    for name, model in candidates.items():
        fold_metrics = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
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
            "mean_r2": float(np.nanmean(r2_values)),
        }

    best_name = max(metrics, key=lambda name: metrics[name]["mean_r2"])
    best_model = candidates[best_name]
    best_model.fit(X, y)
    _, numeric_columns, categorical_columns = build_score_preprocessor(data, feature_columns)
    return best_name, best_model, metrics, feature_columns, numeric_columns, categorical_columns


def evaluate_classifier_candidates(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    include_task_type: bool,
    labels: list[Any],
    average_mode: str,
    positive_label: Any | None = None,
) -> tuple[str, Pipeline, dict[str, Any], list[str]]:
    if len(data) < 4:
        raise RuntimeError(f"Not enough classifier rows for {target_column}: {len(data)}")

    candidates, input_columns = build_classifier_candidates(feature_columns, include_task_type)
    groups = data["userId"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise RuntimeError("GroupKFold requires at least two distinct userId groups.")

    if data[target_column].nunique() < 2:
        raise RuntimeError(f"{target_column} requires at least two classes.")

    n_splits = min(5, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    X = data[input_columns]
    y = data[target_column]
    metrics: dict[str, Any] = {}

    for name, model in candidates.items():
        fold_metrics = []
        all_true = []
        all_pred = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = model.predict(X.iloc[test_idx])
            truth = y.iloc[test_idx]
            all_true.extend(truth.tolist())
            all_pred.extend(predictions.tolist())
            fold_metrics.append(
                classifier_metric_row(
                    fold_index=fold_index,
                    y_true=truth,
                    y_pred=predictions,
                    labels=labels,
                    average_mode=average_mode,
                    positive_label=positive_label,
                )
            )

        aggregate = classifier_metric_row(
            fold_index=None,
            y_true=pd.Series(all_true),
            y_pred=np.array(all_pred),
            labels=labels,
            average_mode=average_mode,
            positive_label=positive_label,
        )
        selection_metric = aggregate["macro_f1"] if average_mode == "macro" else aggregate["f1"]
        metrics[name] = {
            "folds": fold_metrics,
            **{key: value for key, value in aggregate.items() if key != "fold"},
            "selectionMetric": "macro_f1" if average_mode == "macro" else "f1",
            "selection_score": float(1.0 - selection_metric),
        }

    best_name = select_best_classifier(metrics, "macro_f1" if average_mode == "macro" else "f1")
    best_model = candidates[best_name]
    best_model.fit(X, y)
    return best_name, best_model, metrics, input_columns


def select_best_classifier(metrics: dict[str, Any], metric_key: str, tolerance: float = 0.01) -> str:
    best_metric = max(float(row[metric_key]) for row in metrics.values())
    candidates = [
        name
        for name, row in metrics.items()
        if float(row[metric_key]) >= best_metric - tolerance
    ]
    return max(candidates, key=lambda name: (float(metrics[name]["accuracy"]), float(metrics[name][metric_key])))


def classifier_metric_row(
    fold_index: int | None,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[Any],
    average_mode: str,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    if fold_index is not None:
        row["fold"] = fold_index

    if average_mode == "macro":
        row["macro_precision"] = float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )
        row["macro_recall"] = float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )
        row["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    else:
        kwargs = {"zero_division": 0}
        if positive_label is not None:
            kwargs["pos_label"] = positive_label
            row["positiveLabel"] = positive_label
        row["precision"] = float(precision_score(y_true, y_pred, **kwargs))
        row["recall"] = float(recall_score(y_true, y_pred, **kwargs))
        row["f1"] = float(f1_score(y_true, y_pred, **kwargs))

    return row


def train_naturalness_level_experiment(
    data: pd.DataFrame,
    feature_columns: list[str],
    experiment_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "experimentName": experiment_name,
        "trained": False,
        "trainingRows": int(len(data)),
        "classDistribution": class_distribution(data, "naturalnessLevel", NATURALNESS_LEVEL_LABELS),
    }
    if len(data) < 4:
        result["warning"] = f"Skipped: not enough rows after filtering ({len(data)})."
        return result
    if data["userId"].astype(str).nunique() < 2:
        result["warning"] = "Skipped: fewer than two userId groups after filtering."
        return result
    if data["naturalnessLevel"].nunique() < 2:
        result["warning"] = "Skipped: fewer than two naturalnessLevel classes after filtering."
        return result

    name, model, metrics, input_columns = evaluate_classifier_candidates(
        data=data,
        target_column="naturalnessLevel",
        feature_columns=feature_columns,
        include_task_type=True,
        labels=NATURALNESS_LEVEL_LABELS,
        average_mode="macro",
    )
    result.update(
        {
            "trained": True,
            "selectedModel": name,
            "model": model,
            "metrics": metrics,
            "inputColumns": input_columns,
            "selectedMetrics": metrics[name],
        }
    )
    return result


def experiment_report(experiment: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "experimentName",
        "trained",
        "trainingRows",
        "classDistribution",
        "selectedModel",
        "metrics",
        "selectedMetrics",
        "warning",
    ]
    return {key: experiment[key] for key in keys if key in experiment}


def class_distribution(data: pd.DataFrame, column: str, labels: list[Any]) -> dict[str, int]:
    values = data[column].value_counts().reindex(labels, fill_value=0)
    return {str(key): int(value) for key, value in values.to_dict().items()}


def naturalness_level_rules() -> dict[str, str]:
    return {
        "needs_improvement": "0 <= finalNaturalnessScore < 40",
        "slightly_awkward": "40 <= finalNaturalnessScore < 60",
        "mostly_natural": "60 <= finalNaturalnessScore < 80",
        "very_natural": "80 <= finalNaturalnessScore <= 100",
    }


def boundary_ambiguous_rules() -> list[str]:
    return [
        "38 <= finalNaturalnessScore <= 42",
        "58 <= finalNaturalnessScore <= 62",
        "78 <= finalNaturalnessScore <= 82",
    ]


def train_negative_pattern_classifier(
    data: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    numeric_feature_columns = [
        col for col in feature_columns
        if col != "taskType" and pd.api.types.is_numeric_dtype(data[col])
    ]

    work = data.copy()
    work["negative_pattern"] = work["taskType"].apply(
        lambda t: 1 if t in NEGATIVE_TASKS else 0
    )

    result: dict[str, Any] = {
        "trained": False,
        "trainingRows": int(len(work)),
        "classDistribution": class_distribution(work, "negative_pattern", NEGATIVE_PATTERN_LABELS),
    }

    if work["negative_pattern"].nunique() < 2:
        result["warning"] = "Skipped: no negative task examples in dataset."
        return result
    if work["userId"].astype(str).nunique() < 2:
        result["warning"] = "Skipped: fewer than two userId groups."
        return result

    name, model, metrics, input_columns = evaluate_classifier_candidates(
        data=work,
        target_column="negative_pattern",
        feature_columns=numeric_feature_columns,
        include_task_type=False,
        labels=NEGATIVE_PATTERN_LABELS,
        average_mode="binary",
        positive_label=1,
    )

    artifact = {
        "model": model,
        "modelName": name,
        "target": "negative_pattern",
        "classes": NEGATIVE_PATTERN_LABELS,
        "featureColumns": numeric_feature_columns,
        "inputColumns": input_columns,
        "metrics": metrics,
        "selectedMetrics": metrics[name],
        "negativeTasks": sorted(NEGATIVE_TASKS),
    }
    joblib.dump(artifact, output_dir / "faceease_negative_pattern_model.pkl")

    result.update({
        "trained": True,
        "selectedModel": name,
        "selectedMetrics": metrics[name],
    })
    return result


def train_and_save_models(
    labeled: pd.DataFrame,
    output_dir: Path,
    exclude_review_needed: bool,
    exclude_boundary_ambiguous: bool = False,
) -> dict[str, Any]:
    del exclude_boundary_ambiguous

    all_data = filtered_training_data(labeled, exclude_review_needed)
    feature_columns = infer_feature_columns(all_data)
    if len(all_data) < 4:
        raise RuntimeError(f"Not enough training rows after filtering: {len(all_data)}")

    # score model: 긍정 과제(자연미소, 경청, 차분)만으로 학습
    positive_data = all_data[all_data["taskType"].isin(POSITIVE_TASKS)].copy()
    if len(positive_data) < 4:
        raise RuntimeError(f"Not enough positive task rows for score model: {len(positive_data)}")
    positive_data.to_csv(output_dir / "faceease_labeled_dataset.csv", index=False, encoding="utf-8-sig")

    model_name, model, metrics, input_columns, numeric_columns, categorical_columns = evaluate_candidates(
        data=positive_data,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        include_task_type=False,
    )

    artifact = {
        "model": model,
        "modelName": model_name,
        "target": TARGET_COLUMN,
        "scoreDefinition": SCORE_DEFINITION,
        "negativePatternDefinition": NEGATIVE_PATTERN_DEFINITION,
        "featureColumns": feature_columns,
        "inputColumns": input_columns,
        "numericFeatureColumns": numeric_columns,
        "categoricalFeatureColumns": categorical_columns,
        "includeTaskType": "taskType" in categorical_columns,
        "excludeReviewNeeded": exclude_review_needed,
        "featureAdjustment": {
            "calmBaselineTaskType": CALM_BASELINE_TASK_TYPE,
            "calmBaselineScale": CALM_BASELINE_SCALE,
            "calmBaselineFeatureKeys": list(CALM_BASELINE_FEATURE_KEYS),
        },
        "metrics": metrics,
        "selectedMetrics": metrics[model_name],
        "positiveTasks": sorted(POSITIVE_TASKS),
        "negativeTasks": sorted(NEGATIVE_TASKS),
    }
    joblib.dump(artifact, output_dir / "faceease_score_model.pkl")

    # negative pattern model: 전체 데이터로 이진 분류 학습
    negative_report = train_negative_pattern_classifier(all_data, feature_columns, output_dir)

    report = {
        "scoreModelTrainingRows": int(len(positive_data)),
        "allTrainingRows": int(len(all_data)),
        "target": TARGET_COLUMN,
        "scoreDefinition": SCORE_DEFINITION,
        "negativePatternDefinition": NEGATIVE_PATTERN_DEFINITION,
        "selectedModel": model_name,
        "selectedMetrics": metrics[model_name],
        "metrics": metrics,
        "negativePatternModel": negative_report,
        "presentationSummary": PRESENTATION_SUMMARY,
    }
    (output_dir / "faceease_training_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FaceEase score regression model.")
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
    parser.add_argument(
        "--exclude-boundary-ambiguous",
        action="store_true",
        help="Deprecated. Kept for CLI compatibility; score regression does not use boundary relabeling.",
    )
    return parser.parse_args()


def has_feature_columns(labels: pd.DataFrame) -> bool:
    candidates = [column for column in labels.columns if column not in NON_FEATURE_COLUMNS]
    if any(column != "taskType" for column in candidates):
        return True
    return bool(candidates and "videoName" not in labels.columns)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.label_file)
    if has_feature_columns(labels):
        labeled = labels.copy()
        labeled.to_csv(args.output_dir / "faceease_labeled_dataset.csv", index=False, encoding="utf-8-sig")
    else:
        if "_joinKey" not in labels.columns:
            raise RuntimeError("Labeling.xlsx has no feature columns and no videoName/join key for video feature extraction.")
        model_path = ensure_model_file(args.output_dir, args.model_path)
        summary = build_summary_dataset(args.data_dir, args.output_dir, model_path, args.interval_seconds)
        labeled = join_labels(summary, labels, args.output_dir)
    report = train_and_save_models(
        labeled=labeled,
        output_dir=args.output_dir,
        exclude_review_needed=not args.include_review_needed,
        exclude_boundary_ambiguous=args.exclude_boundary_ambiguous,
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
