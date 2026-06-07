import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";
import "./styles.css";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const BROWSER_MODEL_PATH = "/models/faceease_models.json";
const ANALYSIS_DURATION_MS = 5000;

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const analyzeButton = document.querySelector("#analyzeButton");
const taskButtons = document.querySelectorAll("[data-task-type]");
const taskPromptElement = document.querySelector("#taskPrompt");
const snapshotButton = document.querySelector("#snapshotButton");
const recordButton = document.querySelector("#recordButton");
const downloadButton = document.querySelector("#downloadButton");
const clearButton = document.querySelector("#clearButton");
const statusElement = document.querySelector("#status");
const progressElement = document.querySelector("#analysisProgress");
const naturalnessTargetInput = document.querySelector("#naturalnessTarget");
const rigidityTargetInput = document.querySelector("#rigidityTarget");
const feedbackList = document.querySelector("#feedbackList");
const scoreElements = {
  naturalness: document.querySelector("#naturalnessScore"),
  task: document.querySelector("#taskScore"),
  negative: document.querySelector("#negativeScore"),
  serviceSampleCount: document.querySelector("#serviceSampleCount")
};

export const FEATURE_KEYS = [
  "expressionActivity",
  "frameMotion",
  "motionAverage",
  "asymmetry",
  "mouthTension",
  "blink",
  "smile",
  "jawOpen",
  "browActivity"
];
export const SUMMARY_FEATURE_KEYS = [
  ...FEATURE_KEYS,
  "baselineNaturalness",
  "baselineRigidity"
];
export const POSITIVE_TASK_TYPES = ["natural_smile", "listening", "calm"];
export const FEEDBACK_THRESHOLDS = {
  natural_smile: {
    weakSmileMax: 0.04,
    lowEyeBlinkMax: 0.08,
    lowEyeBrowMax: 0.045,
    asymmetryHigh: 0.07,
    mouthTensionHigh: 0.25
  },
  listening: {
    overSmileHigh: 0.12,
    mouthTensionHigh: 0.11,
    lowBlinkMax: 0.06,
    restlessExpressionStdHigh: 0.04,
    jawOpenHigh: 0.07
  },
  calm: {
    mouthTensionHigh: 0.13,
    browActivityHigh: 0.22,
    asymmetryHigh: 0.05,
    expressionActivityHigh: 0.13,
    frameMotionStdHigh: 0.03
  }
};
const SAMPLE_INTERVAL_MS = 500;
const TASK_PROMPTS = {
  natural_smile: "자연스러운 미소를 지어보세요",
  listening: "경청하는 표정을 지어보세요",
  calm: "차분한 표정을 지어보세요"
};

const ui = {
  sampleCount: document.querySelector("#sampleCount"),
  recordState: document.querySelector("#recordState"),
  activity: document.querySelector("#activityValue"),
  motion: document.querySelector("#motionValue"),
  asymmetry: document.querySelector("#asymmetryValue"),
  mouthTension: document.querySelector("#mouthTensionValue"),
  blink: document.querySelector("#blinkValue")
};

let faceLandmarker;
let drawingUtils;
let isRunning = false;
let lastVideoTime = -1;
let lastConsoleTime = 0;
let lastSampleTime = 0;
let latestVariables = null;
let isRecording = false;
let previousBlendshapeMap = null;
let browserModels = null;
let selectedTaskType = "natural_smile";
let isAnalyzing = false;
let analysisStartTime = 0;
let lastAnalysisSampleTime = 0;
const motionWindow = [];
const datasetRows = [];
const analysisRows = [];

startButton.disabled = true;
startButton.addEventListener("click", startCamera);
analyzeButton.addEventListener("click", startServiceAnalysis);
taskButtons.forEach((button) => {
  button.addEventListener("click", () => selectTaskType(button.dataset.taskType));
});
snapshotButton.addEventListener("click", collectSample);
recordButton.addEventListener("click", toggleRecording);
downloadButton.addEventListener("click", downloadCsv);
clearButton.addEventListener("click", clearDataset);

initFaceLandmarker();
loadBrowserModels();

async function initFaceLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_PATH,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    drawingUtils = new DrawingUtils(context);
    startButton.disabled = false;
    statusElement.textContent = browserModels
      ? "카메라를 시작하면 분석할 수 있습니다."
      : "회귀 모델을 불러오는 중입니다.";
    updateServiceControls();
  } catch (error) {
    console.error("FaceLandmarker 초기화 실패:", error);
    statusElement.textContent = "모델 로딩 실패. 콘솔을 확인하세요.";
  }
}

async function loadBrowserModels() {
  try {
    const response = await fetch(BROWSER_MODEL_PATH);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    browserModels = await response.json();
    statusElement.textContent = faceLandmarker
      ? "카메라를 시작하면 분석할 수 있습니다."
      : "얼굴 인식 모델을 불러오는 중입니다.";
    updateServiceControls();
  } catch (error) {
    console.error("브라우저 회귀 모델 로딩 실패:", error);
    statusElement.textContent = "회귀 모델 로딩 실패. npm run ml:export를 실행했는지 확인하세요.";
  }
}

async function startCamera() {
  if (!faceLandmarker || isRunning) {
    return isRunning;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    resizeCanvasToVideo();

    isRunning = true;
    startButton.disabled = true;
    statusElement.textContent = "얼굴을 인식하면 5초 분석을 시작할 수 있습니다.";
    updateServiceControls();
    requestAnimationFrame(renderLoop);
    return true;
  } catch (error) {
    console.error("카메라 시작 실패:", error);
    statusElement.textContent = "카메라 권한을 허용해야 분석할 수 있습니다.";
    return false;
  }
}

function renderLoop() {
  if (!isRunning) {
    return;
  }

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    resizeCanvasToVideo();

    const result = faceLandmarker.detectForVideo(video, performance.now());
    drawResult(result);
    processResult(result);
  }

  requestAnimationFrame(renderLoop);
}

function resizeCanvasToVideo() {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height || (canvas.width === width && canvas.height === height)) {
    return;
  }

  canvas.width = width;
  canvas.height = height;
}

function drawResult(result) {
  context.clearRect(0, 0, canvas.width, canvas.height);

  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) {
    if (!isAnalyzing) {
      statusElement.textContent = "얼굴을 찾는 중입니다.";
    }
    return;
  }

  if (!isAnalyzing) {
    statusElement.textContent = "얼굴 인식 중입니다. 5초 분석을 시작할 수 있습니다.";
  }
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
    color: "#24d18b",
    lineWidth: 2
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
    color: "#ff7a59",
    lineWidth: 2
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
    color: "#5aa9ff",
    lineWidth: 1.5
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
    color: "#5aa9ff",
    lineWidth: 1.5
  });
}

function processResult(result) {
  const categories = result.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) {
    resetMetrics();
    return;
  }

  const blendshapes = Object.fromEntries(
    categories.map((category) => [category.categoryName, category.score])
  );
  const variables = calculateExpressionVariables(blendshapes);
  latestVariables = variables;

  updateMetrics(variables);

  if (performance.now() - lastConsoleTime > 500) {
    lastConsoleTime = performance.now();
    console.table({
      baselineNaturalness: variables.baselineNaturalness,
      baselineRigidity: variables.baselineRigidity,
      targetNaturalness: getTargets()?.targetNaturalness ?? null,
      targetRigidity: getTargets()?.targetRigidity ?? null,
      expressionActivity: variables.expressionActivity,
      frameMotion: variables.frameMotion,
      motionAverage: variables.motionAverage,
      asymmetry: variables.asymmetry,
      mouthTension: variables.mouthTension,
      blink: variables.blink,
      smile: variables.smile,
      jawOpen: variables.jawOpen,
      browActivity: variables.browActivity
    });
  }

  if (isRecording && performance.now() - lastSampleTime > SAMPLE_INTERVAL_MS) {
    collectSample();
  }

  if (isAnalyzing) {
    collectAnalysisFrame(variables);
  }
}

function selectTaskType(taskType) {
  selectedTaskType = taskType;
  taskPromptElement.textContent = TASK_PROMPTS[taskType];
  taskButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.taskType === taskType);
  });
  resetServiceResult();
}

async function startServiceAnalysis() {
  if (isAnalyzing || !faceLandmarker || !browserModels) {
    return;
  }

  if (!isRunning) {
    const started = await startCamera();
    if (!started) {
      return;
    }
  }

  resetExpressionState();
  analysisRows.length = 0;
  isAnalyzing = true;
  analysisStartTime = performance.now();
  lastAnalysisSampleTime = 0;
  progressElement.style.width = "0%";
  scoreElements.serviceSampleCount.textContent = "0";
  feedbackList.innerHTML = "<li>분석 중입니다.</li>";
  setScoreText("--", "--", "--");
  updateServiceControls();
}

function collectAnalysisFrame(variables) {
  const now = performance.now();
  const elapsed = now - analysisStartTime;
  const progress = Math.min(100, (elapsed / ANALYSIS_DURATION_MS) * 100);
  progressElement.style.width = `${progress}%`;
  statusElement.textContent = `분석 중입니다. ${Math.max(0, Math.ceil((ANALYSIS_DURATION_MS - elapsed) / 1000))}초`;

  if (now - lastAnalysisSampleTime >= SAMPLE_INTERVAL_MS) {
    analysisRows.push({ ...variables });
    lastAnalysisSampleTime = now;
    scoreElements.serviceSampleCount.textContent = String(analysisRows.length);
  }

  if (elapsed >= ANALYSIS_DURATION_MS) {
    finishServiceAnalysis();
  }
}

function finishServiceAnalysis() {
  isAnalyzing = false;
  updateServiceControls();

  if (!analysisRows.length) {
    statusElement.textContent = "분석할 수 있는 얼굴 feature가 충분하지 않습니다.";
    feedbackList.innerHTML = "<li>얼굴이 화면 중앙에 보이도록 맞춘 뒤 다시 시도하세요.</li>";
    progressElement.style.width = "0%";
    return;
  }

  const summaryFeatures = summarizeFrameVariables(analysisRows);
  const predictedTaskPerformanceScore = predictModel(browserModels.taskModel, {
    taskType: selectedTaskType,
    summaryFeatures
  });
  const predictedNegativeExpressionScore = predictModel(browserModels.negativeModel, {
    taskType: selectedTaskType,
    summaryFeatures
  });
  const finalNaturalnessScore = calculateFinalNaturalnessScore(
    selectedTaskType,
    predictedTaskPerformanceScore,
    predictedNegativeExpressionScore
  );
  const feedback = generateTaskFeedback({
    taskType: selectedTaskType,
    predictedTaskPerformanceScore,
    predictedNegativeExpressionScore,
    finalNaturalnessScore,
    summaryFeatures
  });

  setScoreText(
    formatScore(finalNaturalnessScore),
    formatScore(predictedTaskPerformanceScore),
    formatScore(predictedNegativeExpressionScore)
  );
  feedbackList.innerHTML = feedback.map((message) => `<li>${escapeHtml(message)}</li>`).join("");
  statusElement.textContent = "분석 완료";
  progressElement.style.width = "100%";
  console.log("faceease_service_result", {
    taskType: selectedTaskType,
    predictedTaskPerformanceScore,
    predictedNegativeExpressionScore,
    finalNaturalnessScore,
    summaryFeatures,
    feedback
  });
}

function updateServiceControls() {
  analyzeButton.disabled = !faceLandmarker || !browserModels || isAnalyzing;
  taskButtons.forEach((button) => {
    button.disabled = isAnalyzing;
  });
}

function resetServiceResult() {
  setScoreText("--", "--", "--");
  scoreElements.serviceSampleCount.textContent = "0";
  feedbackList.innerHTML = "<li>분석 결과가 여기에 표시됩니다.</li>";
  progressElement.style.width = "0%";
}

function setScoreText(naturalness, task, negative) {
  scoreElements.naturalness.textContent = naturalness;
  scoreElements.task.textContent = task;
  scoreElements.negative.textContent = negative;
}

function predictModel(model, { taskType, summaryFeatures }) {
  const input = buildModelInput(model, taskType, summaryFeatures);
  let prediction = 0;

  if (model.kind === "linear") {
    prediction = model.intercept + model.coef.reduce((sum, coefficient, index) => {
      return sum + coefficient * input[index];
    }, 0);
  } else if (model.kind === "randomForest") {
    prediction = average(model.trees.map((tree) => predictTree(tree, input)));
  } else if (model.kind === "gradientBoosting") {
    prediction =
      model.initialPrediction +
      model.learningRate * model.trees.reduce((sum, tree) => sum + predictTree(tree, input), 0);
  }

  return clampScore(prediction);
}

function buildModelInput(model, taskType, summaryFeatures) {
  const values = [];

  if (model.includeTaskType) {
    model.taskTypeCategories.forEach((category) => {
      values.push(category === taskType ? 1 : 0);
    });
  }

  model.featureColumns.forEach((column, index) => {
    const rawValue = Number(summaryFeatures[column] ?? 0);
    const scale = model.featureScaler.scale[index] || 1;
    values.push((rawValue - model.featureScaler.mean[index]) / scale);
  });

  return values;
}

function predictTree(tree, input) {
  let node = 0;

  while (tree.childrenLeft[node] !== -1) {
    const featureIndex = tree.feature[node];
    node = input[featureIndex] <= tree.threshold[node]
      ? tree.childrenLeft[node]
      : tree.childrenRight[node];
  }

  return tree.value[node];
}

export function calculateExpressionVariables(blendshapes) {
  const smileLeft = score(blendshapes, "mouthSmileLeft");
  const smileRight = score(blendshapes, "mouthSmileRight");
  const jawOpen = score(blendshapes, "jawOpen");
  const mouthPress = average([
    score(blendshapes, "mouthPressLeft"),
    score(blendshapes, "mouthPressRight")
  ]);
  const mouthPucker = score(blendshapes, "mouthPucker");
  const mouthFunnel = score(blendshapes, "mouthFunnel");
  const browUp = score(blendshapes, "browInnerUp");
  const browDown = average([
    score(blendshapes, "browDownLeft"),
    score(blendshapes, "browDownRight")
  ]);
  const eyeBlinkLeft = score(blendshapes, "eyeBlinkLeft");
  const eyeBlinkRight = score(blendshapes, "eyeBlinkRight");

  const frameMotion = getFrameMotion(blendshapes);
  motionWindow.push(frameMotion);
  if (motionWindow.length > 24) {
    motionWindow.shift();
  }

  const smile = average([smileLeft, smileRight]);
  const blink = average([eyeBlinkLeft, eyeBlinkRight]);
  const mouthTension = clamp01(mouthPress * 0.75 + mouthPucker * 0.15 + mouthFunnel * 0.1);
  const asymmetry = clamp01(
    Math.abs(smileLeft - smileRight) * 0.55 +
      Math.abs(eyeBlinkLeft - eyeBlinkRight) * 0.25 +
      Math.abs(score(blendshapes, "browOuterUpLeft") - score(blendshapes, "browOuterUpRight")) * 0.2
  );
  const browActivity = clamp01(browUp * 0.55 + browDown * 0.45);
  const expressionActivity = clamp01(
    smile * 0.3 + jawOpen * 0.2 + browActivity * 0.2 + blink * 0.12 + mouthTension * 0.18
  );
  const motionAverage = average(motionWindow);

  const stillness = 1 - clamp01(motionAverage / 0.035);
  const tensionSignal = clamp01(mouthTension * 0.65 + browDown * 0.2 + asymmetry * 0.15);
  const baselineRigidity = Math.round(clamp01(stillness * 0.58 + tensionSignal * 0.42) * 100);
  const baselineNaturalness = Math.round(
    clamp01(0.72 - baselineRigidity / 170 + expressionActivity * 0.28 - asymmetry * 0.18) * 100
  );

  previousBlendshapeMap = blendshapes;

  return {
    baselineNaturalness,
    baselineRigidity,
    expressionActivity: round(expressionActivity),
    frameMotion: round(frameMotion),
    motionAverage: round(motionAverage),
    asymmetry: round(asymmetry),
    mouthTension: round(mouthTension),
    blink: round(blink),
    smile: round(smile),
    jawOpen: round(jawOpen),
    browActivity: round(browActivity)
  };
}

function getFrameMotion(blendshapes) {
  if (!previousBlendshapeMap) {
    return 0;
  }

  const trackedKeys = [
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
    "cheekSquintRight"
  ];

  return average(
    trackedKeys.map((key) => Math.abs(score(blendshapes, key) - score(previousBlendshapeMap, key)))
  );
}

function updateMetrics(variables) {
  snapshotButton.disabled = false;
  recordButton.disabled = false;
  ui.activity.textContent = variables.expressionActivity.toFixed(3);
  ui.motion.textContent = variables.motionAverage.toFixed(3);
  ui.asymmetry.textContent = variables.asymmetry.toFixed(3);
  ui.mouthTension.textContent = variables.mouthTension.toFixed(3);
  ui.blink.textContent = variables.blink.toFixed(3);
}

function resetMetrics() {
  resetExpressionState();
  ui.activity.textContent = "--";
  ui.motion.textContent = "--";
  ui.asymmetry.textContent = "--";
  ui.mouthTension.textContent = "--";
  ui.blink.textContent = "--";
  snapshotButton.disabled = true;
  recordButton.disabled = !isRecording;

  if (isAnalyzing) {
    updateAnalysisWithoutSample();
  }
}

function updateAnalysisWithoutSample() {
  const elapsed = performance.now() - analysisStartTime;
  const progress = Math.min(100, (elapsed / ANALYSIS_DURATION_MS) * 100);
  progressElement.style.width = `${progress}%`;
  statusElement.textContent = "얼굴을 찾는 중입니다.";

  if (elapsed >= ANALYSIS_DURATION_MS) {
    finishServiceAnalysis();
  }
}

function resetExpressionState() {
  previousBlendshapeMap = null;
  latestVariables = null;
  motionWindow.length = 0;
}

function toggleRecording() {
  if (isRecording) {
    isRecording = false;
    recordButton.textContent = "연속 수집";
    updateCollectorState();
    return;
  }

  if (!getTargets()) {
    statusElement.textContent = "회귀 학습용 정답 라벨을 0-100으로 입력하세요.";
    return;
  }

  isRecording = true;
  lastSampleTime = 0;
  recordButton.textContent = "수집 중지";
  updateCollectorState();
}

function collectSample() {
  const targets = getTargets();

  if (!latestVariables || !targets) {
    statusElement.textContent = "얼굴 인식 후 자연도/경직도 라벨을 0-100으로 입력하세요.";
    return;
  }

  const row = {
    sampleId: datasetRows.length + 1,
    timestampIso: new Date().toISOString(),
    targetNaturalness: targets.targetNaturalness,
    targetRigidity: targets.targetRigidity
  };

  FEATURE_KEYS.forEach((key) => {
    row[key] = latestVariables[key];
  });

  datasetRows.push(row);
  lastSampleTime = performance.now();
  updateCollectorState();
  console.log("regression_sample", row);
}

function getTargets() {
  if (!naturalnessTargetInput.value.trim() || !rigidityTargetInput.value.trim()) {
    return null;
  }

  const targetNaturalness = Number(naturalnessTargetInput.value);
  const targetRigidity = Number(rigidityTargetInput.value);

  if (!Number.isFinite(targetNaturalness) || !Number.isFinite(targetRigidity)) {
    return null;
  }

  if (
    targetNaturalness < 0 ||
    targetNaturalness > 100 ||
    targetRigidity < 0 ||
    targetRigidity > 100
  ) {
    return null;
  }

  return {
    targetNaturalness,
    targetRigidity
  };
}

function updateCollectorState() {
  ui.sampleCount.textContent = String(datasetRows.length);
  ui.recordState.textContent = isRecording ? "수집" : "대기";
  downloadButton.disabled = datasetRows.length === 0;
  clearButton.disabled = datasetRows.length === 0;
}

function downloadCsv() {
  if (!datasetRows.length) {
    return;
  }

  const headers = ["sampleId", "timestampIso", "targetNaturalness", "targetRigidity", ...FEATURE_KEYS];
  const csvRows = [
    headers.join(","),
    ...datasetRows.map((row) => headers.map((header) => csvCell(row[header])).join(","))
  ];
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `faceease-regression-${Date.now()}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function clearDataset() {
  datasetRows.length = 0;
  isRecording = false;
  recordButton.textContent = "연속 수집";
  updateCollectorState();
}

function csvCell(value) {
  const text = String(value ?? "");

  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replaceAll('"', '""')}"`;
  }

  return text;
}

function score(blendshapes, key) {
  return blendshapes[key] ?? 0;
}

function average(values) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function formatScore(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  return String(Math.round(value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function summarizeFrameVariables(frameRows) {
  const summary = {};

  SUMMARY_FEATURE_KEYS.forEach((key) => {
    const values = frameRows
      .map((row) => row[key])
      .filter((value) => Number.isFinite(value));

    if (!values.length) {
      summary[`mean_${key}`] = 0;
      summary[`std_${key}`] = 0;
      return;
    }

    const mean = average(values);
    const variance = average(values.map((value) => (value - mean) ** 2));
    summary[`mean_${key}`] = Math.round(mean * 1000000) / 1000000;
    summary[`std_${key}`] = Math.round(Math.sqrt(variance) * 1000000) / 1000000;
  });

  return summary;
}

export function calculateFinalNaturalnessScore(
  taskType,
  predictedTaskPerformanceScore,
  predictedNegativeExpressionScore
) {
  if (!POSITIVE_TASK_TYPES.includes(taskType)) {
    return null;
  }

  return Math.round(
    clampScore(predictedTaskPerformanceScore * 0.7 + (100 - predictedNegativeExpressionScore) * 0.3)
  );
}

export function generateTaskFeedback({
  taskType,
  predictedTaskPerformanceScore,
  predictedNegativeExpressionScore,
  finalNaturalnessScore,
  summaryFeatures
}) {
  const messages = [];
  const mean = (key) => Number(summaryFeatures?.[`mean_${key}`] ?? 0);
  const std = (key) => Number(summaryFeatures?.[`std_${key}`] ?? 0);

  if (
    POSITIVE_TASK_TYPES.includes(taskType) &&
    finalNaturalnessScore !== null &&
    finalNaturalnessScore >= 75 &&
    predictedTaskPerformanceScore >= 75 &&
    predictedNegativeExpressionScore <= 35
  ) {
    return ["현재 표정은 선택한 과제와 비교적 잘 맞고 부정 표정 신호도 낮게 나타났습니다."];
  }

  if (predictedTaskPerformanceScore < 55) {
    messages.push("요청된 표정 과제 수행도가 낮아 목표 표정이 명확하게 전달되지 않을 수 있습니다.");
  }

  if (predictedNegativeExpressionScore > 65) {
    messages.push("어색하거나 과하게 웃는 부정 표정 패턴과 유사도가 높게 나타났습니다.");
  }

  if (taskType === "natural_smile") {
    const thresholds = FEEDBACK_THRESHOLDS.natural_smile;

    if (mean("smile") < thresholds.weakSmileMax) {
      messages.push("입꼬리 움직임이 부족해 미소가 약하게 보일 수 있습니다.");
    }
    if (mean("blink") < thresholds.lowEyeBlinkMax && mean("browActivity") < thresholds.lowEyeBrowMax) {
      messages.push("눈가 움직임이 적어 다소 딱딱한 미소로 보일 수 있습니다.");
    }
    if (mean("asymmetry") > thresholds.asymmetryHigh) {
      messages.push("좌우 미소 균형이 불안정해 어색하게 보일 수 있습니다.");
    }
    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push("입 주변 긴장도가 높아 자연스러운 미소보다 굳은 표정으로 보일 수 있습니다.");
    }
  } else if (taskType === "listening") {
    const thresholds = FEEDBACK_THRESHOLDS.listening;

    if (std("expressionActivity") > thresholds.restlessExpressionStdHigh) {
      messages.push("표정 변화가 커서 안정적인 경청 표정보다 산만해 보일 수 있습니다.");
    }
    if (mean("smile") > thresholds.overSmileHigh) {
      messages.push("미소가 과하게 나타나 경청 표정보다는 웃는 표정에 가깝게 보일 수 있습니다.");
    }
    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push("입 주변 긴장도가 높아 다소 굳어 보일 수 있습니다.");
    }
    if (mean("blink") < thresholds.lowBlinkMax) {
      messages.push("눈가 움직임이 너무 적으면 반응이 부족해 보일 수 있습니다.");
    }
    if (mean("jawOpen") > thresholds.jawOpenHigh) {
      messages.push("입이 벌어진 시간이 길어 경청 표정보다 말하거나 놀란 표정에 가깝게 보일 수 있습니다.");
    }
  } else if (taskType === "calm") {
    const thresholds = FEEDBACK_THRESHOLDS.calm;

    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push("입 주변 긴장도가 높아 차분하기보다 굳어 보일 수 있습니다.");
    }
    if (mean("browActivity") > thresholds.browActivityHigh) {
      messages.push("눈썹 움직임이 많아 불안정한 인상으로 보일 수 있습니다.");
    }
    if (mean("asymmetry") > thresholds.asymmetryHigh) {
      messages.push("좌우 표정 차이가 커서 표정이 어색하게 보일 수 있습니다.");
    }
    if (
      mean("expressionActivity") > thresholds.expressionActivityHigh ||
      std("frameMotion") > thresholds.frameMotionStdHigh
    ) {
      messages.push("표정 변화가 지나치게 크면 차분한 기본 표정과 다르게 보일 수 있습니다.");
    }
  } else if (taskType === "awkward" || taskType === "over_smile") {
    messages.push("awkward와 over_smile은 서비스 권장 과제가 아니라 부정 패턴 학습용 taskType입니다.");
  }

  if (!messages.length && finalNaturalnessScore !== null) {
    messages.push("현재 표정은 선택한 과제와 비교적 잘 맞고 부정 표정 신호도 낮게 나타났습니다.");
  }

  return messages.slice(0, 4);
}

function clampScore(value) {
  return Math.min(100, Math.max(0, Number(value)));
}
