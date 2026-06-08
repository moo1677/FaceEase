import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import "./styles.css";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const BROWSER_MODEL_PATH = "/models/faceease_models.json";
const ANALYSIS_DURATION_MS = 5000;
const CALM_BASELINE_SCALE = 0.8;
const CALM_BASELINE_SAMPLE_LIMIT = 90;
const CALM_BASELINE_MIN_SAMPLES = 30;
const CALM_BASELINE_DURATION_MS = 3000;
const NEUTRAL_BASELINE_TASK_TYPE = "neutral_baseline";
const CALM_BASELINE_KEYS = [
  "smile",
  "jawOpen",
  "browActivity",
  "mouthTension",
  "blink",
  "asymmetry",
  "browDown",
];

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const analyzeButton = document.querySelector("#analyzeButton");
const taskTabContainer = document.querySelector(".task-tabs");
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
  grade: document.querySelector("#naturalnessGrade"),
  naturalness: document.querySelector("#naturalnessScore"),
  task: document.querySelector("#taskScore"),
  negative: document.querySelector("#negativeScore"),
  negativePatternProbability: document.querySelector(
    "#negativePatternProbability",
  ),
  serviceSampleCount: document.querySelector("#serviceSampleCount"),
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
  "browActivity",
];
export const SUMMARY_FEATURE_KEYS = [
  ...FEATURE_KEYS,
  "baselineNaturalness",
  "baselineRigidity",
];
export const POSITIVE_TASK_TYPES = ["natural_smile", "listening", "calm"];
export const GRADE_LABELS = {
  very_natural: "매우 자연스러움",
  mostly_natural: "대체로 자연스러움",
  slightly_awkward: "다소 어색함",
  needs_improvement: "개선 필요",
};
export const FEEDBACK_THRESHOLDS = {
  natural_smile: {
    weakSmileMax: 0.04,
    lowEyeBlinkMax: 0.08,
    lowEyeBrowMax: 0.045,
    asymmetryHigh: 0.07,
    mouthTensionHigh: 0.25,
  },
  listening: {
    overSmileHigh: 0.12,
    mouthTensionHigh: 0.11,
    lowBlinkMax: 0.06,
    restlessExpressionStdHigh: 0.04,
    jawOpenHigh: 0.07,
  },
  calm: {
    mouthTensionHigh: 0.13,
    browActivityHigh: 0.22,
    asymmetryHigh: 0.05,
    expressionActivityHigh: 0.13,
    frameMotionStdHigh: 0.03,
  },
};
const SAMPLE_INTERVAL_MS = 500;
const TASK_PROMPTS = {
  neutral_baseline: "무표정으로 얼굴 기준값을 잡아주세요",
  natural_smile: "자연스러운 미소를 지어보세요",
  listening: "경청하는 표정을 지어보세요",
  calm: "차분한 표정을 지어보세요",
};

const ui = {
  sampleCount: document.querySelector("#sampleCount"),
  recordState: document.querySelector("#recordState"),
  activity: document.querySelector("#activityValue"),
  motion: document.querySelector("#motionValue"),
  asymmetry: document.querySelector("#asymmetryValue"),
  mouthTension: document.querySelector("#mouthTensionValue"),
  blink: document.querySelector("#blinkValue"),
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
let selectedTaskType = NEUTRAL_BASELINE_TASK_TYPE;
let isAnalyzing = false;
let analysisStartTime = 0;
let lastAnalysisSampleTime = 0;
let calmBaseline = null;
let calmBaselineStartTime = 0;
let isCollectingNeutralBaseline = false;
const motionWindow = [];
const calmBaselineSamples = [];
const datasetRows = [];
const analysisRows = [];

startButton.disabled = true;
startButton.addEventListener("click", startCamera);
analyzeButton.addEventListener("click", startServiceAnalysis);
taskButtons.forEach((button) => {
  button.addEventListener("click", () =>
    selectTaskType(button.dataset.taskType),
  );
});
snapshotButton?.addEventListener("click", collectSample);
recordButton?.addEventListener("click", toggleRecording);
downloadButton?.addEventListener("click", downloadCsv);
clearButton?.addEventListener("click", clearDataset);

updateTaskNavigation();
initFaceLandmarker();
loadBrowserModels();

async function initFaceLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_PATH,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    drawingUtils = new DrawingUtils(context);
    startButton.disabled = false;
    statusElement.textContent = browserModels
      ? "카메라를 시작하면 분석할 수 있습니다."
      : "분석 모델을 불러오는 중입니다.";
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
    console.error("브라우저 분석 모델 로딩 실패:", error);
    statusElement.textContent =
      "분석 모델 로딩 실패. npm run ml:export를 실행했는지 확인하세요.";
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
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    resizeCanvasToVideo();

    isRunning = true;
    startButton.disabled = true;
    statusElement.textContent =
      "무표정 버튼을 눌러 얼굴 기준값 수집을 시작하세요.";
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

  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== lastVideoTime
  ) {
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

  if (
    !width ||
    !height ||
    (canvas.width === width && canvas.height === height)
  ) {
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

  if (!isAnalyzing && selectedTaskType !== NEUTRAL_BASELINE_TASK_TYPE) {
    statusElement.textContent =
      "얼굴 인식 중입니다. 5초 분석을 시작할 수 있습니다.";
  }
  drawingUtils.drawConnectors(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    {
      color: "#24d18b",
      lineWidth: 2,
    },
  );
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
    color: "#ff7a59",
    lineWidth: 2,
  });
  drawingUtils.drawConnectors(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    {
      color: "#5aa9ff",
      lineWidth: 1.5,
    },
  );
  drawingUtils.drawConnectors(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    {
      color: "#5aa9ff",
      lineWidth: 1.5,
    },
  );
}

function processResult(result) {
  const categories = result.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) {
    resetMetrics();
    return;
  }

  const blendshapes = Object.fromEntries(
    categories.map((category) => [category.categoryName, category.score]),
  );

  if (isCollectingNeutralBaseline && !isAnalyzing) {
    updateCalmBaseline(calculateCalmBaselineValues(blendshapes));
  }

  const variables = calculateExpressionVariables(blendshapes, {
    calmBaseline: hasCalmBaseline() ? calmBaseline : null,
  });
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
      browActivity: variables.browActivity,
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
  if (taskType === NEUTRAL_BASELINE_TASK_TYPE) {
    startNeutralBaselineCollection();
    return;
  }

  if (taskType !== NEUTRAL_BASELINE_TASK_TYPE && !hasCalmBaseline()) {
    selectedTaskType = NEUTRAL_BASELINE_TASK_TYPE;
    taskPromptElement.textContent = TASK_PROMPTS[NEUTRAL_BASELINE_TASK_TYPE];
    statusElement.textContent =
      "먼저 무표정 버튼을 눌러 얼굴 기준값을 잡아주세요.";
    updateTaskNavigation();
    resetServiceResult();
    return;
  }

  selectedTaskType = taskType;
  taskPromptElement.textContent = TASK_PROMPTS[taskType];
  updateTaskNavigation();
  resetServiceResult();
}

function startNeutralBaselineCollection() {
  if (!isRunning) {
    statusElement.textContent = "카메라를 먼저 시작하세요.";
    return;
  }

  selectedTaskType = NEUTRAL_BASELINE_TASK_TYPE;
  taskPromptElement.textContent = TASK_PROMPTS[NEUTRAL_BASELINE_TASK_TYPE];
  calmBaseline = null;
  calmBaselineStartTime = performance.now();
  calmBaselineSamples.length = 0;
  isCollectingNeutralBaseline = true;
  resetExpressionState();
  resetServiceResult();
  statusElement.textContent = "무표정을 유지하세요. 얼굴 기준값을 수집합니다.";
  updateTaskNavigation();
  updateServiceControls();
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

  if (!hasCalmBaseline()) {
    statusElement.textContent =
      "먼저 무표정 버튼을 눌러 얼굴 기준값을 잡은 뒤 분석하세요.";
    return;
  }

  if (selectedTaskType === NEUTRAL_BASELINE_TASK_TYPE) {
    selectTaskType("natural_smile");
    return;
  }

  resetExpressionState();
  analysisRows.length = 0;
  isAnalyzing = true;
  analysisStartTime = performance.now();
  lastAnalysisSampleTime = 0;
  progressElement.style.width = "0%";
  scoreElements.serviceSampleCount.textContent = "0";
  feedbackList.innerHTML = "<li>분석 중입니다.</li>";
  setScoreText("--", "--", "--", "--");
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
    statusElement.textContent =
      "분석할 수 있는 얼굴 feature가 충분하지 않습니다.";
    feedbackList.innerHTML =
      "<li>얼굴이 화면 중앙에 보이도록 맞춘 뒤 다시 시도하세요.</li>";
    progressElement.style.width = "0%";
    return;
  }

  const summaryFeatures = summarizeFrameVariables(analysisRows);
  const taskScore = predictModel(browserModels.scoreModel, {
    taskType: selectedTaskType,
    summaryFeatures,
  });
  const negativePatternScore = calculateNegativePatternScore(summaryFeatures);
  const finalScore = Math.round(
    clampScore(taskScore * 0.6 + negativePatternScore * 0.4),
  );
  const naturalnessGrade = gradeFromScore(finalScore);
  const feedback = generateTaskFeedback({
    taskType: selectedTaskType,
    predictedScore: finalScore,
    taskScore,
    negativePatternScore,
    naturalnessGrade,
    summaryFeatures,
  });

  setScoreText(
    formatGrade(naturalnessGrade),
    formatScore(finalScore),
    formatScore(taskScore),
    formatScore(negativePatternScore),
  );
  feedbackList.innerHTML = feedback
    .map((message) => `<li>${escapeHtml(message)}</li>`)
    .join("");
  statusElement.textContent = "분석 완료";
  progressElement.style.width = "100%";
  console.log("faceease_service_result", {
    taskType: selectedTaskType,
    taskScore,
    negativePatternScore,
    finalScore,
    naturalnessGrade,
    summaryFeatures,
    feedback,
  });
}

function updateServiceControls() {
  analyzeButton.disabled =
    !faceLandmarker ||
    !browserModels ||
    isAnalyzing ||
    !hasCalmBaseline() ||
    selectedTaskType === NEUTRAL_BASELINE_TASK_TYPE;
  taskButtons.forEach((button) => {
    button.disabled = isAnalyzing;
  });
}

function resetServiceResult() {
  setScoreText("--", "--", "--", "--");
  scoreElements.serviceSampleCount.textContent = "0";
  feedbackList.innerHTML = "<li>분석 결과가 여기에 표시됩니다.</li>";
  progressElement.style.width = "0%";
}

function setScoreText(grade, naturalness, task, negative) {
  scoreElements.grade.textContent = grade;
  scoreElements.naturalness.textContent = naturalness;
  if (scoreElements.task) {
    scoreElements.task.textContent = task;
  }
  if (scoreElements.negative) {
    scoreElements.negative.textContent = negative;
  }
}

function predictModel(model, { taskType, summaryFeatures }) {
  const input = buildModelInput(model, taskType, summaryFeatures);
  let prediction = 0;

  if (model.kind === "linear") {
    prediction =
      model.intercept +
      model.coef.reduce((sum, coefficient, index) => {
        return sum + coefficient * input[index];
      }, 0);
  } else if (model.kind === "randomForest") {
    prediction = average(model.trees.map((tree) => predictTree(tree, input)));
  } else if (model.kind === "gradientBoosting") {
    prediction =
      model.initialPrediction +
      model.learningRate *
        model.trees.reduce((sum, tree) => sum + predictTree(tree, input), 0);
  }

  return clampScore(prediction);
}

function predictClassifierProbabilities(model, { taskType, summaryFeatures }) {
  const input = buildModelInput(model, taskType, summaryFeatures);

  if (model.kind === "logisticClassifier") {
    return logisticProbabilities(model, input);
  }

  if (model.kind === "randomForestClassifier") {
    const summed = model.trees.reduce(
      (totals, tree) => {
        const probabilities = normalizeProbabilities(predictTree(tree, input));
        return totals.map(
          (total, index) => total + (probabilities[index] ?? 0),
        );
      },
      model.classes.map(() => 0),
    );
    return probabilitiesByClass(model.classes, normalizeProbabilities(summed));
  }

  if (model.kind === "gradientBoostingClassifier") {
    return gradientBoostingClassifierProbabilities(model, input);
  }

  return probabilitiesByClass(model.classes ?? [], []);
}

function logisticProbabilities(model, input) {
  if (model.classes.length === 2 && model.coef.length === 1) {
    const raw =
      model.intercept[0] +
      model.coef[0].reduce((sum, coefficient, index) => {
        return sum + coefficient * input[index];
      }, 0);
    const positiveProbability = sigmoid(raw);
    return probabilitiesByClass(model.classes, [
      1 - positiveProbability,
      positiveProbability,
    ]);
  }

  const rawScores = model.coef.map((coefficients, classIndex) => {
    return (
      model.intercept[classIndex] +
      coefficients.reduce((sum, coefficient, featureIndex) => {
        return sum + coefficient * input[featureIndex];
      }, 0)
    );
  });
  return probabilitiesByClass(model.classes, softmax(rawScores));
}

function gradientBoostingClassifierProbabilities(model, input) {
  if (model.classes.length === 2) {
    const raw =
      model.initialRawPredictions[0] +
      model.learningRate *
        model.trees.reduce((sum, tree) => sum + predictTree(tree, input), 0);
    const positiveProbability = sigmoid(raw);
    return probabilitiesByClass(model.classes, [
      1 - positiveProbability,
      positiveProbability,
    ]);
  }

  const rawScores = [...model.initialRawPredictions];
  model.trees.forEach((stage) => {
    stage.forEach((tree, classIndex) => {
      rawScores[classIndex] += model.learningRate * predictTree(tree, input);
    });
  });
  return probabilitiesByClass(model.classes, softmax(rawScores));
}

function buildModelInput(model, taskType, summaryFeatures) {
  const values = [];
  const row = {
    taskType,
    ...summaryFeatures,
  };

  if (model.categoricalFeatureColumns?.length) {
    model.categoricalFeatureColumns.forEach((column) => {
      const categories = model.categoricalCategories?.[column] ?? [];
      categories.forEach((category) => {
        values.push(String(category) === String(row[column] ?? "") ? 1 : 0);
      });
    });
  } else if (model.includeTaskType && model.taskTypeCategories) {
    model.taskTypeCategories.forEach((category) => {
      values.push(category === taskType ? 1 : 0);
    });
  }

  const numericColumns = model.numericFeatureColumns ?? model.featureColumns;
  const scaler = model.numericScaler ?? model.featureScaler;
  numericColumns.forEach((column, index) => {
    const rawValue = Number(row[column] ?? 0);
    const scale = scaler?.scale?.[index] || 1;
    const mean = scaler?.mean?.[index] || 0;
    values.push((rawValue - mean) / scale);
  });

  return values;
}

function predictTree(tree, input) {
  let node = 0;

  while (tree.childrenLeft[node] !== -1) {
    const featureIndex = tree.feature[node];
    node =
      input[featureIndex] <= tree.threshold[node]
        ? tree.childrenLeft[node]
        : tree.childrenRight[node];
  }

  return tree.value[node];
}

function probabilitiesByClass(classes, probabilities) {
  return Object.fromEntries(
    classes.map((className, index) => [
      String(className),
      clampProbability(probabilities[index] ?? 0),
    ]),
  );
}

function normalizeProbabilities(values) {
  const numbers = values.map((value) => Number(value) || 0);
  const total = numbers.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return numbers.map(() => 1 / Math.max(1, numbers.length));
  }
  return numbers.map((value) => value / total);
}

function topProbabilityClass(probabilities) {
  return Object.entries(probabilities).reduce(
    (best, current) => {
      return current[1] > best[1] ? current : best;
    },
    ["normal", -1],
  )[0];
}

export function calculateExpressionVariables(blendshapes, options = {}) {
  const raw = calculateCalmBaselineValues(blendshapes);
  const smile = adjustForCalmBaseline("smile", raw.smile, options.calmBaseline);
  const jawOpen = adjustForCalmBaseline(
    "jawOpen",
    raw.jawOpen,
    options.calmBaseline,
  );
  const browActivity = adjustForCalmBaseline(
    "browActivity",
    raw.browActivity,
    options.calmBaseline,
  );
  const mouthTension = adjustForCalmBaseline(
    "mouthTension",
    raw.mouthTension,
    options.calmBaseline,
  );
  const blink = adjustForCalmBaseline("blink", raw.blink, options.calmBaseline);
  const asymmetry = adjustForCalmBaseline(
    "asymmetry",
    raw.asymmetry,
    options.calmBaseline,
  );
  const browDown = adjustForCalmBaseline(
    "browDown",
    raw.browDown,
    options.calmBaseline,
  );
  const frameMotion = getFrameMotion(blendshapes);
  motionWindow.push(frameMotion);
  if (motionWindow.length > 24) {
    motionWindow.shift();
  }

  const expressionActivity = clamp01(
    smile * 0.3 +
      jawOpen * 0.2 +
      browActivity * 0.2 +
      blink * 0.12 +
      mouthTension * 0.18,
  );
  const motionAverage = average(motionWindow);

  const stillness = 1 - clamp01(motionAverage / 0.035);
  const tensionSignal = clamp01(
    mouthTension * 0.65 + browDown * 0.2 + asymmetry * 0.15,
  );
  const baselineRigidity = Math.round(
    clamp01(stillness * 0.58 + tensionSignal * 0.42) * 100,
  );
  const baselineNaturalness = Math.round(
    clamp01(
      0.72 -
        baselineRigidity / 170 +
        expressionActivity * 0.28 -
        asymmetry * 0.18,
    ) * 100,
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
    browActivity: round(browActivity),
  };
}

function calculateCalmBaselineValues(blendshapes) {
  const smileLeft = score(blendshapes, "mouthSmileLeft");
  const smileRight = score(blendshapes, "mouthSmileRight");
  const jawOpen = score(blendshapes, "jawOpen");
  const mouthPress = average([
    score(blendshapes, "mouthPressLeft"),
    score(blendshapes, "mouthPressRight"),
  ]);
  const mouthPucker = score(blendshapes, "mouthPucker");
  const mouthFunnel = score(blendshapes, "mouthFunnel");
  const browUp = score(blendshapes, "browInnerUp");
  const browDown = average([
    score(blendshapes, "browDownLeft"),
    score(blendshapes, "browDownRight"),
  ]);
  const eyeBlinkLeft = score(blendshapes, "eyeBlinkLeft");
  const eyeBlinkRight = score(blendshapes, "eyeBlinkRight");

  const smile = average([smileLeft, smileRight]);
  const blink = average([eyeBlinkLeft, eyeBlinkRight]);
  const mouthTension = clamp01(
    mouthPress * 0.75 + mouthPucker * 0.15 + mouthFunnel * 0.1,
  );
  const asymmetry = clamp01(
    Math.abs(smileLeft - smileRight) * 0.55 +
      Math.abs(eyeBlinkLeft - eyeBlinkRight) * 0.25 +
      Math.abs(
        score(blendshapes, "browOuterUpLeft") -
          score(blendshapes, "browOuterUpRight"),
      ) *
        0.2,
  );
  const browActivity = clamp01(browUp * 0.55 + browDown * 0.45);

  return {
    smile,
    jawOpen,
    browActivity,
    mouthTension,
    blink,
    asymmetry,
    browDown,
  };
}

function updateCalmBaseline(values) {
  calmBaselineSamples.push(values);
  if (calmBaselineSamples.length > CALM_BASELINE_SAMPLE_LIMIT) {
    calmBaselineSamples.shift();
  }

  calmBaseline = Object.fromEntries(
    CALM_BASELINE_KEYS.map((key) => [
      key,
      average(calmBaselineSamples.map((sample) => sample[key] ?? 0)),
    ]),
  );

  const elapsed = performance.now() - calmBaselineStartTime;
  const elapsedSeconds = Math.min(
    CALM_BASELINE_DURATION_MS / 1000,
    elapsed / 1000,
  );

  if (hasCalmBaseline()) {
    isCollectingNeutralBaseline = false;
    if (selectedTaskType === NEUTRAL_BASELINE_TASK_TYPE) {
      selectTaskType("natural_smile");
      statusElement.textContent =
        "얼굴 기준값이 잡혔습니다. 분석할 표정을 선택하세요.";
    }
  } else {
    statusElement.textContent = `무표정 기준값을 잡는 중입니다. ${elapsedSeconds.toFixed(1)}/${(CALM_BASELINE_DURATION_MS / 1000).toFixed(0)}초`;
  }

  updateTaskNavigation();
  updateServiceControls();
}

function adjustForCalmBaseline(key, value, baseline) {
  if (!baseline) {
    return clamp01(value);
  }

  return clamp01(value - (baseline[key] ?? 0) * CALM_BASELINE_SCALE);
}

function hasCalmBaseline() {
  return Boolean(
    calmBaseline &&
    calmBaselineSamples.length >= CALM_BASELINE_MIN_SAMPLES &&
    performance.now() - calmBaselineStartTime >= CALM_BASELINE_DURATION_MS,
  );
}

function updateTaskNavigation() {
  const baselineReady = hasCalmBaseline();
  taskTabContainer.classList.toggle("is-baseline-mode", !baselineReady);

  taskButtons.forEach((button) => {
    const isBaselineTask =
      button.dataset.taskType === NEUTRAL_BASELINE_TASK_TYPE;
    button.classList.toggle(
      "is-hidden",
      baselineReady ? isBaselineTask : !isBaselineTask,
    );
    button.classList.toggle(
      "is-active",
      button.dataset.taskType === selectedTaskType,
    );
  });
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
    "cheekSquintRight",
  ];

  return average(
    trackedKeys.map((key) =>
      Math.abs(score(blendshapes, key) - score(previousBlendshapeMap, key)),
    ),
  );
}

function updateMetrics(variables) {
  if (snapshotButton) {
    snapshotButton.disabled = false;
  }
  if (recordButton) {
    recordButton.disabled = false;
  }
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
  if (snapshotButton) {
    snapshotButton.disabled = true;
  }
  if (recordButton) {
    recordButton.disabled = !isRecording;
  }

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
    if (recordButton) {
      recordButton.textContent = "연속 수집";
    }
    updateCollectorState();
    return;
  }

  if (!getTargets()) {
    statusElement.textContent = "회귀 학습용 정답 라벨을 0-100으로 입력하세요.";
    return;
  }

  isRecording = true;
  lastSampleTime = 0;
  if (recordButton) {
    recordButton.textContent = "수집 중지";
  }
  updateCollectorState();
}

function collectSample() {
  const targets = getTargets();

  if (!latestVariables || !targets) {
    statusElement.textContent =
      "얼굴 인식 후 자연도/경직도 라벨을 0-100으로 입력하세요.";
    return;
  }

  if (!hasCalmBaseline()) {
    statusElement.textContent =
      "먼저 무표정으로 얼굴 기준값을 잡은 뒤 수집하세요.";
    return;
  }

  const row = {
    sampleId: datasetRows.length + 1,
    timestampIso: new Date().toISOString(),
    targetNaturalness: targets.targetNaturalness,
    targetRigidity: targets.targetRigidity,
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
  if (!naturalnessTargetInput || !rigidityTargetInput) {
    return null;
  }

  if (
    !naturalnessTargetInput.value.trim() ||
    !rigidityTargetInput.value.trim()
  ) {
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
    targetRigidity,
  };
}

function updateCollectorState() {
  if (ui.sampleCount) {
    ui.sampleCount.textContent = String(datasetRows.length);
  }
  if (ui.recordState) {
    ui.recordState.textContent = isRecording ? "수집" : "대기";
  }
  if (downloadButton) {
    downloadButton.disabled = datasetRows.length === 0;
  }
  if (clearButton) {
    clearButton.disabled = datasetRows.length === 0;
  }
}

function downloadCsv() {
  if (!datasetRows.length) {
    return;
  }

  const headers = [
    "sampleId",
    "timestampIso",
    "targetNaturalness",
    "targetRigidity",
    ...FEATURE_KEYS,
  ];
  const csvRows = [
    headers.join(","),
    ...datasetRows.map((row) =>
      headers.map((header) => csvCell(row[header])).join(","),
    ),
  ];
  const blob = new Blob([csvRows.join("\n")], {
    type: "text/csv;charset=utf-8",
  });
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
  if (recordButton) {
    recordButton.textContent = "연속 수집";
  }
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

function clampProbability(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maxValue));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
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

function formatGrade(value) {
  return GRADE_LABELS[value] ?? "--";
}

function gradeFromScore(value) {
  if (value >= 80) {
    return "very_natural";
  }
  if (value >= 60) {
    return "mostly_natural";
  }
  if (value >= 40) {
    return "slightly_awkward";
  }
  return "needs_improvement";
}

function formatProbability(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  return `${Math.round(value * 100)}%`;
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

// negativePatternScore: 100 = 부정패턴 없음(좋음), 0 = 부정패턴 강함(나쁨)
export function calculateFinalNaturalnessScore(
  taskType,
  taskPerformanceScore,
  negativePatternScore,
) {
  if (!POSITIVE_TASK_TYPES.includes(taskType)) {
    return null;
  }

  return Math.round(
    clampScore(taskPerformanceScore * 0.6 + negativePatternScore * 0.4),
  );
}

export function calculateNegativePatternScore(summaryFeatures) {
  // 학습된 이진 분류기 사용 (어색한표정/과하게웃는표정 = 1)
  if (browserModels?.negativePatternModel) {
    const probs = predictClassifierProbabilities(
      browserModels.negativePatternModel,
      {
        taskType: selectedTaskType,
        summaryFeatures,
      },
    );
    const negativeProbability = Number(probs["1"] ?? 0);
    return Math.round(clampScore(100 * (1 - negativeProbability)));
  }

  // rule-based 폴백 (모델 없을 때)
  const mean = (key) => Number(summaryFeatures?.[`mean_${key}`] ?? 0);
  const asymmetrySignal = clamp01(mean("asymmetry") / 0.1);
  const tensionSignal = clamp01(mean("mouthTension") / 0.25);
  const rigidBlinkSignal = clamp01(Math.max(0, 0.06 - mean("blink")) / 0.06);
  const awkwardPenalty = clamp01(
    asymmetrySignal * 0.4 + tensionSignal * 0.4 + rigidBlinkSignal * 0.2,
  );
  const overSmilePenalty = clamp01(Math.max(0, mean("smile") - 0.08) / 0.12);
  const negativePenalty = clamp01(
    awkwardPenalty * 0.5 + overSmilePenalty * 0.5,
  );
  return Math.round(clampScore(100 * (1 - negativePenalty)));
}

export function generateTaskFeedback({
  taskType,
  predictedScore,
  taskScore,
  negativePatternScore,
  naturalnessGrade,
  summaryFeatures,
}) {
  const messages = [];
  const mean = (key) => Number(summaryFeatures?.[`mean_${key}`] ?? 0);
  const std = (key) => Number(summaryFeatures?.[`std_${key}`] ?? 0);

  if (Number.isFinite(negativePatternScore) && negativePatternScore < 60) {
    if (mean("asymmetry") > 0.07 || mean("mouthTension") > 0.2) {
      messages.push(
        "표정이 어색하게 굳어 있습니다. 긴장을 풀고 자연스럽게 표정을 지어보세요.",
      );
    }
    if (mean("smile") > 0.12) {
      messages.push(
        "미소가 과하게 나타납니다. 좀 더 자연스러운 미소를 지어보세요.",
      );
    }
  }

  if (predictedScore < 55) {
    messages.push(
      "최종 자연스러움 score가 낮아 과제 수행 또는 부정 패턴 보정 측면의 개선이 필요합니다.",
    );
  }

  if (taskType === "natural_smile") {
    const thresholds = FEEDBACK_THRESHOLDS.natural_smile;

    if (mean("smile") < thresholds.weakSmileMax) {
      messages.push("입꼬리 움직임이 부족해 미소가 약하게 보일 수 있습니다.");
    }
    if (
      mean("blink") < thresholds.lowEyeBlinkMax &&
      mean("browActivity") < thresholds.lowEyeBrowMax
    ) {
      messages.push("눈가 움직임이 적어 다소 딱딱한 미소로 보일 수 있습니다.");
    }
    if (mean("asymmetry") > thresholds.asymmetryHigh) {
      messages.push("좌우 미소 균형이 불안정해 어색하게 보일 수 있습니다.");
    }
    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push(
        "입 주변 긴장도가 높아 자연스러운 미소보다 굳은 표정으로 보일 수 있습니다.",
      );
    }
  } else if (taskType === "listening") {
    const thresholds = FEEDBACK_THRESHOLDS.listening;

    if (std("expressionActivity") > thresholds.restlessExpressionStdHigh) {
      messages.push(
        "표정 변화가 커서 안정적인 경청 표정보다 산만해 보일 수 있습니다.",
      );
    }
    if (mean("smile") > thresholds.overSmileHigh) {
      messages.push(
        "미소가 과하게 나타나 경청 표정보다는 웃는 표정에 가깝게 보일 수 있습니다.",
      );
    }
    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push("입 주변 긴장도가 높아 다소 굳어 보일 수 있습니다.");
    }
    if (mean("blink") < thresholds.lowBlinkMax) {
      messages.push(
        "눈가 움직임이 너무 적으면 반응이 부족해 보일 수 있습니다.",
      );
    }
    if (mean("jawOpen") > thresholds.jawOpenHigh) {
      messages.push(
        "입이 벌어진 시간이 길어 경청 표정보다 말하거나 놀란 표정에 가깝게 보일 수 있습니다.",
      );
    }
  } else if (taskType === "calm") {
    const thresholds = FEEDBACK_THRESHOLDS.calm;

    if (mean("mouthTension") > thresholds.mouthTensionHigh) {
      messages.push(
        "입 주변 긴장도가 높아 차분하기보다 굳어 보일 수 있습니다.",
      );
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
      messages.push(
        "표정 변화가 지나치게 크면 차분한 기본 표정과 다르게 보일 수 있습니다.",
      );
    }
  }

  if (!messages.length && Number.isFinite(predictedScore)) {
    messages.push(
      "현재 표정은 선택한 과제와 비교적 잘 맞는 score로 예측되었습니다.",
    );
  }

  return messages.slice(0, 4);
}

function clampScore(value) {
  return Math.min(100, Math.max(0, Number(value)));
}
