import './style.css';

import * as tf from '@tensorflow/tfjs';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const MODEL_URL = '/models/gesture_model/model.json';
const CLASS_NAMES = ['call', 'two_up', 'like', 'fist'];

const INPUT_SIZE = 224;
const CONFIDENCE_THRESHOLD = 0.75;
const PREDICTION_INTERVAL_MS = 180;
const ACTION_COOLDOWN_MS = 1200;
const STABLE_FRAMES = 4;
const BBOX_MARGIN_RATIO = 0.28;

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

const statusEl = document.getElementById('status');
const gestureEl = document.getElementById('gesture');
const confidenceEl = document.getElementById('confidence');
const lastActionEl = document.getElementById('last-action');

const pdfInput = document.getElementById('pdf-input');
const pdfCanvas = document.getElementById('pdf-canvas');
const pdfCtx = pdfCanvas.getContext('2d');
const pageInfo = document.getElementById('page-info');

const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const zoomInBtn = document.getElementById('zoom-in-btn');
const zoomOutBtn = document.getElementById('zoom-out-btn');

const appEl = document.querySelector('.app');
const presentationBtn = document.getElementById('presentation-btn');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');

let handLandmarker = null;
let gestureModel = null;

let pdfDoc = null;
let currentPage = 1;
let zoomFactor = 1;
let rendering = false;
let pendingPage = null;

let lastPredictionAt = 0;
let lastActionAt = 0;
let predictionHistory = [];

const cropCanvas = document.createElement('canvas');
cropCanvas.width = INPUT_SIZE;
cropCanvas.height = INPUT_SIZE;
const cropCtx = cropCanvas.getContext('2d');

function setStatus(message) {
  statusEl.textContent = message;
}

async function loadPdfFromFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  currentPage = 1;
  await renderPage(currentPage);
}

async function renderPage(pageNumber) {
  if (!pdfDoc) return;

  if (rendering) {
    pendingPage = pageNumber;
    return;
  }

  rendering = true;

  const page = await pdfDoc.getPage(pageNumber);
  const pdfWrapper = document.querySelector('.pdf-wrapper');

  const baseViewport = page.getViewport({ scale: 1 });

  const availableWidth = Math.max(320, pdfWrapper.clientWidth - 16);
  const availableHeight = Math.max(240, pdfWrapper.clientHeight - 16);

  const fitScale = Math.min(
    availableWidth / baseViewport.width,
    availableHeight / baseViewport.height
  );

  const finalScale = fitScale * zoomFactor;
  const viewport = page.getViewport({ scale: finalScale });

  const outputScale = window.devicePixelRatio || 1;

  pdfCanvas.width = Math.floor(viewport.width * outputScale);
  pdfCanvas.height = Math.floor(viewport.height * outputScale);

  pdfCanvas.style.width = `${Math.floor(viewport.width)}px`;
  pdfCanvas.style.height = `${Math.floor(viewport.height)}px`;

  const transform = outputScale !== 1
    ? [outputScale, 0, 0, outputScale, 0, 0]
    : null;

  await page.render({
    canvasContext: pdfCtx,
    viewport,
    transform
  }).promise;

  pageInfo.textContent = `Página ${currentPage} de ${pdfDoc.numPages}`;
  rendering = false;

  if (pendingPage !== null) {
    const nextPending = pendingPage;
    pendingPage = null;
    await renderPage(nextPending);
  }
}

async function nextPage() {
  if (!pdfDoc || currentPage >= pdfDoc.numPages) return;
  currentPage += 1;
  await renderPage(currentPage);
}

async function prevPage() {
  if (!pdfDoc || currentPage <= 1) return;
  currentPage -= 1;
  await renderPage(currentPage);
}

async function zoomIn() {
  zoomFactor = Math.min(zoomFactor + 0.1, 2);
  await renderPage(currentPage);
}

async function zoomOut() {
  zoomFactor = Math.max(zoomFactor - 0.1, 0.5);
  await renderPage(currentPage);
}

function toggleFullscreen() {
  const element = document.documentElement;
  if (!document.fullscreenElement) {
    element.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}

async function executeGesture(gesture) {
  const now = performance.now();
  if (now - lastActionAt < ACTION_COOLDOWN_MS) return;

  if (gesture === 'two_up') {
  await nextPage();
  lastActionEl.textContent = 'Siguiente página';
} else if (gesture === 'fist') {
  await prevPage();
  lastActionEl.textContent = 'Página anterior';
} else if (gesture === 'like') {
  await toggleCameraPanel();
  lastActionEl.textContent = 'Mostrar/Ocultar cámara';
} else if (gesture === 'call') {
  await togglePresentationMode();
  lastActionEl.textContent = 'Modo presentación';
}

  lastActionAt = now;
}

function resizeOverlayToVideo() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;

  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
}

function getBBoxFromLandmarks(landmarks) {
  const width = video.videoWidth;
  const height = video.videoHeight;

  const xs = landmarks.map((point) => point.x * width);
  const ys = landmarks.map((point) => point.y * height);

  let xMin = Math.min(...xs);
  let xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);

  const boxWidth = xMax - xMin;
  const boxHeight = yMax - yMin;
  const margin = Math.max(boxWidth, boxHeight) * BBOX_MARGIN_RATIO;

  xMin = Math.max(0, xMin - margin);
  yMin = Math.max(0, yMin - margin);
  xMax = Math.min(width, xMax + margin);
  yMax = Math.min(height, yMax + margin);

  return {
    x: xMin,
    y: yMin,
    width: Math.max(1, xMax - xMin),
    height: Math.max(1, yMax - yMin)
  };
}

function drawDetection(landmarks, bbox, predictionText) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  overlayCtx.lineWidth = 4;
  overlayCtx.strokeStyle = '#22c55e';
  overlayCtx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

  overlayCtx.fillStyle = '#22c55e';
  for (const point of landmarks) {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * overlay.width, point.y * overlay.height, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }

  overlayCtx.font = '24px Arial';
  overlayCtx.fillStyle = '#22c55e';
  overlayCtx.fillText(predictionText, bbox.x, Math.max(30, bbox.y - 12));
}

function cropHandToCanvas(bbox) {
  cropCtx.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  cropCtx.drawImage(
    video,
    bbox.x,
    bbox.y,
    bbox.width,
    bbox.height,
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE
  );
}

async function predictGestureFromCrop() {
  const prediction = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(cropCanvas)
      .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
      .toFloat();

    // MobileNetV2 preprocess_input:
    // [0, 255] → [-1, 1]
    const normalized = pixels.div(127.5).sub(1);
    const batched = normalized.expandDims(0);

    const output = gestureModel.predict(batched);

    if (Array.isArray(output)) {
      return output[0];
    }

    if (output instanceof tf.Tensor) {
      return output;
    }

    return Object.values(output)[0];
  });

  const probabilities = Array.from(await prediction.data());
  prediction.dispose();

  let bestIndex = 0;
  let bestScore = probabilities[0];

  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > bestScore) {
      bestScore = probabilities[i];
      bestIndex = i;
    }
  }

  console.log(
    CLASS_NAMES.map((name, index) => `${name}: ${probabilities[index]?.toFixed(4)}`).join(' | ')
  );

  return {
    gesture: CLASS_NAMES[bestIndex] ?? `class_${bestIndex}`,
    confidence: bestScore,
    probabilities
  };
}

function updateStablePrediction(gesture) {
  predictionHistory.push(gesture);
  if (predictionHistory.length > STABLE_FRAMES) {
    predictionHistory.shift();
  }

  if (predictionHistory.length < STABLE_FRAMES) return false;
  return predictionHistory.every((item) => item === gesture);
}

async function detectLoop() {
  if (!handLandmarker || !gestureModel || video.readyState < 2) {
    requestAnimationFrame(detectLoop);
    return;
  }

  resizeOverlayToVideo();

  const now = performance.now();
  const result = handLandmarker.detectForVideo(video, now);

  if (!result.landmarks || result.landmarks.length === 0) {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    gestureEl.textContent = 'sin mano';
    confidenceEl.textContent = '---';
    predictionHistory = [];
    requestAnimationFrame(detectLoop);
    return;
  }

  const landmarks = result.landmarks[0];
  const bbox = getBBoxFromLandmarks(landmarks);

  if (now - lastPredictionAt >= PREDICTION_INTERVAL_MS) {
    cropHandToCanvas(bbox);

    const { gesture, confidence } = await predictGestureFromCrop();
    lastPredictionAt = now;

    gestureEl.textContent = gesture;
    confidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;

    const predictionText = `${gesture} ${(confidence * 100).toFixed(0)}%`;
    drawDetection(landmarks, bbox, predictionText);

    const isStable = updateStablePrediction(gesture);

    if (confidence >= CONFIDENCE_THRESHOLD && isStable) {
      await executeGesture(gesture);
    }
  } else {
    drawDetection(landmarks, bbox, gestureEl.textContent);
  }

  requestAnimationFrame(detectLoop);
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: 'user'
    },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });

  await video.play();
}

async function setupMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
}

async function setupGestureModel() {
  gestureModel = await tf.loadGraphModel(MODEL_URL);

  const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
  const warmup = gestureModel.predict(dummy);

  const warmupTensor = Array.isArray(warmup)
    ? warmup[0]
    : warmup instanceof tf.Tensor
      ? warmup
      : Object.values(warmup)[0];

  await warmupTensor.data();

  tf.dispose(warmup);
  dummy.dispose();

  console.log('GraphModel cargado correctamente:', gestureModel);
}

async function init() {
  try {
    setStatus('Abriendo cámara...');
    await setupCamera();

    setStatus('Cargando MediaPipe...');
    await setupMediaPipe();

    setStatus('Cargando modelo MobileNetV2...');
    await setupGestureModel();

    setStatus('Listo');
    detectLoop();
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
  }
}

pdfInput.addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    await loadPdfFromFile(file);
  } catch (error) {
    console.error(error);
    pageInfo.textContent = 'No se pudo cargar el PDF.';
  }
});

async function togglePresentationMode() {
  appEl.classList.toggle('presentation-mode');
  appEl.classList.remove('show-camera');

  await new Promise((resolve) => setTimeout(resolve, 80));
  await renderPage(currentPage);
}

async function toggleCameraPanel() {
  appEl.classList.toggle('show-camera');

  if (appEl.classList.contains('show-camera')) {
    toggleCameraBtn.textContent = 'Ocultar cámara';
  } else {
    toggleCameraBtn.textContent = 'Mostrar cámara';
  }

  await new Promise((resolve) => setTimeout(resolve, 80));
  await renderPage(currentPage);
}

prevBtn.addEventListener('click', prevPage);
nextBtn.addEventListener('click', nextPage);
zoomInBtn.addEventListener('click', zoomIn);
zoomOutBtn.addEventListener('click', zoomOut);
presentationBtn.addEventListener('click', togglePresentationMode);
toggleCameraBtn.addEventListener('click', toggleCameraPanel);

window.addEventListener('resize', () => {
  if (pdfDoc) {
    renderPage(currentPage);
  }
});

document.addEventListener('keydown', async (event) => {
  const key = event.key.toLowerCase();

  if (key === 'p') {
    await togglePresentationMode();
  }

  if (key === 'c') {
    await toggleCameraPanel();
  }
});
init();
