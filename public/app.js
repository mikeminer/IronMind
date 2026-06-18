const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendEl = document.querySelector("#send");
const clearEl = document.querySelector("#clear");
const statusEl = document.querySelector("#status");
const modelEl = document.querySelector("#model");
const ctxEl = document.querySelector("#ctx");
const kvDiskEl = document.querySelector("#kvDisk");
const thinkEl = document.querySelector("#think");
const clinicalFileEl = document.querySelector("#clinicalFile");
const clinicalPreviewEl = document.querySelector("#clinicalPreview");
const clinicalAnalyzeEl = document.querySelector("#clinicalAnalyze");
const clinicalModalityEl = document.querySelector("#clinicalModality");
const clinicalRegionEl = document.querySelector("#clinicalRegion");
const clinicalResultEl = document.querySelector("#clinicalResult");
const qualityScoreEl = document.querySelector("#qualityScore");
const qualityReadinessEl = document.querySelector("#qualityReadiness");
const resolutionScoreEl = document.querySelector("#resolutionScore");
const sharpnessScoreEl = document.querySelector("#sharpnessScore");
const qualityReasonsEl = document.querySelector("#qualityReasons");

let clinicalImagePayload = null;
let clinicalPreviewUrl = null;

const messages = [
  {
    role: "assistant",
    content: "IronMind is checking the local backend."
  }
];

function setStatus(text, danger = false) {
  statusEl.textContent = text;
  statusEl.style.color = danger ? "var(--danger)" : "var(--muted)";
}

function render() {
  messagesEl.innerHTML = "";
  for (const message of messages) {
    const item = document.createElement("article");
    item.className = `message ${message.role}`;

    const role = document.createElement("span");
    role.className = "role";
    role.textContent = message.role;

    const content = document.createElement("div");
    content.textContent = message.content;

    item.append(role, content);
    messagesEl.append(item);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatScore(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;
}

function formatReadiness(value) {
  if (!value) return "-";
  return String(value).replace(/_/g, " ");
}

function setClinicalQualityState(result = {}) {
  const scores = result.scores || {};
  qualityScoreEl.textContent = formatScore(result.score);
  qualityReadinessEl.textContent = formatReadiness(result.screeningReadiness);
  resolutionScoreEl.textContent = formatScore(scores.resolutionScore);
  sharpnessScoreEl.textContent = formatScore(scores.sharpnessScore);
  qualityReasonsEl.textContent = result.reasons?.length
    ? result.reasons.map(formatReadiness).join(", ")
    : "Ready for model review.";
  clinicalResultEl.classList.toggle("needs-review", Boolean(result.humanReviewRequired));
  clinicalResultEl.classList.toggle("ready", result.screeningReadiness === "ready_for_model_review");
}

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Unable to load image"));
    };
    image.src = url;
  });
}

function lumaForPixel(data, offset) {
  return data[offset] * 0.2126 + data[offset + 1] * 0.7152 + data[offset + 2] * 0.0722;
}

function imagePayloadFromCanvas(file, image, maxSide = 384) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas image analysis unavailable");
  canvas.width = width;
  canvas.height = height;
  context.drawImage(image, 0, 0, width, height);

  const data = context.getImageData(0, 0, width, height).data;
  const pixelCount = width * height;
  const luma = new Float32Array(pixelCount);
  let lumaSum = 0;
  let lumaSqSum = 0;
  let dark = 0;
  let bright = 0;
  let saturated = 0;
  let missing = 0;

  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    const y = lumaForPixel(data, offset);
    luma[pixel] = y;
    lumaSum += y;
    lumaSqSum += y * y;
    if (y < 16) dark += 1;
    if (y > 240) bright += 1;
    if (data[offset + 3] < 8) missing += 1;
    if (
      data[offset] <= 3 || data[offset] >= 252 ||
      data[offset + 1] <= 3 || data[offset + 1] >= 252 ||
      data[offset + 2] <= 3 || data[offset + 2] >= 252
    ) saturated += 1;
  }

  let laplacianSum = 0;
  let noiseSum = 0;
  let interiorCount = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const center = luma[index];
      const north = luma[index - width];
      const south = luma[index + width];
      const west = luma[index - 1];
      const east = luma[index + 1];
      const localMean = (north + south + west + east) / 4;
      laplacianSum += Math.abs(center * 4 - north - south - west - east);
      noiseSum += Math.abs(center - localMean) / 255;
      interiorCount += 1;
    }
  }

  const lumaMean = lumaSum / pixelCount;
  const variance = Math.max(0, lumaSqSum / pixelCount - lumaMean * lumaMean);

  return {
    fileName: file.name,
    mimeType: file.type || "application/octet-stream",
    width: sourceWidth,
    height: sourceHeight,
    modality: clinicalModalityEl.value,
    bodyRegion: clinicalRegionEl.value.trim() || null,
    pixelStats: {
      lumaMean,
      lumaStdDev: Math.sqrt(variance),
      laplacianMean: interiorCount ? laplacianSum / interiorCount : 0,
      highFrequencyNoise: interiorCount ? noiseSum / interiorCount : 1,
      saturationRatio: saturated / pixelCount,
      missingPixelRatio: missing / pixelCount,
      darkRatio: dark / pixelCount,
      brightRatio: bright / pixelCount
    }
  };
}

async function analyzeClinicalImage() {
  if (!clinicalImagePayload) return;
  clinicalAnalyzeEl.disabled = true;
  qualityReasonsEl.textContent = "Analyzing image quality...";
  clinicalResultEl.classList.remove("ready", "needs-review");

  try {
    const response = await fetch("/v1/clinical/image/quality", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        ...clinicalImagePayload,
        modality: clinicalModalityEl.value,
        bodyRegion: clinicalRegionEl.value.trim() || null
      })
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    setClinicalQualityState(await response.json());
  } catch (error) {
    qualityReasonsEl.textContent = `Quality gate error: ${error.message}`;
    clinicalResultEl.classList.add("needs-review");
  } finally {
    clinicalAnalyzeEl.disabled = false;
  }
}

async function prepareClinicalImage(file) {
  if (!file) return;
  clinicalAnalyzeEl.disabled = true;
  qualityScoreEl.textContent = "-";
  qualityReadinessEl.textContent = "-";
  resolutionScoreEl.textContent = "-";
  sharpnessScoreEl.textContent = "-";
  qualityReasonsEl.textContent = "Loading image...";
  clinicalResultEl.classList.remove("ready", "needs-review");

  try {
    const { image, url } = await loadImageElement(file);
    if (clinicalPreviewUrl) URL.revokeObjectURL(clinicalPreviewUrl);
    clinicalPreviewUrl = url;
    clinicalPreviewEl.src = url;
    clinicalPreviewEl.alt = file.name;
    clinicalPreviewEl.parentElement.classList.add("loaded");
    clinicalImagePayload = imagePayloadFromCanvas(file, image);
    clinicalAnalyzeEl.disabled = false;
    await analyzeClinicalImage();
  } catch (error) {
    clinicalImagePayload = null;
    qualityReasonsEl.textContent = `Image load error: ${error.message}`;
    clinicalResultEl.classList.add("needs-review");
  }
}

async function readNdjson(response, onObject) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) onObject(JSON.parse(trimmed));
    }
  }

  const tail = buffer.trim();
  if (tail) onObject(JSON.parse(tail));
}

async function sendPrompt(text) {
  messages.push({ role: "user", content: text });
  const assistant = { role: "assistant", content: "" };
  messages.push(assistant);
  render();

  sendEl.disabled = true;
  setStatus("Generating...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: modelEl.value.trim(),
        ctx: Number(ctxEl.value),
        think: thinkEl.checked,
        messages: messages
          .filter((message) => message.content.trim())
          .map((message) => ({ role: message.role, content: message.content }))
      })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    await readNdjson(response, (event) => {
      if (event.type === "delta") {
        assistant.content += event.content;
        render();
      }
      if (event.type === "done") {
        const snapshot = event.contextSnapshot;
        const cached = snapshot ? `, disk=${snapshot.estimatedTokens}t` : "";
        setStatus(`Done. prompt=${event.promptEvalCount || 0}, output=${event.evalCount || 0}${cached}`);
      }
    });
  } catch (error) {
    assistant.content = `Backend error: ${error.message}`;
    setStatus("Backend unavailable", true);
    render();
  } finally {
    sendEl.disabled = false;
    promptEl.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";
  sendPrompt(text);
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    composer.requestSubmit();
  }
});

clearEl.addEventListener("click", () => {
  messages.splice(0, messages.length, {
    role: "assistant",
    content: "New IronMind session."
  });
  setStatus("Ready");
  render();
});

clinicalFileEl.addEventListener("change", () => {
  const file = clinicalFileEl.files?.[0];
  if (file) prepareClinicalImage(file);
});

clinicalAnalyzeEl.addEventListener("click", () => {
  analyzeClinicalImage();
});

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    modelEl.value = health.model;
    ctxEl.value = health.context;
    kvDiskEl.value = health.kvDiskDir;
    if (messages.length === 1 && messages[0].role === "assistant") {
      messages[0].content = `IronMind is ready with ${health.model}.`;
      render();
    }
    const files = health.contextStore?.files || 0;
    setStatus(`Backend: ${health.backend}; disk context files=${files}`);
  } catch {
    setStatus("Health check failed", true);
  }
}

render();
loadHealth();
