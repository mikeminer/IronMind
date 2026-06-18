const DEFAULT_IMAGE_QUALITY_WEIGHTS = Object.freeze({
  resolution: 0.2,
  exposure: 0.18,
  contrast: 0.18,
  sharpness: 0.18,
  noise: 0.14,
  artifacts: 0.12
});

function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function clampNumber(value, min, max, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function weightedScore(scores, weights) {
  let total = 0;
  let weightTotal = 0;
  for (const [key, weight] of Object.entries(weights)) {
    total += clamp01(scores[key]) * weight;
    weightTotal += weight;
  }
  return weightTotal ? total / weightTotal : 0;
}

export function scoreResolution(width, height) {
  const pixels = Math.max(0, Number(width || 0) * Number(height || 0));
  if (!pixels) return 0;
  return clamp01(Math.sqrt(pixels) / 1024);
}

export function scoreExposure(stats = {}) {
  const mean = clampNumber(stats.lumaMean, 0, 255, 128);
  const darkRatio = clamp01(stats.darkRatio);
  const brightRatio = clamp01(stats.brightRatio);
  const centered = 1 - Math.abs(mean - 128) / 128;
  const clippingPenalty = clamp01((darkRatio + brightRatio) * 2.5);
  return clamp01(centered * (1 - clippingPenalty));
}

export function scoreContrast(stats = {}) {
  const stdDev = clampNumber(stats.lumaStdDev, 0, 128, 0);
  return clamp01(stdDev / 64);
}

export function scoreSharpness(stats = {}) {
  const laplacianMean = clampNumber(stats.laplacianMean, 0, 255, 0);
  return clamp01(laplacianMean / 18);
}

export function scoreNoise(stats = {}) {
  const noise = clamp01(stats.highFrequencyNoise);
  return clamp01(1 - noise * 1.4);
}

export function scoreArtifacts(stats = {}) {
  const saturationRatio = clamp01(stats.saturationRatio);
  const missingPixelRatio = clamp01(stats.missingPixelRatio);
  return clamp01(1 - saturationRatio * 2 - missingPixelRatio * 3);
}

export function assessImageQuality(input = {}, options = {}) {
  const width = Number(input.width || input.metadata?.width || 0);
  const height = Number(input.height || input.metadata?.height || 0);
  const stats = input.pixelStats || input.stats || {};
  const weights = { ...DEFAULT_IMAGE_QUALITY_WEIGHTS, ...(options.weights || {}) };
  const scores = {
    resolution: input.resolutionScore ?? scoreResolution(width, height),
    exposure: input.exposureScore ?? scoreExposure(stats),
    contrast: input.contrastScore ?? scoreContrast(stats),
    sharpness: input.sharpnessScore ?? scoreSharpness(stats),
    noise: input.noiseScore ?? scoreNoise(stats),
    artifacts: input.artifactScore ?? scoreArtifacts(stats)
  };
  const score = clamp01(input.score ?? weightedScore(scores, weights));
  const minReadyScore = Number(options.minReadyScore ?? input.minReadyScore ?? 0.55);
  const reasons = [];

  if (scores.resolution < 0.5) reasons.push("resolution_below_screening_threshold");
  if (scores.exposure < 0.5) reasons.push("exposure_or_clipping_issue");
  if (scores.contrast < 0.4) reasons.push("contrast_below_screening_threshold");
  if (scores.sharpness < 0.35) reasons.push("possible_blur_or_motion");
  if (scores.noise < 0.45) reasons.push("noise_above_screening_threshold");
  if (scores.artifacts < 0.6) reasons.push("artifact_or_missing_pixel_issue");
  if (score < minReadyScore) reasons.push("overall_quality_below_threshold");

  const screeningReadiness = score >= minReadyScore && reasons.length === 0
    ? "ready_for_model_review"
    : "repeat_or_manual_image_review";

  return {
    kind: "ironmind.image-quality.v1",
    intendedUse: "image_quality_gate_for_screening_triage",
    score,
    screeningReadiness,
    humanReviewRequired: screeningReadiness !== "ready_for_model_review",
    scores: {
      resolutionScore: clamp01(scores.resolution),
      exposureScore: clamp01(scores.exposure),
      contrastScore: clamp01(scores.contrast),
      sharpnessScore: clamp01(scores.sharpness),
      noiseScore: clamp01(scores.noise),
      artifactScore: clamp01(scores.artifacts),
      modalityFitScore: clamp01(input.modalityFitScore ?? 0.85)
    },
    reasons,
    metadata: {
      fileName: input.fileName || null,
      mimeType: input.mimeType || null,
      modality: input.modality || null,
      bodyRegion: input.bodyRegion || null,
      width,
      height,
      pixelCount: width * height
    },
    clinicalTriageImageQuality: {
      score,
      resolutionScore: clamp01(scores.resolution),
      contrastScore: clamp01(scores.contrast),
      noiseScore: clamp01(scores.noise),
      artifactScore: clamp01(scores.artifacts),
      modalityFitScore: clamp01(input.modalityFitScore ?? 0.85)
    }
  };
}
