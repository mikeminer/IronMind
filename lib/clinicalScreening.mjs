import { createClinicalTriage } from "./clinicalScoring.mjs";
import { assessImageQuality } from "./imageQuality.mjs";

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

function stableHash(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function caseIdFor(input, imageQuality, createdAt) {
  const metadata = imageQuality.metadata || {};
  const source = [
    input.caseId || "",
    metadata.fileName || "",
    metadata.modality || "",
    metadata.bodyRegion || "",
    metadata.width || 0,
    metadata.height || 0,
    createdAt.slice(0, 10)
  ].join("|");
  return `IM-${stableHash(source).slice(0, 10).toUpperCase()}`;
}

function imageInput(input = {}) {
  return input.image && typeof input.image === "object" ? input.image : input;
}

export function createDemoModelOutputs(input = {}, imageQuality = assessImageQuality(input)) {
  const image = imageInput(input);
  const stats = image.pixelStats || image.stats || {};
  const scores = imageQuality.scores || {};
  const quality = clamp01(imageQuality.score, 0.5);
  const exposureStress = Math.abs(clampNumber(stats.lumaMean, 0, 255, 128) - 128) / 128;
  const contrastSignal = clamp01(clampNumber(stats.lumaStdDev, 0, 128, 0) / 72);
  const edgeSignal = clamp01(clampNumber(stats.laplacianMean, 0, 255, 0) / 32);
  const blurSignal = 1 - clamp01(scores.sharpnessScore, 0.5);
  const noiseSignal = 1 - clamp01(scores.noiseScore, 0.5);
  const artifactSignal = 1 - clamp01(scores.artifactScore, 0.5);
  const qualityPenalty = 1 - quality;
  const baseRisk = clamp01(
    0.18 +
    contrastSignal * 0.18 +
    edgeSignal * 0.12 +
    exposureStress * 0.1 +
    blurSignal * 0.12 +
    noiseSignal * 0.12 +
    artifactSignal * 0.1
  );
  const adjustedRisk = imageQuality.screeningReadiness === "ready_for_model_review"
    ? baseRisk
    : clamp01(Math.min(baseRisk, 0.42) + qualityPenalty * 0.08);
  const confidence = clamp01(0.42 + quality * 0.42 + clamp01(scores.contrastScore, 0.5) * 0.1 - noiseSignal * 0.08);
  const uncertainty = clamp01(0.18 + qualityPenalty * 0.45 + noiseSignal * 0.12 + blurSignal * 0.1);

  return [
    {
      modelId: "ironmind-demo-screening-cpu-a",
      intendedUse: "non_diagnostic_screening_demo",
      riskScore: adjustedRisk,
      confidenceScore: confidence,
      uncertaintyScore: uncertainty,
      weight: 1
    },
    {
      modelId: "ironmind-demo-screening-cpu-b",
      intendedUse: "non_diagnostic_screening_demo",
      riskScore: clamp01(adjustedRisk * 0.92 + edgeSignal * 0.05 + exposureStress * 0.03),
      confidenceScore: clamp01(confidence - qualityPenalty * 0.08),
      uncertaintyScore: clamp01(uncertainty + Math.abs(edgeSignal - contrastSignal) * 0.05),
      weight: 0.9
    }
  ];
}

export function createDemoExplainability(input = {}, imageQuality = assessImageQuality(input)) {
  const scores = imageQuality.scores || {};
  const score = clamp01(
    clamp01(scores.contrastScore, 0.5) * 0.28 +
    clamp01(scores.sharpnessScore, 0.5) * 0.28 +
    clamp01(scores.resolutionScore, 0.5) * 0.2 +
    clamp01(scores.artifactScore, 0.5) * 0.14 +
    clamp01(imageQuality.score, 0.5) * 0.1
  );
  return {
    score,
    heatmapAvailabilityScore: score,
    regionPlausibilityScore: clamp01(scores.contrastScore, 0.5),
    saliencyStabilityScore: clamp01(scores.sharpnessScore, 0.5),
    rationaleCompletenessScore: clamp01(scores.resolutionScore, 0.5),
    refs: [
      "quality://contrast",
      "quality://sharpness",
      "quality://resolution"
    ]
  };
}

function nextActionsFor(triage, imageQuality) {
  if (triage.reviewPriority === "quality_gate") {
    return [
      "Repeat image acquisition or request manual image-quality review.",
      "Do not route to automated screening until quality issues are resolved."
    ];
  }
  if (triage.reviewPriority === "urgent") {
    return [
      "Place case in urgent specialist review queue.",
      "Require clinician sign-off before any patient-facing decision."
    ];
  }
  if (triage.reviewPriority === "high") {
    return [
      "Prioritise specialist review.",
      "Attach model scores, uncertainty, and explanation references to the review package."
    ];
  }
  if (triage.humanReviewRequired) {
    return [
      "Route to clinician review because one or more safety thresholds were triggered.",
      "Use confidence, uncertainty, and agreement fields to guide review order."
    ];
  }
  return imageQuality.screeningReadiness === "ready_for_model_review"
    ? ["Place case in routine screening queue with audit trail attached."]
    : ["Hold case until image quality is reviewed."];
}

export function createClinicalScreeningCase(input = {}, options = {}) {
  const image = imageInput(input);
  const createdAt = options.createdAt || input.createdAt || new Date().toISOString();
  const imageQuality = assessImageQuality(image, input.qualityOptions || options.qualityOptions || {});
  const modelOutputs = Array.isArray(input.modelOutputs) && input.modelOutputs.length
    ? input.modelOutputs
    : createDemoModelOutputs(image, imageQuality);
  const explainability = input.explainability || createDemoExplainability(image, imageQuality);
  const triage = createClinicalTriage({
    modality: image.modality || input.modality,
    bodyRegion: image.bodyRegion || input.bodyRegion,
    imageQuality: imageQuality.clinicalTriageImageQuality,
    explainability,
    modelOutputs
  }, {
    thresholds: input.thresholds || options.thresholds
  });
  const caseId = input.caseId || caseIdFor(input, imageQuality, createdAt);

  return {
    kind: "ironmind.clinical-screening-case.v1",
    caseId,
    createdAt,
    intendedUse: "screening_triage_decision_support",
    productStatus: "mvp_non_diagnostic_decision_support",
    patientCareNotice: "This output is not a diagnosis. It supports screening triage and requires clinical governance, validation, and clinician review.",
    imageQuality,
    modelOutputs,
    explainability,
    triage,
    reviewQueue: {
      caseId,
      priority: triage.reviewPriority,
      recommendation: triage.recommendation,
      humanReviewRequired: triage.humanReviewRequired,
      reasons: triage.reasons,
      nextActions: nextActionsFor(triage, imageQuality)
    },
    audit: {
      modelAdapterMode: Array.isArray(input.modelOutputs) && input.modelOutputs.length ? "provided_model_outputs" : "demo_cpu_adapter",
      modelIds: modelOutputs.map((output) => output.modelId || "unknown"),
      thresholds: triage.thresholds,
      imageMetadata: imageQuality.metadata
    },
    exportFileName: `${caseId}.ironmind-screening.json`
  };
}
