export const DEFAULT_CLINICAL_TRIAGE_THRESHOLDS = Object.freeze({
  minImageQuality: 0.55,
  mediumRisk: 0.45,
  highRisk: 0.75,
  urgentRisk: 0.9,
  minConfidenceForRoutine: 0.65,
  maxUncertaintyForRoutine: 0.35,
  minModelAgreementForRoutine: 0.55,
  minExplainabilityForRoutine: 0.45
});

function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function weightedAverage(items, valueKey, weightKey = "weight", fallback = 0) {
  let total = 0;
  let weightTotal = 0;
  for (const item of items || []) {
    const weight = Math.max(0, Number(item?.[weightKey] ?? 1));
    if (!weight) continue;
    total += clamp01(item?.[valueKey], fallback) * weight;
    weightTotal += weight;
  }
  return weightTotal ? total / weightTotal : fallback;
}

export function scoreImageQuality(metrics = {}) {
  if (metrics.score !== undefined) return clamp01(metrics.score);
  const weights = [
    { value: metrics.resolutionScore, weight: 0.25 },
    { value: metrics.contrastScore, weight: 0.2 },
    { value: metrics.noiseScore, weight: 0.2 },
    { value: metrics.artifactScore, weight: 0.2 },
    { value: metrics.modalityFitScore, weight: 0.15 }
  ];
  return weightedAverage(weights.map((item) => ({ score: item.value, weight: item.weight })), "score", "weight", 0.5);
}

export function scoreExplainability(evidence = {}) {
  if (evidence.score !== undefined) return clamp01(evidence.score);
  const weights = [
    { value: evidence.heatmapAvailabilityScore, weight: 0.25 },
    { value: evidence.regionPlausibilityScore, weight: 0.3 },
    { value: evidence.saliencyStabilityScore, weight: 0.25 },
    { value: evidence.rationaleCompletenessScore, weight: 0.2 }
  ];
  return weightedAverage(weights.map((item) => ({ score: item.value, weight: item.weight })), "score", "weight", 0.5);
}

export function scoreModelAgreement(modelOutputs = []) {
  const risks = modelOutputs
    .map((output) => clamp01(output?.riskScore, NaN))
    .filter(Number.isFinite);
  if (risks.length <= 1) return risks.length ? 1 : 0;

  const mean = risks.reduce((sum, value) => sum + value, 0) / risks.length;
  const variance = risks.reduce((sum, value) => sum + (value - mean) ** 2, 0) / risks.length;
  const stdDev = Math.sqrt(variance);
  return clamp01(1 - stdDev * 2);
}

export function aggregateModelOutputs(modelOutputs = []) {
  const outputs = modelOutputs.filter(Boolean);
  const riskScore = weightedAverage(outputs, "riskScore", "weight", 0);
  const confidenceScore = weightedAverage(outputs, "confidenceScore", "weight", 0);
  const reportedUncertainty = weightedAverage(outputs, "uncertaintyScore", "weight", 1 - confidenceScore);
  const uncertaintyScore = clamp01(reportedUncertainty || (1 - confidenceScore));
  const modelAgreementScore = scoreModelAgreement(outputs);

  return {
    riskScore: clamp01(riskScore),
    confidenceScore: clamp01(confidenceScore),
    uncertaintyScore,
    modelAgreementScore,
    modelCount: outputs.length
  };
}

export function clinicalReviewRecommendation(scores, thresholds = DEFAULT_CLINICAL_TRIAGE_THRESHOLDS) {
  const imageQualityScore = clamp01(scores.imageQualityScore);
  const riskScore = clamp01(scores.riskScore);
  const confidenceScore = clamp01(scores.confidenceScore);
  const uncertaintyScore = clamp01(scores.uncertaintyScore);
  const modelAgreementScore = clamp01(scores.modelAgreementScore);
  const explainabilityScore = clamp01(scores.explainabilityScore);
  const reasons = [];

  if (imageQualityScore < thresholds.minImageQuality) reasons.push("image_quality_below_threshold");
  if (uncertaintyScore > thresholds.maxUncertaintyForRoutine) reasons.push("uncertainty_above_threshold");
  if (confidenceScore < thresholds.minConfidenceForRoutine) reasons.push("confidence_below_threshold");
  if (modelAgreementScore < thresholds.minModelAgreementForRoutine) reasons.push("model_disagreement");
  if (explainabilityScore < thresholds.minExplainabilityForRoutine) reasons.push("explainability_below_threshold");

  if (imageQualityScore < thresholds.minImageQuality) {
    return {
      recommendation: "repeat_or_manual_image_review",
      humanReviewRequired: true,
      reviewPriority: "quality_gate",
      reasons
    };
  }

  if (riskScore >= thresholds.urgentRisk) {
    return {
      recommendation: "urgent_specialist_review",
      humanReviewRequired: true,
      reviewPriority: "urgent",
      reasons: ["risk_above_urgent_threshold", ...reasons]
    };
  }

  if (riskScore >= thresholds.highRisk) {
    return {
      recommendation: "priority_specialist_review",
      humanReviewRequired: true,
      reviewPriority: "high",
      reasons: ["risk_above_high_threshold", ...reasons]
    };
  }

  if (riskScore >= thresholds.mediumRisk || reasons.length) {
    return {
      recommendation: "clinician_review",
      humanReviewRequired: true,
      reviewPriority: riskScore >= thresholds.mediumRisk ? "medium" : "safety_check",
      reasons: riskScore >= thresholds.mediumRisk ? ["risk_above_medium_threshold", ...reasons] : reasons
    };
  }

  return {
    recommendation: "routine_queue",
    humanReviewRequired: false,
    reviewPriority: "routine",
    reasons: []
  };
}

export function createClinicalTriage(input = {}, options = {}) {
  const thresholds = { ...DEFAULT_CLINICAL_TRIAGE_THRESHOLDS, ...(options.thresholds || {}) };
  const aggregate = aggregateModelOutputs(input.modelOutputs || []);
  const imageQualityScore = scoreImageQuality(input.imageQuality || {});
  const explainabilityScore = scoreExplainability(input.explainability || {});
  const scores = {
    ...aggregate,
    imageQualityScore,
    explainabilityScore
  };
  const review = clinicalReviewRecommendation(scores, thresholds);

  return {
    kind: "ironmind.clinical-triage.v1",
    intendedUse: "screening_triage_decision_support",
    patientCareNotice: "This output is not a diagnosis and requires clinical governance according to local procedures.",
    scores,
    thresholds,
    ...review,
    evidence: {
      modality: input.modality || null,
      bodyRegion: input.bodyRegion || null,
      modelIds: (input.modelOutputs || []).map((output) => output.modelId || "unknown"),
      explanationRefs: input.explainability?.refs || []
    }
  };
}
