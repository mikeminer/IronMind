# IronMind CPU Clinical Screening

## Positioning

IronMind can be positioned for DIGITAL-2026-AI-PILOTING-10-SCREENING as a
CPU-efficient clinical imaging triage and orchestration layer. The system is not a
diagnostic replacement for clinicians. It is a decision-support framework that helps
medical centres prioritise cases for specialist review when resources are limited.

The strongest claim is:

> CPU-only, quantized AI screening support for medical imaging triage, with
> uncertainty-aware scoring, explainability, audit trails, and human clinical review.

## Why This Fits The Call

The call asks for piloting AI-based image screening in medical centres, especially for
resource-limited clinical settings, cancer and cardiovascular screening, integration
with PACS/RIS/EHR workflows, clinical validation, privacy, cybersecurity, and
post-deployment monitoring.

IronMind can contribute the local and efficient compute layer:

- CPU-first inference for centres without GPU capacity.
- Quantized model execution for lower-cost deployment.
- Model orchestration above existing medical imaging models.
- Triage scoring instead of black-box positive/negative output.
- Human-in-the-loop routing for uncertain, high-risk, or low-quality cases.
- Audit-ready logs of model version, thresholds, scores, and review decisions.

## Proposed Architecture

```text
Medical image / DICOM
        |
        v
Image quality gate
        |
        v
Medical model adapters
        |
        v
Model orchestration and agreement scoring
        |
        v
Clinical triage score package
        |
        +--> risk score
        +--> confidence score
        +--> uncertainty score
        +--> image quality score
        +--> model agreement score
        +--> explainability score
        |
        v
Human review routing
        |
        v
Clinician feedback and post-deployment monitoring
```

## Scoring Contract

The first reusable contract is implemented in `lib/clinicalScoring.mjs` and exposed
through `POST /v1/clinical/triage`.

Example output:

```json
{
  "kind": "ironmind.clinical-triage.v1",
  "intendedUse": "screening_triage_decision_support",
  "scores": {
    "riskScore": 0.9,
    "confidenceScore": 0.8,
    "uncertaintyScore": 0.2,
    "modelAgreementScore": 0.95,
    "imageQualityScore": 0.92,
    "explainabilityScore": 0.7
  },
  "recommendation": "urgent_specialist_review",
  "humanReviewRequired": true,
  "reviewPriority": "urgent"
}
```

This separates clinical risk from confidence and uncertainty. That distinction matters:
a high-risk output with low confidence should not be treated the same as a high-risk
output with high model agreement, good image quality, and stable explanation.

Example request:

```json
{
  "modality": "xray",
  "bodyRegion": "chest",
  "imageQuality": { "score": 0.92 },
  "explainability": { "score": 0.7, "refs": ["heatmap://case-1"] },
  "modelOutputs": [
    { "modelId": "cxr-risk-a", "riskScore": 0.93, "confidenceScore": 0.82, "uncertaintyScore": 0.18 },
    { "modelId": "cxr-risk-b", "riskScore": 0.88, "confidenceScore": 0.78, "uncertaintyScore": 0.22 }
  ]
}
```

## MVP Work Packages

1. Clinical triage scoring engine
   - Implemented baseline scoring contract.
   - Next: calibrate thresholds by modality and clinical use case.

2. Medical image input layer
   - DICOM ingestion.
   - Image metadata checks.
   - Quality scoring for resolution, contrast, noise, artifacts, and modality fit.

3. Model adapter layer
   - Wrap existing medical imaging models.
   - Normalise model outputs into risk/confidence/uncertainty fields.
   - Track model version and intended use.

4. Explainability layer
   - Store heatmap or region references.
   - Score plausibility and stability of explanation.
   - Present evidence package for clinician review.

5. Human review workflow
   - Route urgent, high-risk, uncertain, low-quality, or model-disagreement cases.
   - Capture clinician feedback.
   - Build post-deployment monitoring datasets.

6. CPU optimization path
   - Quantization.
   - Pruning.
   - Distillation.
   - Native CPU model runner once performance is interactive.

## Clinical And Regulatory Boundaries

The system should be described as triage and decision support. It should not claim
autonomous diagnosis.

Required governance areas:

- GDPR and health data protection.
- Cybersecurity controls for medical data.
- AI Act risk management and human oversight.
- Clinical evaluation and post-market clinical follow-up where applicable.
- Integration with existing clinical systems such as PACS, RIS, and EHR.
- Local validation in participating medical centres.

## Consortium Needs

IronMind alone is not enough for the call. A competitive proposal needs:

- Medical centres in multiple eligible countries.
- Radiology or specialist clinical partners.
- A university or clinical research organisation.
- Medical imaging AI partners with validated models.
- GDPR, cybersecurity, AI Act, and medical device expertise.
- Access to real-world validation workflows and clinical feedback.

## Near-Term Technical Next Steps

- Add a DICOM-safe ingestion prototype.
- Define a model adapter interface for medical imaging models.
- Add a sample triage API endpoint returning the scoring package.
- Add UI panels for risk, confidence, uncertainty, quality, and review priority.
- Keep CPU-only deployment as a differentiator, while using cloud or GPU only where
  clinically and operationally justified.
