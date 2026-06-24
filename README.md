# IronMind

IronMind is a small native CPU inference engine and local AI orchestration layer optimized for one focused model path at a time. It targets ordinary laptops and workstations with 64GB+ RAM, using quantized GGUF weights, RAM/disk KV cache, OpenAI-compatible APIs, and an integrated local chatbot/agent.

The current strategic direction includes CPU-efficient clinical imaging triage: using local, quantized inference and model orchestration to support early screening workflows in resource-limited medical centres.

> Status: pre-alpha. The current bootstrap ships the server, chatbot, installer, and API shape while the native CPU engine is developed. For usable inference today it connects to a local Ollama or llama.cpp-compatible runtime and forces a CPU-only low-latency profile by default.

## Install

```powershell
irm https://raw.githubusercontent.com/mikeminer/IronMind/main/install.ps1 | iex
```

Then pull the default CPU/RAM model and start IronMind:

```powershell
ollama pull qwen3-coder:30b
ironmind
```

Open http://127.0.0.1:4141.

## Why IronMind

IronMind targets machines like an i9 laptop with 64GB RAM: enough memory for strong quantized 14B-32B models, but not enough for DwarfStar's DeepSeek V4 Flash class.
The context target is 100k+ tokens, with RAM used for the active working set and disk used for persistent prefix/KV state.

The first recommended target is `qwen3-coder:30b` because it is useful for coding agents, has a practical memory footprint, and gives IronMind a narrow model path to optimize around.

For clinical screening, the value proposition is different: IronMind is the CPU-first orchestration and triage layer around specialised medical imaging models. It is intended to prioritise cases for human review, not to replace clinical judgement.

## CPU-Only Low-Latency Mode

IronMind now defaults to a CPU-only runtime policy for interactive inference:

- forces Ollama GPU layers to zero with `num_gpu: 0`;
- keeps the model warm with `keep_alive`;
- caps interactive context with `IRONMIND_CPU_CTX` to reduce prefill latency;
- caps default output with `IRONMIND_CPU_MAX_TOKENS`;
- sets CPU thread and batch options through `IRONMIND_CPU_THREADS` and `IRONMIND_CPU_BATCH`;
- injects a runtime system message so the local chatbot does not claim GPU acceleration.

The default profile is `low-latency`: `ctx=4096`, `max_tokens=128`, `batch=128`, and CPU threads selected from the local machine. Use `balanced` or `full-context` only when longer context matters more than response latency.

### Native ik_llama.cpp CPU Runtime

For CPU-first local inference, IronMind can run through `ik_llama.cpp` instead of
Ollama. IronMind stays the API/UI/orchestration layer, while `ik_llama.cpp`
provides the CPU-optimised GGUF runtime. This is the intended path for making
IronMind native local inference on CPU without GPU dependency.

`ik_llama.cpp` is useful because it focuses on CPU performance, row-interleaved
quant packing, newer IQK/Trellis quantization paths, and faster CPU prompt
processing.

Build `ik_llama.cpp` for CPU:

```powershell
git clone https://github.com/ikawrakow/ik_llama.cpp C:\ai\ik_llama.cpp
cd C:\ai\ik_llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release
```

Run IronMind with a managed `ik_llama.cpp` server:

```powershell
$env:IRONMIND_BACKEND="ik_llama"
$env:IRONMIND_IK_LLAMA_SERVER="C:\ai\ik_llama.cpp\build\bin\Release\llama-server.exe"
$env:IRONMIND_IK_LLAMA_MODEL="C:\models\qwen3.gguf"
$env:IRONMIND_CPU_ONLY="true"
ironmind
```

IronMind starts `llama-server` with CPU-only flags, including `--n-gpu-layers 0`,
`--ctx-size`, `--threads`, and `--batch-size` from the active CPU profile. If you
prefer to run the server yourself, set `IRONMIND_BACKEND=llama` and
`IRONMIND_LLAMA_URL=http://127.0.0.1:8080`.

The integration plan is tracked in `docs/IK_LLAMA_NATIVE_RUNTIME.md`.

## Design

IronMind is organized around a vertical local stack:

- CPU/RAM inference target: one supported model family first, not a generic model zoo.
- Prompt renderer: model-specific chat and tool formatting.
- Connected chatbot: local browser UI streamed from the IronMind server.
- OpenAI-compatible API: `/v1/models` and `/v1/chat/completions`.
- Clinical screening APIs: `/v1/clinical/screening`, `/v1/clinical/image/quality`, and `/v1/clinical/triage`.
- Agent path: future `/v1/responses` and Anthropic-compatible `/v1/messages`.
- KV strategy: RAM session first, disk persistence next.
- Bench/eval discipline: token throughput, prompt rendering checks, and regression traces.

## CPU Clinical Screening

IronMind can be positioned for medical imaging pilots as a CPU-only, quantized AI screening assistant for triage and decision support.

The clinical path is documented in `docs/CPU_CLINICAL_SCREENING.md` and starts with a browser image-quality gate plus a reusable triage scoring contract implemented in `lib/imageQuality.mjs` and `lib/clinicalScoring.mjs`.

The intended output is not a diagnosis. It is a structured screening package:

- image quality score;
- clinical risk score;
- model confidence;
- uncertainty score;
- model agreement score;
- explainability score;
- human review recommendation and priority.

The first MVP step is available in the local UI as `Clinical Image Triage`: load a PNG, JPEG, or WebP image, or run the built-in demo case, and IronMind computes CPU-side readiness metrics, a non-diagnostic demo screening package, review priority, and an exportable case JSON.

Expected clinical impact after validation includes earlier cancer and cardiovascular screening support, faster prioritisation of critical cases, lower radiology workload pressure, remote screening support for underserved regions, and reduced delays through PACS/RIS/EHR-oriented workflows.

Image quality gate API:

```powershell
curl http://127.0.0.1:4141/v1/clinical/image/quality `
  -H "Content-Type: application/json" `
  -d '{ "fileName": "case.png", "mimeType": "image/png", "modality": "xray", "bodyRegion": "chest", "width": 1600, "height": 1400, "pixelStats": { "lumaMean": 126, "lumaStdDev": 58, "laplacianMean": 24, "highFrequencyNoise": 0.08, "saturationRatio": 0.01, "darkRatio": 0.01, "brightRatio": 0.01 } }'
```

End-to-end screening case API:

```powershell
curl http://127.0.0.1:4141/v1/clinical/screening `
  -H "Content-Type: application/json" `
  -d '{ "image": { "fileName": "case.png", "mimeType": "image/png", "modality": "xray", "bodyRegion": "chest", "width": 1600, "height": 1400, "pixelStats": { "lumaMean": 126, "lumaStdDev": 58, "laplacianMean": 24, "highFrequencyNoise": 0.08, "saturationRatio": 0.01, "darkRatio": 0.01, "brightRatio": 0.01 } } }'
```

Example:

```json
{
  "kind": "ironmind.clinical-triage.v1",
  "intendedUse": "screening_triage_decision_support",
  "scores": {
    "riskScore": 0.905,
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

The API endpoint is:

```powershell
curl http://127.0.0.1:4141/v1/clinical/triage `
  -H "Content-Type: application/json" `
  -d '{ "modality": "xray", "bodyRegion": "chest", "imageQuality": { "score": 0.92 }, "explainability": { "score": 0.7, "refs": ["heatmap://case-1"] }, "modelOutputs": [{ "modelId": "cxr-risk-a", "riskScore": 0.93, "confidenceScore": 0.82, "uncertaintyScore": 0.18 }, { "modelId": "cxr-risk-b", "riskScore": 0.88, "confidenceScore": 0.78, "uncertaintyScore": 0.22 }] }'
```

This direction fits pilots where small clinics or underserved regions need low-cost screening support, local review queues, and auditable human-in-the-loop workflows. A real clinical deployment still requires medical partners, validated imaging models, GDPR and cybersecurity controls, AI Act risk management, clinical evaluation, and integration with systems such as PACS, RIS, and EHR.

## Run

```powershell
ironmind --model qwen3-coder:30b --ctx 131072 --kv-disk-dir C:\IronMindKV --kv-disk-space-mb 16384
```

Environment variables:

```text
IRONMIND_MODEL=qwen3-coder:30b
IRONMIND_CTX=131072
IRONMIND_KV_DISK_DIR=C:\IronMindKV
IRONMIND_KV_DISK_SPACE_MB=16384
IRONMIND_PORT=4141
IRONMIND_OLLAMA_URL=http://127.0.0.1:11434
IRONMIND_LLAMA_URL=http://127.0.0.1:8080
IRONMIND_BACKEND=ik_llama
IRONMIND_IK_LLAMA_SERVER=C:\ai\ik_llama.cpp\build\bin\Release\llama-server.exe
IRONMIND_IK_LLAMA_MODEL=C:\models\qwen3.gguf
IRONMIND_IK_LLAMA_HOST=127.0.0.1
IRONMIND_IK_LLAMA_PORT=8080
IRONMIND_NATIVE_MODEL=C:\path\to\model.gguf
IRONMIND_CPU_ONLY=true
IRONMIND_CPU_PROFILE=low-latency
IRONMIND_CPU_THREADS=10
IRONMIND_CPU_BATCH=128
IRONMIND_CPU_CTX=4096
IRONMIND_CPU_MAX_TOKENS=128
IRONMIND_CPU_KEEP_ALIVE=30m
IRONMIND_NATIVE_CACHE_MB=512
IRONMIND_NATIVE_CACHE_MAX_TENSOR_MB=64
```

Use an internal SSD/NVMe for `IRONMIND_KV_DISK_DIR`. A removable SD card is usually too slow for 100k+ token KV-cache restore/write patterns.
The native cache keeps raw quantized GGUF tensors resident under a bounded budget; norm tensors are pinned, while very large matrices stay file-backed and rely on streaming reads plus the OS page cache.

Health check:

```powershell
ironmind doctor
```

Inspect a GGUF before trying to run it:

```powershell
ironmind inspect C:\path\to\model.gguf
ironmind map C:\path\to\model.gguf
ironmind native C:\path\to\model.gguf
ironmind native C:\path\to\model.gguf --decode 0 --ctx 1
ironmind tokenize C:\path\to\model.gguf "Ciao mondo"
```

Build the native core:

```powershell
npm run native:build
.\build\Release\ironmind-inspect.exe C:\path\to\model.gguf
.\build\Release\ironmind-native.exe C:\path\to\model.gguf
.\build\Release\ironmind-native.exe C:\path\to\model.gguf --decode 0 --ctx 1
npm run native:test
```

The `--decode` path is still correctness-first: rows are dequantized into a small work buffer, while dot products use runtime AVX2/AVX512F dispatch and hot tensors use bounded residency.
`ironmind native` reports the selected SIMD backend and residency stats. On supported x86 CPUs it dispatches to AVX2 or AVX512F at runtime.
Set `IRONMIND_BACKEND=native` with `IRONMIND_NATIVE_MODEL` to route server completions through the built CPU backend. The native path can also discover local Ollama GGUF blobs for the selected model, but it is still correctness-first and slow on large models. In `auto` mode, IronMind uses native only for an explicit GGUF path or configured `nativeModel`; otherwise it keeps the Ollama-compatible backend for interactive chat.

Run the built-in 100-question evaluation suite:

```powershell
npm run eval -- stats
npm run eval -- run --model qwen3:14b --limit 10
```

OpenAI-compatible example:

```powershell
curl http://127.0.0.1:4141/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{ "model": "qwen3-coder:30b", "messages": [{"role": "user", "content": "Explain IronMind in one paragraph."}], "stream": false }'
```

## Native Engine Roadmap

The bootstrap intentionally keeps the API, UI, session model, and model target stable while the native CPU core is built.

Planned core milestones:

1. GGUF metadata reader for the selected target model. Implemented as `ironmind inspect` and native `ironmind-inspect`.
2. Qwen3 prompt rendering tests. Implemented in `lib/qwen3Prompt.mjs`.
3. Disk context store for 100k+ token sessions. Implemented as prompt-prefix snapshots today, native KV payload next.
4. Tokenizer compatibility. Implemented for Qwen3 GGUF `gpt2/qwen2` BPE.
5. Tensor mapping. Implemented for dense Qwen3 and Qwen3MoE tensor names.
6. Scalar RMSNorm, RoPE, softmax, and attention kernels. Implemented in JS and native C.
7. Native dense decode step with RAM KV cache save/restore. Implemented in `native/ironmind_forward.c`.
8. Native GGUF tensor loader and runtime gate. Implemented in `native/ironmind_gguf.c` and `ironmind native`.
9. Quantized CPU matmul scalar baseline plus SIMD dot dispatch. Implemented for F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K in `native/ironmind_quant.c`; Q4_K/Q6_K use direct quantized dot without a full F32 row buffer, and AVX2/AVX512F dot kernels are in `native/ironmind_simd*.c`.
10. MoE top-k routing and expert mixing primitive. Implemented in `native/ironmind_moe.c`.
11. GGUF-backed Qwen3 decode wiring. Implemented in `native/ironmind_qwen3.c`.
12. Logit/token reference comparison. Implemented in `native/ironmind_qwen3_test.c`.
13. Evaluation suite for physics, mathematics, and defensive security. Implemented as IronMind Eval 100.
14. Native IronKV payload integration for full server sessions. Implemented with `IRONKV1` session files, JSON sidecars, and native KV save/restore payloads.
15. Native tool-call replay and canonicalization. Implemented with deterministic tool schemas, assistant tool-call replay, tool-response rendering, and generated `<tool_call>` extraction.
16. Replace the bootstrap runtime path with the IronMind CPU backend once full-token decode is fast enough for interactive use. Implemented as an `auto|ollama|native` backend selector with native GGUF prefill/generation and Ollama fallback.

## License

MIT
