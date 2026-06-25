# Iurexa

Iurexa is a local Italian legal-support assistant powered by the IronMind CPU inference and orchestration stack. It targets ordinary laptops and workstations, using CPU-only `ik_llama.cpp`, quantized GGUF weights, RAM/disk context storage, OpenAI-compatible APIs, and an integrated local chatbot.

Iurexa helps with legal orientation, issue spotting, document review, clause analysis, drafting, checklists, and preparation. It is designed as an assistant for professional work, not as a replacement for a licensed lawyer or for final legal advice on binding decisions.

> Status: pre-alpha. The current bootstrap ships the server, chatbot, installer, document workflow, API shape, and CPU-only `ik_llama.cpp` runtimes. The fastest native-local path is `ik_embedded` with the persistent daemon, which keeps the GGUF model loaded and reuses the current prompt-prefix KV cache without a `llama-server` HTTP hop.

## Install

```powershell
irm https://raw.githubusercontent.com/mikeminer/IronMind/iurexa/install.ps1 | iex
```

Then start Iurexa:

```powershell
ironmind
```

Open http://127.0.0.1:4141.

## Why Iurexa

Iurexa targets machines like an i9 laptop with 64GB RAM: enough memory for useful quantized local models, but without relying on GPU acceleration.
The context target is 100k+ tokens, with RAM used for the active working set and disk used for persistent prefix/KV state.

The recommended runtime path is `ik_llama.cpp` with a quantized GGUF model. Iurexa exposes this as a simple local chat product while keeping the model file as a runtime detail.

## Iurexa Lite

The current local profile is **Iurexa Lite**:

- product/API model: `iurexa` / **Iurexa**;
- runtime: pinned `ik_llama.cpp` under `third_party/ik_llama.cpp`;
- policy: CPU-only with `--n-gpu-layers 0`;
- default model candidate: 1.7B `IQ4_XS` GGUF calibrated on Italian legal text;
- compact experimental candidate: 1.7B `IQ3_KS`, smaller but less reliable for legal Italian;
- interface: local browser chatbot at http://127.0.0.1:4141 plus OpenAI-compatible endpoints.
- document workflow: local PDF/DOCX/TXT upload, extraction, chunking, multi-document comparison, source citations, structured reports, and RAG for documents longer than the interactive prompt window.

## CPU-Only Low-Latency Mode

Iurexa now defaults to a CPU-only runtime policy for interactive inference:

- forces Ollama GPU layers to zero with `num_gpu: 0`;
- keeps the model warm with `keep_alive`;
- caps interactive context with `IRONMIND_CPU_CTX` to reduce prefill latency;
- caps default output with `IRONMIND_CPU_MAX_TOKENS`;
- sets CPU thread and batch options through `IRONMIND_CPU_THREADS` and `IRONMIND_CPU_BATCH`;
- injects a runtime system message so the local chatbot does not claim GPU acceleration.

The default profile is `low-latency`: `ctx=4096`, `max_tokens=256`, `batch=128`, and CPU threads selected from the local machine. Use `balanced` or `full-context` only when longer context matters more than response latency.

### Native ik_llama.cpp CPU Runtime

For CPU-first local inference, Iurexa can run through `ik_llama.cpp` instead of
Ollama. IronMind stays the API/UI/orchestration layer, while `ik_llama.cpp`
provides the CPU-optimised GGUF runtime. This is the intended path for making
Iurexa native local inference on CPU without GPU dependency.

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

Run Iurexa with the local `ik_worker` path:

```powershell
$env:IRONMIND_BACKEND="ik_worker"
$env:IRONMIND_IK_LLAMA_SERVER="C:\ai\ik_llama.cpp\build\bin\Release\llama-server.exe"
$env:IRONMIND_IK_LLAMA_WORKER="C:\ai\ik_llama.cpp\build\bin\Release\llama-cli.exe"
$env:IRONMIND_IK_LLAMA_MODEL="C:\models\qwen3.gguf"
$env:IRONMIND_CPU_ONLY="true"
ironmind
```

`IRONMIND_BACKEND=ik_worker` runs `llama-cli.exe` as a local native worker
process with CPU-only flags, prompt-cache files next to IronMind context
snapshots, and no `llama-server` HTTP hop. If you prefer a warm HTTP runtime for
lower latency, set `IRONMIND_BACKEND=ik_llama` to let Iurexa manage
`llama-server`, or set `IRONMIND_BACKEND=llama` with
`IRONMIND_LLAMA_URL=http://127.0.0.1:8080`.

For the direct embedded wrapper against the pinned submodule, build:

```powershell
npm run native:ik:build
$env:IRONMIND_BACKEND="ik_embedded"
$env:IRONMIND_IK_EMBEDDED_RUNNER=".\build-ik\Release\ironmind-ik-native.exe"
$env:IRONMIND_IK_EMBEDDED_DAEMON=".\build-ik\Release\ironmind-ik-daemon.exe"
$env:IRONMIND_IK_LLAMA_MODEL="C:\models\qwen3.gguf"
$env:IRONMIND_CPU_ONLY="true"
ironmind
```

`ik_embedded` links to `ik_llama.cpp` through `llama.h` and renders prompts from
IronMind directly, without `llama-server` or `llama-cli`. When
`ironmind-ik-daemon` is available, IronMind keeps one hidden native runtime
alive, the model remains loaded in RAM, and repeated requests reuse the common
prompt-prefix KV cache. If the daemon is missing, IronMind falls back to the
one-shot `ironmind-ik-native` runner.

When `IRONMIND_BACKEND=ik_worker`, `IRONMIND_BACKEND=ik_embedded`, or
`IRONMIND_BACKEND=ik_llama`, the public
agent is exposed as `iurexa` / **Iurexa**. `ik_llama.cpp` remains the CPU runtime
under the hood, while the GGUF file path stays a runtime detail. Iurexa speaks
Italian by default, assumes Italy as the initial jurisdiction when none is
provided, and strips any residual `<think>` block from the visible assistant
message unless you explicitly enable reasoning mode.

The next native milestone is replacing the persistent process boundary with a
true C ABI/Node binding, then adding direct token streaming and multi-session KV
restore/save. The current daemon already removes per-request model loading for
the active runtime.

The integration plan is tracked in `docs/IK_LLAMA_NATIVE_RUNTIME.md`.
The reproducible Iurexa quantization path, including Italian legal calibration,
`llama-imatrix`, IK-family quantization, benchmarks, and model selection, is in
`docs/IUREXA_QUANTIZATION.md`.
The local PDF/DOCX/TXT upload, extraction, chunking, citation, comparison, and
report workflow is documented in `docs/IUREXA_DOCUMENT_RAG.md`.
For Magistra Desktop packaging, `iurexa-runtime.exe` is the headless local
OpenAI-compatible daemon: it keeps the GGUF model loaded, binds to `127.0.0.1`,
and exposes `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. See `docs/IUREXA_RUNTIME.md`.

## Design

Iurexa is organized around a vertical local stack:

- CPU/RAM inference target: one supported model family first, not a generic model zoo.
- Prompt renderer: model-specific chat and tool formatting.
- Connected chatbot: local browser UI streamed from the IronMind server.
- OpenAI-compatible API: `/v1/models` and `/v1/chat/completions`.
- Agent path: future `/v1/responses` and Anthropic-compatible `/v1/messages`.
- KV strategy: RAM session first, disk persistence next.
- Bench/eval discipline: token throughput, prompt rendering checks, and regression traces.

## Run

```powershell
ironmind --model iurexa --ctx 4096 --kv-disk-dir C:\IronMindKV --kv-disk-space-mb 16384
```

Environment variables:

```text
IRONMIND_MODEL=iurexa
IRONMIND_CTX=131072
IRONMIND_KV_DISK_DIR=C:\IronMindKV
IRONMIND_KV_DISK_SPACE_MB=16384
IRONMIND_PORT=4141
IRONMIND_OLLAMA_URL=http://127.0.0.1:11434
IRONMIND_LLAMA_URL=http://127.0.0.1:8080
IRONMIND_BACKEND=ik_llama
IRONMIND_IK_LLAMA_SERVER=C:\ai\ik_llama.cpp\build\bin\Release\llama-server.exe
IRONMIND_IK_LLAMA_WORKER=C:\ai\ik_llama.cpp\build\bin\Release\llama-cli.exe
IRONMIND_IK_EMBEDDED_RUNNER=C:\path\to\ironmind-ik-native.exe
IRONMIND_IK_EMBEDDED_DAEMON=C:\path\to\ironmind-ik-daemon.exe
IRONMIND_IK_EMBEDDED_PERSISTENT=true
IRONMIND_IK_LLAMA_MODEL=C:\models\qwen3.gguf
IRONMIND_IK_LLAMA_HOST=127.0.0.1
IRONMIND_IK_LLAMA_PORT=8080
IRONMIND_NATIVE_MODEL=C:\path\to\model.gguf
IRONMIND_CPU_ONLY=true
IRONMIND_CPU_PROFILE=low-latency
IRONMIND_CPU_THREADS=10
IRONMIND_CPU_BATCH=128
IRONMIND_CPU_CTX=4096
IRONMIND_CPU_MAX_TOKENS=256
IRONMIND_CPU_KEEP_ALIVE=30m
IRONMIND_DOCUMENT_STORE_DIR=C:\IronMindDocuments
IRONMIND_DOCUMENT_PYTHON=C:\path\to\python.exe
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
npm run eval -- run --model iurexa --limit 10
```

OpenAI-compatible example:

```powershell
curl http://127.0.0.1:4141/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{ "model": "iurexa", "messages": [{"role": "user", "content": "Analizza questa clausola: il foro competente e Roma."}], "stream": false }'
```

Document RAG example:

```powershell
curl http://127.0.0.1:4141/api/documents/query `
  -H "Content-Type: application/json" `
  -d '{ "mode": "compare", "question": "Confronta i documenti su recesso e modifica prezzi.", "max_tokens": 512 }'
```

## IronMind Engine Roadmap

Iurexa intentionally keeps the API, UI, session model, and model target stable while the underlying IronMind native CPU core is built.

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
