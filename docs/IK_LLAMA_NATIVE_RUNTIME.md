# IronMind ik_llama.cpp Native Runtime

## Goal

IronMind should use CPU-optimised local inference as its primary runtime path.
The practical path is:

1. IronMind owns API, UI, session persistence, and chat orchestration.
2. `ik_llama.cpp` owns GGUF execution, CPU kernels, quantized matmul, KV cache, and token generation.
3. The first integration runs `ik_llama.cpp` through its `llama-server` process.
4. A later integration can replace the process boundary with direct native linking or a small C ABI.

This avoids depending on Ollama for production CPU inference while keeping IronMind
small enough to evolve.

## Current Integration

Set:

```powershell
$env:IRONMIND_BACKEND="ik_llama"
$env:IRONMIND_IK_LLAMA_SERVER="C:\ai\ik_llama.cpp\build\bin\Release\llama-server.exe"
$env:IRONMIND_IK_LLAMA_MODEL="C:\models\qwen3.gguf"
$env:IRONMIND_CPU_ONLY="true"
ironmind
```

IronMind will start `llama-server` with:

- `--model <IRONMIND_IK_LLAMA_MODEL>`
- `--host <IRONMIND_IK_LLAMA_HOST>`
- `--port <IRONMIND_IK_LLAMA_PORT>`
- `--ctx-size <CPU profile context>`
- `--threads <CPU profile threads>`
- `--batch-size <CPU profile batch>`
- `--n-gpu-layers 0`

Then IronMind routes chat through the OpenAI-compatible
`/v1/chat/completions` endpoint exposed by `llama-server`.

The public product name for this managed CPU path is **Iurexa**, with API
model id `iurexa`. `ik_llama.cpp` remains the runtime, and the GGUF path
is only a runtime configuration detail. Iurexa speaks Italian by default and is
tuned as a legal-support assistant. For GGUF chat templates that emit hidden thinking text, Iurexa removes any residual
`<think>...</think>` block before returning data to `/api/chat`,
`/v1/chat/completions`, `/v1/responses`, or `/v1/messages`. Requests that set
`think: true`, `reasoning`, or `reasoning_effort` keep reasoning mode available.

Validated locally on Windows with `ik_llama.cpp` commit `d5507e33`,
`llama-server.exe`, `Qwen3 14B` GGUF `Q4_K - Medium`, `--n-gpu-layers 0`,
`ctx=4096`, `batch=128`, and six CPU threads. Smoke tests covered:

- `GET /health` reporting `backendMode: "ik_llama"` and `cpuOnly: true`;
- `POST /v1/chat/completions` returning an `iurexa` Italian answer without `<think>` tags;
- `POST /api/chat` returning NDJSON consumed by the browser chatbot;

## Why Process First

The process boundary is the safest first native step:

- keeps `ik_llama.cpp` updatable without vendoring a large fork;
- avoids ABI churn while evaluating CPU throughput;
- lets IronMind keep its existing OpenAI-compatible API;
- makes failures isolated: the runtime process can restart independently;
- gives a clear benchmark target before deeper C/C++ linking.

## Next Step: Direct Native Binding

Once the managed server path is stable, the next milestone is a direct runtime
adapter:

- add `third_party/ik_llama.cpp` as a pinned submodule or source dependency;
- expose a minimal C ABI for model load, tokenize, decode, KV save/restore, and free;
- create a Node native addon or a small local worker process with a binary protocol;
- map IronMind session snapshots to the runtime KV cache;
- keep `IRONMIND_BACKEND=ik_llama` as the same public backend while changing the
  internal transport from HTTP to direct native calls.

The public product should not change when this happens: the UI, `/api/chat`,
`/v1/chat/completions`, `/v1/responses`, and `/v1/messages` stay stable.
