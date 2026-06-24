# IronMind ik_llama.cpp Native Runtime

## Goal

IronMind should use CPU-optimised local inference as its primary runtime path.
The practical path is:

1. IronMind owns API, UI, session persistence, clinical screening, and orchestration.
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
`/v1/chat/completions`, and clinical screening APIs stay stable.
