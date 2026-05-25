# IronMind

IronMind is a small native CPU inference engine optimized for one local agentic model at a time. It targets ordinary laptops and workstations with 64GB+ RAM, using quantized GGUF weights, RAM/disk KV cache, OpenAI-compatible APIs, and an integrated local chatbot/agent.

> Status: pre-alpha. The current bootstrap ships the server, chatbot, installer, and API shape while the native CPU engine is developed. For usable inference today it connects to a local Ollama or llama.cpp-compatible runtime.

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

## Design

IronMind is organized around a vertical local stack:

- CPU/RAM inference target: one supported model family first, not a generic model zoo.
- Prompt renderer: model-specific chat and tool formatting.
- Connected chatbot: local browser UI streamed from the IronMind server.
- OpenAI-compatible API: `/v1/models` and `/v1/chat/completions`.
- Agent path: future `/v1/responses` and Anthropic-compatible `/v1/messages`.
- KV strategy: RAM session first, disk persistence next.
- Bench/eval discipline: token throughput, prompt rendering checks, and regression traces.

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
```

Use an internal SSD/NVMe for `IRONMIND_KV_DISK_DIR`. A removable SD card is usually too slow for 100k+ token KV-cache restore/write patterns.

Health check:

```powershell
ironmind doctor
```

Inspect a GGUF before trying to run it:

```powershell
ironmind inspect C:\path\to\model.gguf
ironmind map C:\path\to\model.gguf
ironmind native C:\path\to\model.gguf
ironmind tokenize C:\path\to\model.gguf "Ciao mondo"
```

Build the native core:

```powershell
npm run native:build
.\build\Release\ironmind-inspect.exe C:\path\to\model.gguf
.\build\Release\ironmind-native.exe C:\path\to\model.gguf
npm run native:test
```

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
9. Quantized CPU matmul scalar baseline. Implemented for F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K in `native/ironmind_quant.c`; AVX2/AVX512 fast paths are next.
10. MoE top-k routing and expert mixing primitive. Implemented in `native/ironmind_moe.c`.
11. Evaluation suite for physics, mathematics, and defensive security. Implemented as IronMind Eval 100.
12. Native IronKV payload integration for full server sessions.
13. Native tool-call replay and canonicalization.
14. Replace the bootstrap runtime path with the IronMind CPU backend once GGUF-backed forward wiring emits verified tokens.

## License

MIT
