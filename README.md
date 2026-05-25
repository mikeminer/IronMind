# IronMind

IronMind is a native-first CPU/RAM local inference project for high-end laptops and workstations.
It is inspired by DwarfStar's vertical design: one serious local model target at a time, RAM/disk KV-cache work, OpenAI-compatible APIs, and a connected local chatbot.

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
ironmind tokenize C:\path\to\model.gguf "Ciao mondo"
```

Build the first native GGUF inspector:

```powershell
npm run native:build
.\build\Release\ironmind-inspect.exe C:\path\to\model.gguf
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
8. Quantized CPU matmul kernels for AVX2/AVX512.
9. Native IronKV payload format integration for full model sessions.
10. Native tool-call replay and canonicalization.
11. Evaluation suite for physics, mathematics, and defensive security. Implemented as IronMind Eval 100.
12. Replace the bootstrap runtime path with the IronMind CPU backend.

## License

MIT
