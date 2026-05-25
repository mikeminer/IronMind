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
ironmind --model qwen3-coder:30b --ctx 32768
```

Environment variables:

```text
IRONMIND_MODEL=qwen3-coder:30b
IRONMIND_CTX=32768
IRONMIND_PORT=4141
IRONMIND_OLLAMA_URL=http://127.0.0.1:11434
```

Health check:

```powershell
ironmind doctor
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

1. GGUF metadata reader for the selected target model.
2. Tokenizer and prompt rendering tests.
3. Quantized CPU matmul kernels for AVX2/AVX512.
4. Attention and KV-cache snapshots in RAM.
5. Disk KV-cache format with prefix reuse.
6. Native tool-call replay and canonicalization.
7. Replace the bootstrap runtime path with the IronMind CPU backend.

## License

MIT
