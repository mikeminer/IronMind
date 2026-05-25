# IronMind Architecture

IronMind is meant to become a narrow CPU/RAM inference engine, not a model zoo.
The bootstrap keeps the product surface usable while the native backend is developed.

## Layers

1. Chatbot UI
2. IronMind HTTP server
3. OpenAI-compatible API adapter
4. Model-specific prompt/tool renderer
5. Session and KV-cache manager
6. Native CPU backend

During pre-alpha, layer 6 is represented by an Ollama/llama.cpp-compatible local backend.

## Native CPU Target

The first native target should be a single GGUF quantization of a practical model for 64GB RAM machines.
The current candidate is Qwen3-Coder 30B A3B.

The native core should prioritize:

- predictable memory use;
- long-context prefix reuse;
- simple disk cache files;
- AVX2 baseline kernels;
- AVX512/VNNI fast path when available;
- deterministic prompt rendering tests.

## API Surface

Initial:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /api/chat`

Planned:

- `POST /v1/responses`
- `POST /v1/messages`
- disk-backed session switching;
- exact tool replay.
