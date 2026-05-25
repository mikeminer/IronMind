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

## Implemented Foundations

- GGUF metadata and tensor-directory inspector: `lib/gguf.mjs`.
- Qwen3/Qwen3MoE target validator: `lib/target.mjs`.
- Qwen3 chat/tool prompt renderer: `lib/qwen3Prompt.mjs`.
- IronKV disk-cache container: `lib/ironkv.mjs`.
- Persistent disk context snapshots: `lib/contextStore.mjs`.

These are not the final inference core yet. They are the first model-specific contracts the native engine must obey.

## Native CPU Target

The first native target should be a single GGUF quantization of a practical model for 64GB RAM machines.
The current candidate is Qwen3-Coder 30B A3B.

The context target is at least 131072 tokens. On 64GB RAM machines, IronMind treats disk as a first-class context tier:

- RAM holds the active decode window and hottest KV pages.
- Disk holds reusable prompt prefixes and, in the native backend, serialized KV payloads.
- The recommended storage is internal SSD/NVMe. Removable SD storage is acceptable only for cold archives, not active KV restore/write.

The native core should prioritize:

- predictable memory use;
- long-context prefix reuse;
- simple disk cache files;
- AVX2 baseline kernels;
- AVX512/VNNI fast path when available;
- deterministic prompt rendering tests.

## Milestone Order

1. Validate the selected GGUF and refuse unknown architectures.
2. Render Qwen3 prompts byte-for-byte from IronMind, independent of Ollama.
3. Read tokenizer metadata and implement tokenization tests.
4. Map model tensors into typed weight views.
5. Implement RMSNorm, RoPE, attention, and dense FFN path.
6. Add MoE router and expert dispatch for Qwen3MoE.
7. Add AVX2 baseline quantized matmul.
8. Add AVX512/VNNI kernels where available.
9. Save RAM KV state into IronKV and restore it across process restarts for 100k+ token sessions.
10. Add logit/token-vector regression tests.

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
