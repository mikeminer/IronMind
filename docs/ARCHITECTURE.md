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
- Qwen3 GGUF tokenizer loader and BPE tokenizer: `lib/tokenizer.mjs`.
- Dense Qwen3/Qwen3MoE tensor mapping: `lib/tensorMap.mjs`.
- Native GGUF tensor loader and tensor data reader: `native/ironmind_gguf.c`.
- Bounded raw-GGUF tensor residency cache: `native/ironmind_gguf.c`.
- Native quantized row/matvec scalar kernels: `native/ironmind_quant.c`.
- Runtime SIMD dot dispatch: `native/ironmind_simd.c`, `native/ironmind_simd_avx2.c`, and `native/ironmind_simd_avx512.c`.
- Native MoE router and expert mixer: `native/ironmind_moe.c`.
- GGUF-backed Qwen3 decode wiring: `native/ironmind_qwen3.c`.
- RMSNorm, RoPE, softmax, and attention reference kernels: `lib/mathCore.mjs` and `native/ironmind_math.c`.
- Native F32 dense decode step with RAM KV save/restore: `native/ironmind_forward.c`.
- IronKV disk-cache container: `lib/ironkv.mjs`.
- Persistent disk context snapshots: `lib/contextStore.mjs`.

The native pieces now load real GGUF tensor views, validate supported quantized matvec formats, route MoE experts, decode through GGUF-backed Qwen3 tensors, and compare logits/token argmax against the F32 reference path.

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

## Native Residency Strategy

IronMind does not dequantize the whole model into RAM. GGUF tensors stay in their raw quantized form. The native backend uses:

- pinned residency for small high-frequency tensors such as RMSNorm and q/k norm vectors;
- bounded LRU residency for raw quantized tensors that fit under `IRONMIND_NATIVE_CACHE_MAX_TENSOR_MB`;
- sequential chunked reads for large matrices that should stay file-backed;
- runtime SIMD dot dispatch, currently scalar/AVX2/AVX512F.

The default budget is `IRONMIND_NATIVE_CACHE_MB=512` with `IRONMIND_NATIVE_CACHE_MAX_TENSOR_MB=64`, which keeps the memory footprint predictable on 64GB machines.

## Milestone Order

1. Validate the selected GGUF and refuse unknown architectures.
2. Render Qwen3 prompts byte-for-byte from IronMind, independent of Ollama.
3. Read tokenizer metadata and implement tokenization tests. Done for GGUF `gpt2/qwen2`.
4. Map model tensors into typed weight views. Done for Qwen3 and Qwen3MoE names.
5. Implement RMSNorm, RoPE, attention, and dense FFN path. Native F32 decode step is in place.
6. Add MoE router and expert dispatch for Qwen3MoE. Scalar top-k routing and expert mixing are in place.
7. Add scalar quantized matmul. Implemented for common GGUF CPU formats including Q4_K and Q6_K.
8. Wire GGUF tensor views into the native forward pass and emit real logits. Dense and MoE Qwen3 wiring is in place.
9. Add logit/token-vector regression tests. A tiny GGUF fixture compares GGUF-backed logits/token argmax against the F32 reference path.
10. Add AVX2 and AVX512 kernels where available. Runtime AVX2/AVX512F dot dispatch is in place; direct quantized dot kernels are next.
11. Save RAM KV state into IronKV and restore it across process restarts for 100k+ token sessions.

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
