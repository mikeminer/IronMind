# IronMind Native Core

This directory is reserved for the self-contained CPU engine.

The JavaScript bootstrap already defines the contracts the native code must match:

- GGUF metadata and tensor directory parsing;
- Qwen3/Qwen3MoE target validation;
- Qwen3 prompt rendering;
- IronKV disk-cache container.

The native implementation should stay model-specific. The first target is Qwen3-Coder 30B A3B class GGUF, with dense Qwen3 accepted only as a development smoke-test path.

## Core Work Items

1. `ironmind_inspect.c`: parse GGUF metadata and tensor directory without loading weights.
2. `tokenizer_qwen3.c`: load tokenizer metadata and reproduce Qwen3 token IDs.
3. `weights_qwen3.c`: map required tensors into typed model views.
4. `kernels_avx2.c`: baseline quantized dot/matmul kernels.
5. `kernels_avx512.c`: optional high-end laptop fast path.
6. `forward_qwen3.c`: RMSNorm, QKV, RoPE, attention, FFN.
7. `moe_qwen3.c`: router top-k and expert dispatch for Qwen3MoE.
8. `kv_cache.c`: RAM KV state and IronKV save/restore.
9. `eval_vectors.c`: logit/token regression runner.

The rule is simple: if a model does not match the selected target contract, the native backend should refuse it early.
