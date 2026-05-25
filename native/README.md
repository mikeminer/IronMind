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
3. `ironmind_gguf.c`: load GGUF metadata, tensor directory, offsets, sizes, and tensor payloads.
4. `ironmind_quant.c`: scalar GGUF row dequantization and matvec for F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q6_K.
5. `kernels_avx512.c`: optional high-end laptop fast path.
6. `ironmind_math.c`: scalar RMSNorm, RoPE, softmax, attention kernels. AVX paths come next.
7. `ironmind_forward.c`: native dense decode step with RAM KV cache and save/restore.
8. `ironmind_moe.c`: router top-k and expert mixing for Qwen3MoE.
9. `ironmind_qwen3.c`: GGUF-backed dense/MoE Qwen3 decode path.
10. `kv_cache.c`: RAM KV state and IronKV save/restore.
11. `eval_vectors.c`: logit/token regression runner.
12. `ironmind_native.c`: native GGUF readiness/decode gate for a real model file.

`ironmind_forward.c` currently runs an F32 Qwen-like dense decode path and proves the native KV lifecycle:
token -> QKV -> q/k norm -> RoPE -> causal attention over RAM KV -> FFN -> logits, then save/load KV and continue.
`ironmind_qwen3.c` uses `ironmind_gguf.c` and `ironmind_quant.c` to run the same decode against GGUF tensor views without dequantizing the whole model into RAM.

Run:

```powershell
npm run native:test
.\build\Release\ironmind-native.exe C:\path\to\model.gguf
.\build\Release\ironmind-native.exe C:\path\to\model.gguf --decode 0 --ctx 1
```

`ironmind_qwen3_test.c` writes a tiny Qwen3 GGUF fixture, decodes it through the GGUF-backed path, and compares logits plus argmax token against the F32 reference forward path.

The next step is performance: AVX2/AVX512 matvec kernels, tensor residency strategy, and then server-side native session integration.

The rule is simple: if a model does not match the selected target contract, the native backend should refuse it early.
