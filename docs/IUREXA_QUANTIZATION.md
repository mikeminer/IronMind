# Iurexa CPU Quantization Pipeline

This document records the reproducible CPU-only path used to build the local
Iurexa runtime with `ik_llama.cpp`, an Italian legal calibration corpus, an
importance matrix, and IK-family GGUF quantizations.

## Runtime Tools

Build `ik_llama.cpp` with the server and quantization tools:

```powershell
cd C:\Users\mikfo\Documents\IRONMIND-runtimes\ik_llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_IQK_FA_ALL_QUANTS=ON
cmake --build build --config Release --target llama-server llama-quantize llama-imatrix llama-cli llama-bench
```

Expected tools:

```text
build\bin\Release\llama-server.exe
build\bin\Release\llama-quantize.exe
build\bin\Release\llama-imatrix.exe
build\bin\Release\llama-cli.exe
build\bin\Release\llama-bench.exe
```

## Calibration Corpus

Use `calibration/iurexa-legal-it.txt` as the Italian legal calibration text.
It is written for a source-grounded legal assistant that reads clauses, legal
sources, policies, letters, and client material, then produces concise and
professional Italian summaries without inventing missing law.

Generate an importance matrix:

```powershell
$tools = "C:\Users\mikfo\Documents\IRONMIND-runtimes\ik_llama.cpp\build\bin\Release"
$models = "C:\Users\mikfo\Documents\IRONMIND-models\iurexa"

& "$tools\llama-imatrix.exe" `
  -m "$models\Qwen3-1.7B-F16.gguf" `
  -f "C:\Users\mikfo\Documents\IRONMIND\calibration\iurexa-legal-it.txt" `
  -o "$models\iurexa-qwen3-1.7b-instruct-legal-it.imatrix" `
  -t 6 -c 4096 -b 128 -ngl 0
```

The calibration file should contain more than one context window of text so
`llama-imatrix` can collect useful activation statistics.

## Quantization

Start with quality-preserving IK quantizations:

```powershell
& "$tools\llama-quantize.exe" `
  --imatrix "$models\iurexa-qwen3-1.7b-instruct-legal-it.imatrix" `
  "$models\Qwen3-1.7B-F16.gguf" `
  "$models\iurexa-qwen3-1.7b-instruct-IQ4_XS.gguf" `
  IQ4_XS 6

& "$tools\llama-quantize.exe" `
  --imatrix "$models\iurexa-qwen3-1.7b-instruct-legal-it.imatrix" `
  "$models\Qwen3-1.7B-F16.gguf" `
  "$models\iurexa-qwen3-1.7b-instruct-IQ3_KS.gguf" `
  IQ3_KS 6
```

Try more aggressive quantization only after a quality gate passes:

```powershell
& "$tools\llama-quantize.exe" `
  --imatrix "$models\iurexa-qwen3-1.7b-instruct-legal-it.imatrix" `
  "$models\Qwen3-1.7B-F16.gguf" `
  "$models\iurexa-qwen3-1.7b-instruct-IQ2_KS.gguf" `
  IQ2_KS 6
```

## Benchmarks

Local Windows CPU-only measurements with six threads, `ctx=4096`,
`batch=128`, and `--n-gpu-layers 0`:

| Candidate | File size | Prompt processing | Generation | RAM while served | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.6B IQ4_XS | 434 MiB | 126.25 t/s | 29.38 t/s | not selected | Too weak for legal Italian; repeated clauses. |
| 1.7B IQ4_XS | 1.10 GiB | 78.49 t/s | 19.17 t/s | about 1.5 GiB | Default Iurexa Lite profile. |
| 1.7B IQ3_KS | 867 MiB | 97.85 t/s | 14.37 t/s | about 1.3 GiB | Compact experimental profile; lower legal quality. |

The practical conclusion is important: "a few megabytes" is not realistic for a
useful legal assistant. The current small-but-usable floor is roughly 0.9-1.1
GiB for the quantized GGUF, with the 1.7B IQ4_XS model giving the better quality
tradeoff for Italian legal support.

End-to-end API smoke test for the default Iurexa Lite profile, using the local
IronMind wrapper, runtime legal prompt, `stream=false`, and the unilateral price
change clause: 189 completion tokens in 22.38 seconds, about 8.44 visible tokens
per second, with `llama-server.exe` at about 1.5 GiB working set. Browser smoke
test confirmed the local UI returned the legal sections without visible
`<think>` text and without console errors.

Run the benchmark again with:

```powershell
& "$tools\llama-bench.exe" `
  -m "$models\iurexa-qwen3-1.7b-instruct-IQ4_XS.gguf" `
  -p 512 -n 128 -t 6 -b 128 -ub 128 -ngl 0 -r 3 -o md
```

## Runtime Configuration

Use IQ4_XS as the default product model:

```json
{
  "model": "iurexa",
  "backend": "ik_embedded",
  "ikLlamaServer": "C:\\Users\\mikfo\\Documents\\IRONMIND-runtimes\\ik_llama.cpp\\build\\bin\\Release\\llama-server.exe",
  "ikEmbeddedRunner": "C:\\Users\\mikfo\\Documents\\IRONMIND\\build-ik\\Release\\ironmind-ik-native.exe",
  "ikLlamaModel": "C:\\Users\\mikfo\\Documents\\IRONMIND-models\\iurexa\\iurexa-qwen3-1.7b-instruct-IQ4_XS.gguf",
  "cpuOnly": true,
  "cpuProfile": "low-latency",
  "cpuThreads": 6,
  "cpuBatch": 128,
  "cpuInteractiveCtx": 4096,
  "cpuMaxTokens": 256
}
```

`ik_llama.cpp` is the CPU runtime. Iurexa is the product identity exposed by the
local app and API. Use `backend: "ik_llama"` when you want a warm
`llama-server` for lower latency, `backend: "ik_worker"` when you want no HTTP
hop through `llama-cli`, or `backend: "ik_embedded"` when testing the direct
linked wrapper built by `npm run native:ik:build`.
