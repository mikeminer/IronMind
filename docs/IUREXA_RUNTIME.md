# Iurexa Runtime

`iurexa-runtime.exe` is the headless Windows runtime that Magistra Desktop can
start in the background as a local OpenAI-compatible LLM provider.

It does not retrieve sources, import legal documents, search online, or decide
which citations to use. Magistra owns RAG, PGlite, retrieval, source selection,
and citation display. Iurexa Runtime only receives a grounded prompt and runs
CPU-only GGUF inference through the embedded `ik_llama.cpp` backend.

## Start

```powershell
C:\Users\mikfo\Documents\IRONMIND\build-ik\Release\iurexa-runtime.exe `
  --model "C:\Path\To\models\iurexa-lite-IQ4_XS.gguf" `
  --host 127.0.0.1 `
  --port 4141 `
  --ctx 4096 `
  --threads auto `
  --batch 128 `
  --cpu-only
```

The model remains a separate GGUF file. It is not compiled into the executable.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /shutdown` for controlled local shutdown by Magistra Desktop

`/v1/chat/completions` accepts OpenAI-style `system`, `user`, and `assistant`
messages and supports both `stream: false` and Server-Sent Events with
`stream: true`.

## Locality

The default bind address is `127.0.0.1`. Public bind addresses are rejected
unless the process is started with `--allow-public-bind`, which should not be
used for Magistra Desktop packaging.

## Logs

Default log file:

```text
%APPDATA%\Iurexa\logs\iurexa-runtime.log
```

The log includes startup, model-load status, llama backend messages, generation
errors, and controlled shutdown requests.

## Errors

If the GGUF file is missing, invalid, or still loading, `/health` remains
available with `modelLoaded: false`, and chat requests return a stable 503 JSON:

```json
{
  "error": {
    "message": "Model file not found: ...",
    "type": "iurexa_runtime_error",
    "code": "model_not_loaded"
  }
}
```

This lets Magistra Desktop show a controlled diagnostic instead of treating the
runtime as a crashed UI component.
