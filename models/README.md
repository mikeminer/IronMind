# Iurexa model files

Put local GGUF model files in this directory when packaging or testing Iurexa
Runtime with Magistra Desktop.

Recommended local filename:

```text
models/iurexa-lite-IQ4_XS.gguf
```

GGUF files are intentionally ignored by Git because they are large runtime
artifacts. The runtime executable loads the model from disk with `--model`; the
model is not compiled into `iurexa-runtime.exe`.
