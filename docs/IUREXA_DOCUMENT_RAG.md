# Iurexa Document RAG

Iurexa includes a local document workflow for legal text analysis:

1. upload PDF, DOCX, TXT, or Markdown from the browser;
2. extract text locally through the configured Python runtime;
3. split extracted text into citation-ready chunks;
4. rank chunks against the user question;
5. ask the local model to answer, summarize, compare, or produce a report using
   only the retrieved sources;
6. return source IDs and excerpts to the UI/API.

No document content is sent to an external service by this workflow.

## API

List indexed documents:

```powershell
curl http://127.0.0.1:4141/api/documents
```

Upload documents:

```powershell
curl -X POST `
  -F "files=@C:\path\contract.pdf;type=application/pdf" `
  http://127.0.0.1:4141/api/documents/upload
```

Ask over documents:

```powershell
curl http://127.0.0.1:4141/api/documents/query `
  -H "Content-Type: application/json" `
  -d '{ "mode": "compare", "question": "Confronta i documenti su recesso e modifica prezzi.", "max_tokens": 512 }'
```

Supported modes:

- `answer`: focused answer with citations;
- `summary`: professional summary;
- `compare`: multi-document comparison;
- `report`: structured report with executive summary, risks, evidence, and next steps.

## Storage

The default document store is:

```text
%USERPROFILE%\.ironmind\documents
```

It contains:

- `files/`: original uploaded files;
- `index/`: extracted sections, chunks, metadata, and retrieval terms.

Configure it with:

```text
IRONMIND_DOCUMENT_STORE_DIR=C:\IronMindDocuments
IRONMIND_DOCUMENT_PYTHON=C:\path\to\python.exe
```

## Extraction

`tools/extract-document.py` uses:

- `pdfplumber` first for PDFs;
- `pypdf` fallback for PDFs;
- `python-docx` for DOCX;
- direct UTF-8/CP1252 text loading for TXT/Markdown.

If extraction fails, the upload returns an error and the original file is not
added to the searchable index.

## Current Limits

This is a local MVP, not a full e-discovery engine:

- scanned image PDFs need OCR before upload;
- citations are chunk/page/paragraph references, not court-grade pinpoint cites;
- retrieval is lexical and deterministic, not embedding-based yet;
- the default `ik_worker` mode removes HTTP but reloads the model per request,
  so `ik_llama` server mode is still faster for long interactive sessions;
- `ik_embedded` removes the `llama-cli` dependency and links to `ik_llama.cpp`,
  but it is still process-per-request until the persistent ABI lands;
- the next step is a persistent embedded runtime ABI plus vector/semantic
  retrieval for larger document sets.
