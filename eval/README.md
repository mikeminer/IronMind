# IronMind Eval 100

IronMind Eval 100 is an original regression suite for local CPU/RAM models.
It contains 100 multiple-choice questions:

- 34 physics questions;
- 33 mathematics questions;
- 33 defensive security questions.

The security section is intentionally defensive: secure coding, authentication, authorization, web security, secret handling, and vulnerability recognition.

## Commands

```powershell
npm run eval -- stats
npm run eval -- list --category security
npm run eval -- run --model qwen3:14b --limit 10
```

Full local run:

```powershell
npm run eval -- run --model qwen3-coder:30b --out eval-results.json
```

The runner uses the OpenAI-compatible endpoint:

```text
http://127.0.0.1:4141/v1/chat/completions
```

Override it with:

```powershell
$env:IRONMIND_EVAL_BASE_URL="http://127.0.0.1:4141/v1"
$env:IRONMIND_EVAL_MODEL="qwen3:14b"
```
