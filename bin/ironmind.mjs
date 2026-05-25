#!/usr/bin/env node
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");
const publicDir = path.join(rootDir, "public");
const configPath = path.join(os.homedir(), ".ironmind", "ironmind.json");

const defaults = {
  host: "127.0.0.1",
  port: 4141,
  model: "qwen3-coder:30b",
  ctx: 32768,
  ollamaUrl: "http://127.0.0.1:11434"
};

function readUserConfig() {
  try {
    const raw = fs.readFileSync(configPath, "utf8");
    const parsed = JSON.parse(raw);
    return {
      model: parsed.model,
      ctx: parsed.context,
      ollamaUrl: parsed.ollamaUrl
    };
  } catch {
    return {};
  }
}

function parseArgs(argv) {
  const args = [...argv];
  const command = args[0] && !args[0].startsWith("-") ? args.shift() : "serve";
  const config = { ...defaults, ...readUserConfig() };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    if (arg === "--host" && next) config.host = next, i += 1;
    else if (arg === "--port" && next) config.port = Number(next), i += 1;
    else if (arg === "--model" && next) config.model = next, i += 1;
    else if (arg === "--ctx" && next) config.ctx = Number(next), i += 1;
    else if (arg === "--ollama" && next) config.ollamaUrl = next, i += 1;
    else if (arg === "--help" || arg === "-h") return { command: "help", config };
  }

  config.host = process.env.IRONMIND_HOST || config.host;
  config.port = Number(process.env.IRONMIND_PORT || config.port);
  config.model = process.env.IRONMIND_MODEL || config.model;
  config.ctx = Number(process.env.IRONMIND_CTX || config.ctx);
  config.ollamaUrl = process.env.IRONMIND_OLLAMA_URL || config.ollamaUrl;
  config.ollamaUrl = config.ollamaUrl.replace(/\/+$/, "");
  return { command, config };
}

function usage() {
  console.log(`IronMind

Usage:
  ironmind [serve] [--host 127.0.0.1] [--port 4141] [--model qwen3-coder:30b] [--ctx 32768]
  ironmind doctor

Environment:
  IRONMIND_MODEL
  IRONMIND_CTX
  IRONMIND_PORT
  IRONMIND_OLLAMA_URL
`);
}

function json(res, status, body) {
  const payload = JSON.stringify(body, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload)
  });
  res.end(payload);
}

function text(res, status, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(status, {
    "content-type": contentType,
    "content-length": Buffer.byteLength(body)
  });
  res.end(body);
}

function notFound(res) {
  json(res, 404, { error: "not_found" });
}

async function readJson(req) {
  let body = "";
  for await (const chunk of req) body += chunk;
  if (!body.trim()) return {};
  return JSON.parse(body);
}

function safeStaticPath(urlPath) {
  const pathname = decodeURIComponent(new URL(urlPath, "http://ironmind.local").pathname);
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const resolved = path.resolve(publicDir, relative);
  if (resolved !== publicDir && !resolved.startsWith(publicDir + path.sep)) return null;
  return resolved;
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".css")) return "text/css; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".svg")) return "image/svg+xml";
  return "application/octet-stream";
}

async function serveStatic(req, res) {
  const file = safeStaticPath(req.url);
  if (!file) return notFound(res);
  try {
    const data = await fs.promises.readFile(file);
    res.writeHead(200, { "content-type": contentType(file) });
    res.end(data);
  } catch {
    notFound(res);
  }
}

async function ollamaChat(config, payload, stream) {
  const options = {
    ...(payload.options || {}),
    num_ctx: Number(payload.ctx || config.ctx)
  };
  if (payload.temperature !== undefined) options.temperature = payload.temperature;
  if (payload.top_p !== undefined) options.top_p = payload.top_p;
  if (payload.max_tokens !== undefined) options.num_predict = Number(payload.max_tokens);
  if (payload.max_completion_tokens !== undefined) options.num_predict = Number(payload.max_completion_tokens);
  const think = payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort);

  const response = await fetch(`${config.ollamaUrl}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: payload.model || config.model,
      messages: payload.messages || [],
      tools: payload.tools,
      stream,
      think,
      options
    })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Ollama ${response.status}: ${detail}`);
  }

  return response;
}

async function eachNdjson(response, onObject) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) await onObject(JSON.parse(trimmed));
    }
  }

  const tail = buffer.trim();
  if (tail) await onObject(JSON.parse(tail));
}

function openAiChunk(id, model, content, finishReason = null) {
  return {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta: content ? { content } : {},
        finish_reason: finishReason
      }
    ]
  };
}

function chunkText(chunk) {
  return chunk.message?.content || chunk.message?.reasoning_content || chunk.message?.thinking || "";
}

async function handleUiChat(req, res, config) {
  try {
    const body = await readJson(req);
    const response = await ollamaChat(config, body, true);
    res.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-cache"
    });

    await eachNdjson(response, async (chunk) => {
      const content = chunkText(chunk);
      if (content) res.write(JSON.stringify({ type: "delta", content }) + "\n");
      if (chunk.done) {
        res.write(JSON.stringify({
          type: "done",
          model: chunk.model,
          promptEvalCount: chunk.prompt_eval_count,
          evalCount: chunk.eval_count
        }) + "\n");
      }
    });
    res.end();
  } catch (error) {
    json(res, 502, { error: "backend_unavailable", detail: error.message });
  }
}

async function handleOpenAiChat(req, res, config) {
  try {
    const body = await readJson(req);
    const model = body.model || config.model;

    if (body.stream === false) {
      const response = await ollamaChat(config, body, false);
      const out = await response.json();
      return json(res, 200, {
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: chunkText(out)
            },
            finish_reason: "stop"
          }
        ],
        usage: {
          prompt_tokens: out.prompt_eval_count || 0,
          completion_tokens: out.eval_count || 0,
          total_tokens: (out.prompt_eval_count || 0) + (out.eval_count || 0)
        }
      });
    }

    const response = await ollamaChat(config, body, true);
    const id = `chatcmpl-${Date.now()}`;
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive"
    });

    await eachNdjson(response, async (chunk) => {
      const content = chunkText(chunk);
      if (content) res.write(`data: ${JSON.stringify(openAiChunk(id, model, content))}\n\n`);
      if (chunk.done) res.write(`data: ${JSON.stringify(openAiChunk(id, model, "", "stop"))}\n\n`);
    });
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (error) {
    json(res, 502, { error: { message: error.message, type: "backend_unavailable" } });
  }
}

function handleModels(res, config) {
  json(res, 200, {
    object: "list",
    data: [
      {
        id: config.model,
        object: "model",
        created: 0,
        owned_by: "ironmind"
      }
    ]
  });
}

function createServer(config) {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || "127.0.0.1"}`);

    if (req.method === "GET" && url.pathname === "/health") {
      return json(res, 200, {
        ok: true,
        model: config.model,
        context: config.ctx,
        backend: config.ollamaUrl
      });
    }

    if (req.method === "GET" && url.pathname === "/v1/models") {
      return handleModels(res, config);
    }

    if (req.method === "POST" && url.pathname === "/api/chat") {
      return handleUiChat(req, res, config);
    }

    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      return handleOpenAiChat(req, res, config);
    }

    if (req.method === "GET") return serveStatic(req, res);
    return text(res, 405, "method not allowed");
  });
}

async function doctor(config) {
  console.log("IronMind doctor");
  console.log(`  model:   ${config.model}`);
  console.log(`  context: ${config.ctx}`);
  console.log(`  ollama:  ${config.ollamaUrl}`);

  try {
    const response = await fetch(`${config.ollamaUrl}/api/tags`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const names = (data.models || []).map((model) => model.name);
    console.log("  backend: reachable");
    console.log(`  models:  ${names.length ? names.join(", ") : "none pulled yet"}`);
  } catch (error) {
    console.log(`  backend: not reachable (${error.message})`);
    process.exitCode = 1;
  }
}

async function main() {
  const { command, config } = parseArgs(process.argv.slice(2));

  if (command === "help") return usage();
  if (command === "doctor") return doctor(config);
  if (command !== "serve") return usage();

  const server = createServer(config);
  server.listen(config.port, config.host, () => {
    console.log(`IronMind listening on http://${config.host}:${config.port}`);
    console.log(`Model: ${config.model}`);
    console.log(`Backend: ${config.ollamaUrl}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
