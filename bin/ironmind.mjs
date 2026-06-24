#!/usr/bin/env node
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { inspectGguf, summarizeGguf } from "../lib/gguf.mjs";
import { validateIronMindTarget } from "../lib/target.mjs";
import {
  contextStoreStats,
  createContextSnapshot,
  defaultContextDir,
  finalizeNativeContextSnapshot,
  saveContextSnapshot
} from "../lib/contextStore.mjs";
import { loadQwen3Tokenizer } from "../lib/tokenizer.mjs";
import { loadQwenTensorMap, summarizeTensorMap } from "../lib/tensorMap.mjs";
import {
  backendDescription,
  nativeChat,
  nativeModelPath,
  nativeRunnerPath,
  shouldUseNativeBackend
} from "../lib/nativeBackend.mjs";
import { canonicalizeMessages, canonicalizeTools, canonicalizeToolCalls } from "../lib/toolCalls.mjs";
import { createClinicalTriage } from "../lib/clinicalScoring.mjs";
import { assessImageQuality } from "../lib/imageQuality.mjs";
import { createClinicalScreeningCase } from "../lib/clinicalScreening.mjs";
import {
  applyCpuPerformanceOptions,
  defaultCpuThreads,
  ensureCpuSystemMessage,
  resolveCpuPerformanceConfig
} from "../lib/cpuPerformance.mjs";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");
const publicDir = path.join(rootDir, "public");
const configPath = path.join(os.homedir(), ".ironmind", "ironmind.json");
let managedIkLlamaProcess = null;

const defaults = {
  host: "127.0.0.1",
  port: 4141,
  model: "qwen3-coder:30b",
  ctx: 131072,
  kvDiskDir: defaultContextDir(),
  kvDiskSpaceMb: 16384,
  ollamaUrl: "http://127.0.0.1:11434",
  llamaUrl: "http://127.0.0.1:8080",
  llamaApiKey: "",
  ikLlamaServer: "",
  ikLlamaModel: "",
  ikLlamaHost: "127.0.0.1",
  ikLlamaPort: 8080,
  ikLlamaAutoStart: true,
  ikLlamaStartupTimeoutMs: 30000,
  backend: "auto",
  nativeModel: "",
  nativeMaxTokens: 16,
  nativeTimeoutMs: 300000,
  cpuOnly: true,
  cpuProfile: "low-latency",
  cpuThreads: defaultCpuThreads(),
  cpuBatch: 128,
  cpuInteractiveCtx: 4096,
  cpuMaxTokens: 128,
  cpuKeepAlive: "30m"
};

function readUserConfig() {
  try {
    const raw = fs.readFileSync(configPath, "utf8");
    const parsed = JSON.parse(raw);
    const config = {};
    if (parsed.model) config.model = parsed.model;
    if (parsed.context) config.ctx = parsed.context;
    if (parsed.kvDiskDir) config.kvDiskDir = parsed.kvDiskDir;
    if (parsed.kvDiskSpaceMb) config.kvDiskSpaceMb = parsed.kvDiskSpaceMb;
    if (parsed.ollamaUrl) config.ollamaUrl = parsed.ollamaUrl;
    if (parsed.llamaUrl) config.llamaUrl = parsed.llamaUrl;
    if (parsed.llamaApiKey) config.llamaApiKey = parsed.llamaApiKey;
    if (parsed.ikLlamaServer) config.ikLlamaServer = parsed.ikLlamaServer;
    if (parsed.ikLlamaModel) config.ikLlamaModel = parsed.ikLlamaModel;
    if (parsed.ikLlamaHost) config.ikLlamaHost = parsed.ikLlamaHost;
    if (parsed.ikLlamaPort) config.ikLlamaPort = Number(parsed.ikLlamaPort);
    if (parsed.ikLlamaAutoStart !== undefined) config.ikLlamaAutoStart = parsed.ikLlamaAutoStart;
    if (parsed.ikLlamaStartupTimeoutMs) config.ikLlamaStartupTimeoutMs = Number(parsed.ikLlamaStartupTimeoutMs);
    if (parsed.backend) config.backend = parsed.backend;
    if (parsed.nativeModel) config.nativeModel = parsed.nativeModel;
    if (parsed.nativeMaxTokens) config.nativeMaxTokens = Number(parsed.nativeMaxTokens);
    if (parsed.nativeTimeoutMs) config.nativeTimeoutMs = Number(parsed.nativeTimeoutMs);
    if (parsed.cpuOnly !== undefined) config.cpuOnly = parsed.cpuOnly;
    if (parsed.cpuProfile) config.cpuProfile = parsed.cpuProfile;
    if (parsed.cpuThreads) config.cpuThreads = Number(parsed.cpuThreads);
    if (parsed.cpuBatch) config.cpuBatch = Number(parsed.cpuBatch);
    if (parsed.cpuInteractiveCtx) config.cpuInteractiveCtx = Number(parsed.cpuInteractiveCtx);
    if (parsed.cpuMaxTokens) config.cpuMaxTokens = Number(parsed.cpuMaxTokens);
    if (parsed.cpuKeepAlive) config.cpuKeepAlive = parsed.cpuKeepAlive;
    return config;
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
    else if (arg === "--kv-disk-dir" && next) config.kvDiskDir = path.resolve(next), i += 1;
    else if (arg === "--kv-disk-space-mb" && next) config.kvDiskSpaceMb = Number(next), i += 1;
    else if (arg === "--ollama" && next) config.ollamaUrl = next, i += 1;
    else if (arg === "--llama-url" && next) config.llamaUrl = next, i += 1;
    else if (arg === "--llama-api-key" && next) config.llamaApiKey = next, i += 1;
    else if (arg === "--ik-llama-server" && next) config.ikLlamaServer = path.resolve(next), i += 1;
    else if (arg === "--ik-llama-model" && next) config.ikLlamaModel = path.resolve(next), i += 1;
    else if (arg === "--ik-llama-host" && next) config.ikLlamaHost = next, i += 1;
    else if (arg === "--ik-llama-port" && next) config.ikLlamaPort = Number(next), i += 1;
    else if (arg === "--no-ik-llama-autostart") config.ikLlamaAutoStart = false;
    else if (arg === "--backend" && next) config.backend = next, i += 1;
    else if (arg === "--native-model" && next) config.nativeModel = path.resolve(next), i += 1;
    else if (arg === "--native-max-tokens" && next) config.nativeMaxTokens = Number(next), i += 1;
    else if (arg === "--native-timeout-ms" && next) config.nativeTimeoutMs = Number(next), i += 1;
    else if (arg === "--cpu-profile" && next) config.cpuProfile = next, i += 1;
    else if (arg === "--cpu-threads" && next) config.cpuThreads = Number(next), i += 1;
    else if (arg === "--cpu-batch" && next) config.cpuBatch = Number(next), i += 1;
    else if (arg === "--cpu-ctx" && next) config.cpuInteractiveCtx = Number(next), i += 1;
    else if (arg === "--cpu-max-tokens" && next) config.cpuMaxTokens = Number(next), i += 1;
    else if (arg === "--cpu-keep-alive" && next) config.cpuKeepAlive = next, i += 1;
    else if (arg === "--allow-gpu") config.cpuOnly = false;
    else if (arg === "--help" || arg === "-h") return { command: "help", config };
  }

  config.host = process.env.IRONMIND_HOST || config.host;
  config.port = Number(process.env.IRONMIND_PORT || config.port);
  config.model = process.env.IRONMIND_MODEL || config.model;
  config.ctx = Number(process.env.IRONMIND_CTX || config.ctx);
  config.kvDiskDir = path.resolve(process.env.IRONMIND_KV_DISK_DIR || config.kvDiskDir);
  config.kvDiskSpaceMb = Number(process.env.IRONMIND_KV_DISK_SPACE_MB || config.kvDiskSpaceMb);
  config.ollamaUrl = process.env.IRONMIND_OLLAMA_URL || config.ollamaUrl;
  config.llamaUrl = process.env.IRONMIND_LLAMA_URL || config.llamaUrl;
  config.llamaApiKey = process.env.IRONMIND_LLAMA_API_KEY || config.llamaApiKey;
  config.ikLlamaServer = process.env.IRONMIND_IK_LLAMA_SERVER || config.ikLlamaServer;
  config.ikLlamaModel = process.env.IRONMIND_IK_LLAMA_MODEL || config.ikLlamaModel;
  config.ikLlamaHost = process.env.IRONMIND_IK_LLAMA_HOST || config.ikLlamaHost;
  config.ikLlamaPort = Number(process.env.IRONMIND_IK_LLAMA_PORT || config.ikLlamaPort);
  config.ikLlamaAutoStart = process.env.IRONMIND_IK_LLAMA_AUTOSTART === undefined
    ? config.ikLlamaAutoStart
    : !["0", "false", "no", "off"].includes(String(process.env.IRONMIND_IK_LLAMA_AUTOSTART).toLowerCase());
  config.ikLlamaStartupTimeoutMs = Number(process.env.IRONMIND_IK_LLAMA_STARTUP_TIMEOUT_MS || config.ikLlamaStartupTimeoutMs);
  config.backend = process.env.IRONMIND_BACKEND || config.backend;
  config.nativeModel = process.env.IRONMIND_NATIVE_MODEL || config.nativeModel;
  config.nativeMaxTokens = Number(process.env.IRONMIND_NATIVE_MAX_TOKENS || config.nativeMaxTokens);
  config.nativeTimeoutMs = Number(process.env.IRONMIND_NATIVE_TIMEOUT_MS || config.nativeTimeoutMs);
  config.cpuOnly = process.env.IRONMIND_CPU_ONLY === undefined
    ? config.cpuOnly
    : !["0", "false", "no", "off"].includes(String(process.env.IRONMIND_CPU_ONLY).toLowerCase());
  config.cpuProfile = process.env.IRONMIND_CPU_PROFILE || config.cpuProfile;
  config.cpuThreads = Number(process.env.IRONMIND_CPU_THREADS || config.cpuThreads);
  config.cpuBatch = Number(process.env.IRONMIND_CPU_BATCH || config.cpuBatch);
  config.cpuInteractiveCtx = Number(process.env.IRONMIND_CPU_CTX || config.cpuInteractiveCtx);
  config.cpuMaxTokens = Number(process.env.IRONMIND_CPU_MAX_TOKENS || config.cpuMaxTokens);
  config.cpuKeepAlive = process.env.IRONMIND_CPU_KEEP_ALIVE || config.cpuKeepAlive;
  if (config.nativeModel) config.nativeModel = path.resolve(config.nativeModel);
  config.ollamaUrl = config.ollamaUrl.replace(/\/+$/, "");
  if (config.ikLlamaServer) config.ikLlamaServer = path.resolve(config.ikLlamaServer);
  if (config.ikLlamaModel) config.ikLlamaModel = path.resolve(config.ikLlamaModel);
  if (config.backend === "ik_llama") {
    config.llamaUrl = `http://${config.ikLlamaHost}:${config.ikLlamaPort}`;
  }
  config.llamaUrl = config.llamaUrl.replace(/\/+$/, "");
  if (!["auto", "ollama", "native", "llama", "ik_llama"].includes(config.backend)) config.backend = "auto";
  Object.assign(config, resolveCpuPerformanceConfig(config));
  return { command, config };
}

function usage() {
  console.log(`IronMind

Usage:
  ironmind [serve] [--host 127.0.0.1] [--port 4141] [--model qwen3-coder:30b] [--ctx 131072]
           [--kv-disk-dir ~/.ironmind/kvcache] [--kv-disk-space-mb 16384]
           [--backend auto|ollama|native|llama|ik_llama] [--native-model C:\\path\\to\\model.gguf]
           [--llama-url http://127.0.0.1:8080]
           [--ik-llama-server C:\\path\\to\\llama-server.exe --ik-llama-model C:\\path\\to\\model.gguf]
           [--native-max-tokens 16] [--native-timeout-ms 300000]
           [--cpu-profile low-latency|balanced|full-context] [--cpu-threads N]
           [--cpu-batch N] [--cpu-ctx N] [--cpu-max-tokens N] [--cpu-keep-alive 30m]
  ironmind doctor
  ironmind inspect <model.gguf>
  ironmind tokenize <model.gguf> <text>
  ironmind map <model.gguf>
  ironmind native <model.gguf> [--decode TOKEN] [--tokens CSV|--tokens-file PATH --generate N] [--ctx N]

Environment:
  IRONMIND_MODEL
  IRONMIND_CTX
  IRONMIND_KV_DISK_DIR
  IRONMIND_KV_DISK_SPACE_MB
  IRONMIND_PORT
  IRONMIND_OLLAMA_URL
  IRONMIND_LLAMA_URL
  IRONMIND_LLAMA_API_KEY
  IRONMIND_IK_LLAMA_SERVER
  IRONMIND_IK_LLAMA_MODEL
  IRONMIND_IK_LLAMA_HOST
  IRONMIND_IK_LLAMA_PORT
  IRONMIND_IK_LLAMA_AUTOSTART
  IRONMIND_BACKEND
  IRONMIND_NATIVE_MODEL
  IRONMIND_NATIVE_MAX_TOKENS
  IRONMIND_NATIVE_TIMEOUT_MS
  IRONMIND_CPU_ONLY
  IRONMIND_CPU_PROFILE
  IRONMIND_CPU_THREADS
  IRONMIND_CPU_BATCH
  IRONMIND_CPU_CTX
  IRONMIND_CPU_MAX_TOKENS
  IRONMIND_CPU_KEEP_ALIVE
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

async function waitForHttp(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(1000, Math.max(1, deadline - Date.now())));
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (response.ok || response.status === 404 || response.status === 405) return true;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(timer);
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  throw new Error(`Timed out waiting for ${url}${lastError ? ` (${lastError.message})` : ""}`);
}

async function isHttpReachable(url) {
  try {
    await waitForHttp(url, 500);
    return true;
  } catch {
    return false;
  }
}

function ikLlamaArgs(config) {
  const cpu = resolveCpuPerformanceConfig(config);
  const args = [
    "--model", config.ikLlamaModel,
    "--host", config.ikLlamaHost,
    "--port", String(config.ikLlamaPort),
    "--ctx-size", String(cpu.interactiveContext || config.ctx),
    "--threads", String(cpu.threads),
    "--batch-size", String(cpu.batch),
    "--n-gpu-layers", "0"
  ];
  return args;
}

async function ensureIkLlamaRuntime(config) {
  if (config.backend !== "ik_llama" || !config.ikLlamaAutoStart) return { managed: false, started: false };
  if (await isHttpReachable(`${config.llamaUrl}/health`)) return { managed: false, started: false, alreadyRunning: true };
  if (!config.ikLlamaServer || !config.ikLlamaModel) {
    throw new Error("ik_llama backend requires IRONMIND_IK_LLAMA_SERVER and IRONMIND_IK_LLAMA_MODEL.");
  }
  if (!fs.existsSync(config.ikLlamaServer)) throw new Error(`ik_llama server not found: ${config.ikLlamaServer}`);
  if (!fs.existsSync(config.ikLlamaModel)) throw new Error(`ik_llama model not found: ${config.ikLlamaModel}`);

  const args = ikLlamaArgs(config);
  managedIkLlamaProcess = spawn(config.ikLlamaServer, args, {
    cwd: path.dirname(config.ikLlamaServer),
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"]
  });
  managedIkLlamaProcess.stdout?.setEncoding("utf8");
  managedIkLlamaProcess.stderr?.setEncoding("utf8");
  managedIkLlamaProcess.stdout?.on("data", (chunk) => process.stdout.write(`[ik_llama] ${chunk}`));
  managedIkLlamaProcess.stderr?.on("data", (chunk) => process.stderr.write(`[ik_llama] ${chunk}`));
  managedIkLlamaProcess.on("exit", (code, signal) => {
    if (managedIkLlamaProcess) {
      console.log(`ik_llama runtime exited (${signal || code})`);
      managedIkLlamaProcess = null;
    }
  });

  await waitForHttp(`${config.llamaUrl}/health`, Number(config.ikLlamaStartupTimeoutMs || 30000));
  return { managed: true, started: true, pid: managedIkLlamaProcess.pid, args };
}

async function readJson(req) {
  let body = "";
  for await (const chunk of req) body += chunk;
  if (!body.trim()) return {};
  return JSON.parse(body);
}

function normalizeChatPayload(config, body = {}) {
  const messages = ensureCpuSystemMessage(canonicalizeMessages(body.messages || []));
  return {
    ...body,
    model: body.model || config.model,
    ctx: Number(body.ctx || config.ctx),
    messages,
    tools: canonicalizeTools(body.tools || [])
  };
}

async function runNativeChat(config, body) {
  const payload = normalizeChatPayload(config, body);
  const snapshot = createContextSnapshot(config, payload);
  const out = await nativeChat(config, payload, { saveKvPath: snapshot.ironKvPath });
  const contextSnapshot = await finalizeNativeContextSnapshot(config, snapshot, { native: out.native });
  return { payload, out, contextSnapshot };
}

function openAiMessageFromBackend(out) {
  const message = {
    role: "assistant",
    content: chunkText(out)
  };
  const toolCalls = canonicalizeToolCalls(out.message?.tool_calls || out.tool_calls || []);
  if (toolCalls.length) message.tool_calls = toolCalls;
  return message;
}

function finishReasonForMessage(message) {
  return message.tool_calls?.length ? "tool_calls" : "stop";
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
  const { options, keepAlive } = applyCpuPerformanceOptions(config, payload);
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
      keep_alive: keepAlive,
      options
    })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Ollama ${response.status}: ${detail}`);
  }

  return response;
}

function isLlamaCompatibleBackend(config) {
  return config.backend === "llama" || config.backend === "ik_llama";
}

function llamaRequestBody(config, payload = {}, stream = false) {
  const { options } = applyCpuPerformanceOptions(config, payload);
  const body = {
    model: payload.model || config.model,
    messages: payload.messages || [],
    stream,
    temperature: options.temperature,
    top_p: options.top_p,
    max_tokens: options.num_predict
  };
  if (payload.tools?.length) body.tools = payload.tools;
  if (payload.tool_choice) body.tool_choice = payload.tool_choice;
  return body;
}

async function llamaChat(config, payload, stream = false) {
  const headers = { "content-type": "application/json" };
  if (config.llamaApiKey) headers.authorization = `Bearer ${config.llamaApiKey}`;
  const response = await fetch(`${config.llamaUrl}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(llamaRequestBody(config, payload, stream))
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`llama-server ${response.status}: ${detail}`);
  }

  return response;
}

async function completeLlamaChat(config, payload) {
  const response = await llamaChat(config, payload, false);
  const out = await response.json();
  const choice = out.choices?.[0] || {};
  return {
    model: out.model || payload.model || config.model,
    message: choice.message || { role: "assistant", content: choice.text || "" },
    prompt_eval_count: out.usage?.prompt_tokens || 0,
    eval_count: out.usage?.completion_tokens || 0,
    done: true,
    raw: out
  };
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
    const body = normalizeChatPayload(config, await readJson(req));
    if (shouldUseNativeBackend(config, body)) {
      const { out, contextSnapshot } = await runNativeChat(config, body);
      res.writeHead(200, {
        "content-type": "application/x-ndjson; charset=utf-8",
        "cache-control": "no-cache"
      });
      if (out.message?.content) res.write(JSON.stringify({ type: "delta", content: out.message.content }) + "\n");
      if (out.message?.tool_calls?.length) res.write(JSON.stringify({ type: "tool_calls", toolCalls: out.message.tool_calls }) + "\n");
      res.write(JSON.stringify({
        type: "done",
        model: out.model,
        promptEvalCount: out.prompt_eval_count,
        evalCount: out.eval_count,
        contextSnapshot
      }) + "\n");
      return res.end();
    }

    if (isLlamaCompatibleBackend(config)) {
      const contextSnapshot = await saveContextSnapshot(config, body);
      const out = await completeLlamaChat(config, body);
      res.writeHead(200, {
        "content-type": "application/x-ndjson; charset=utf-8",
        "cache-control": "no-cache"
      });
      const content = chunkText(out);
      if (content) res.write(JSON.stringify({ type: "delta", content }) + "\n");
      if (out.message?.tool_calls?.length) res.write(JSON.stringify({ type: "tool_calls", toolCalls: out.message.tool_calls }) + "\n");
      res.write(JSON.stringify({
        type: "done",
        model: out.model,
        promptEvalCount: out.prompt_eval_count,
        evalCount: out.eval_count,
        contextSnapshot
      }) + "\n");
      return res.end();
    }

    const contextSnapshot = await saveContextSnapshot(config, body);
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
          evalCount: chunk.eval_count,
          totalDurationMs: chunk.total_duration ? Math.round(chunk.total_duration / 1e6) : null,
          loadDurationMs: chunk.load_duration ? Math.round(chunk.load_duration / 1e6) : null,
          promptEvalDurationMs: chunk.prompt_eval_duration ? Math.round(chunk.prompt_eval_duration / 1e6) : null,
          evalDurationMs: chunk.eval_duration ? Math.round(chunk.eval_duration / 1e6) : null,
          tokensPerSecond: chunk.eval_duration && chunk.eval_count
            ? Math.round((chunk.eval_count / (chunk.eval_duration / 1e9)) * 10) / 10
            : null,
          contextSnapshot
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
    const body = normalizeChatPayload(config, await readJson(req));
    const model = body.model || config.model;

    if (shouldUseNativeBackend(config, body)) {
      const { out, contextSnapshot } = await runNativeChat(config, body);
      const message = openAiMessageFromBackend(out);
      if (body.stream === false) {
        return json(res, 200, {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              message,
              finish_reason: finishReasonForMessage(message)
            }
          ],
          usage: {
            prompt_tokens: out.prompt_eval_count || 0,
            completion_tokens: out.eval_count || 0,
            total_tokens: (out.prompt_eval_count || 0) + (out.eval_count || 0)
          },
          ironmind: { contextSnapshot, backend: "native" }
        });
      }

      const id = `chatcmpl-${Date.now()}`;
      res.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        connection: "keep-alive"
      });
      if (message.content) res.write(`data: ${JSON.stringify(openAiChunk(id, model, message.content))}\n\n`);
      const done = openAiChunk(id, model, "", finishReasonForMessage(message));
      if (message.tool_calls?.length) done.choices[0].delta.tool_calls = message.tool_calls;
      done.ironmind = { contextSnapshot, backend: "native" };
      res.write(`data: ${JSON.stringify(done)}\n\n`);
      res.write("data: [DONE]\n\n");
      return res.end();
    }

    if (isLlamaCompatibleBackend(config)) {
      const contextSnapshot = await saveContextSnapshot(config, body);
      const out = await completeLlamaChat(config, body);
      const message = openAiMessageFromBackend(out);
      if (body.stream === false) {
        return json(res, 200, {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: out.model || model,
          choices: [
            {
              index: 0,
              message,
              finish_reason: finishReasonForMessage(message)
            }
          ],
          usage: {
            prompt_tokens: out.prompt_eval_count || 0,
            completion_tokens: out.eval_count || 0,
            total_tokens: (out.prompt_eval_count || 0) + (out.eval_count || 0)
          },
          ironmind: { contextSnapshot, backend: config.backend }
        });
      }

      const id = `chatcmpl-${Date.now()}`;
      res.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        connection: "keep-alive"
      });
      if (message.content) res.write(`data: ${JSON.stringify(openAiChunk(id, model, message.content))}\n\n`);
      const done = openAiChunk(id, model, "", finishReasonForMessage(message));
      if (message.tool_calls?.length) done.choices[0].delta.tool_calls = message.tool_calls;
      done.ironmind = { contextSnapshot, backend: config.backend };
      res.write(`data: ${JSON.stringify(done)}\n\n`);
      res.write("data: [DONE]\n\n");
      return res.end();
    }

    const contextSnapshot = await saveContextSnapshot(config, body);

    if (body.stream === false) {
      const response = await ollamaChat(config, body, false);
      const out = await response.json();
      const message = openAiMessageFromBackend(out);
      return json(res, 200, {
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message,
            finish_reason: finishReasonForMessage(message)
          }
        ],
        usage: {
          prompt_tokens: out.prompt_eval_count || 0,
          completion_tokens: out.eval_count || 0,
          total_tokens: (out.prompt_eval_count || 0) + (out.eval_count || 0)
        },
        ironmind: { contextSnapshot, backend: "ollama" }
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
      if (chunk.done) {
        const done = openAiChunk(id, model, "", "stop");
        done.ironmind = { contextSnapshot, backend: "ollama" };
        res.write(`data: ${JSON.stringify(done)}\n\n`);
      }
    });
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (error) {
    json(res, 502, { error: { message: error.message, type: "backend_unavailable" } });
  }
}

async function completeChatOnce(config, body) {
  const payload = normalizeChatPayload(config, body);
  if (shouldUseNativeBackend(config, payload)) {
    const result = await runNativeChat(config, payload);
    return { ...result, backend: "native" };
  }
  const contextSnapshot = await saveContextSnapshot(config, payload);
  if (isLlamaCompatibleBackend(config)) {
    const out = await completeLlamaChat(config, payload);
    return { payload, out, contextSnapshot, backend: config.backend };
  }
  const response = await ollamaChat(config, payload, false);
  const out = await response.json();
  return { payload, out, contextSnapshot, backend: "ollama" };
}

function textFromContentParts(content) {
  if (!Array.isArray(content)) return content == null ? "" : String(content);
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (part?.type === "text" || part?.type === "input_text" || part?.type === "output_text") return part.text || "";
    if (part?.text) return part.text;
    return "";
  }).join("");
}

function responseInputMessages(body) {
  const messages = [];
  if (body.instructions) messages.push({ role: "system", content: body.instructions });
  if (typeof body.input === "string") messages.push({ role: "user", content: body.input });
  else if (Array.isArray(body.input)) {
    for (const item of body.input) {
      if (item?.role) {
        messages.push({
          role: item.role === "developer" ? "system" : item.role,
          content: textFromContentParts(item.content)
        });
      }
    }
  }
  return messages.length ? messages : (body.messages || []);
}

function responseOutputFromMessage(message) {
  const content = [];
  if (message.content) {
    content.push({ type: "output_text", text: message.content, annotations: [] });
  }
  for (const call of message.tool_calls || []) {
    content.push({
      type: "function_call",
      call_id: call.id,
      name: call.function.name,
      arguments: call.function.arguments
    });
  }
  return [{
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    content
  }];
}

async function handleResponses(req, res, config) {
  try {
    const body = await readJson(req);
    const chatBody = {
      ...body,
      model: body.model || config.model,
      messages: responseInputMessages(body),
      tools: body.tools || [],
      max_tokens: body.max_output_tokens || body.max_tokens
    };
    const { out, contextSnapshot, backend } = await completeChatOnce(config, chatBody);
    const message = openAiMessageFromBackend(out);
    const output = responseOutputFromMessage(message);
    const response = {
      id: `resp_${Date.now()}`,
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      status: "completed",
      model: chatBody.model,
      output,
      output_text: message.content || "",
      usage: {
        input_tokens: out.prompt_eval_count || 0,
        output_tokens: out.eval_count || 0,
        total_tokens: (out.prompt_eval_count || 0) + (out.eval_count || 0)
      },
      ironmind: { contextSnapshot, backend }
    };
    return json(res, 200, response);
  } catch (error) {
    json(res, 502, { error: { message: error.message, type: "backend_unavailable" } });
  }
}

function anthropicMessages(body) {
  const messages = [];
  if (body.system) messages.push({ role: "system", content: textFromContentParts(body.system) });
  for (const message of body.messages || []) {
    const textParts = [];
    const toolCalls = [];
    if (Array.isArray(message.content)) {
      for (const part of message.content) {
        if (part?.type === "text") textParts.push(part.text || "");
        else if (part?.type === "tool_use") {
          toolCalls.push({
            id: part.id,
            type: "function",
            function: {
              name: part.name,
              arguments: part.input || {}
            }
          });
        } else if (part?.type === "tool_result") {
          messages.push({
            role: "tool",
            tool_call_id: part.tool_use_id,
            content: textFromContentParts(part.content)
          });
        }
      }
    } else {
      textParts.push(textFromContentParts(message.content));
    }
    if (message.role !== "tool") {
      const out = { role: message.role, content: textParts.join("") };
      if (toolCalls.length) out.tool_calls = toolCalls;
      messages.push(out);
    }
  }
  return messages;
}

function anthropicContentFromMessage(message) {
  const content = [];
  if (message.content) content.push({ type: "text", text: message.content });
  for (const call of message.tool_calls || []) {
    content.push({
      type: "tool_use",
      id: call.id,
      name: call.function.name,
      input: JSON.parse(call.function.arguments || "{}")
    });
  }
  return content.length ? content : [{ type: "text", text: "" }];
}

async function handleAnthropicMessages(req, res, config) {
  try {
    const body = await readJson(req);
    const chatBody = {
      ...body,
      model: body.model || config.model,
      messages: anthropicMessages(body),
      tools: body.tools || [],
      max_tokens: body.max_tokens
    };
    const { out, contextSnapshot, backend } = await completeChatOnce(config, chatBody);
    const message = openAiMessageFromBackend(out);
    return json(res, 200, {
      id: `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: chatBody.model,
      content: anthropicContentFromMessage(message),
      stop_reason: message.tool_calls?.length ? "tool_use" : "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: out.prompt_eval_count || 0,
        output_tokens: out.eval_count || 0
      },
      ironmind: { contextSnapshot, backend }
    });
  } catch (error) {
    json(res, 502, { type: "error", error: { type: "backend_unavailable", message: error.message } });
  }
}

async function handleClinicalTriage(req, res) {
  try {
    const body = await readJson(req);
    return json(res, 200, createClinicalTriage(body, { thresholds: body.thresholds }));
  } catch (error) {
    json(res, 400, { error: { message: error.message, type: "invalid_clinical_triage_request" } });
  }
}

async function handleClinicalImageQuality(req, res) {
  try {
    const body = await readJson(req);
    return json(res, 200, assessImageQuality(body, body.options || {}));
  } catch (error) {
    json(res, 400, { error: { message: error.message, type: "invalid_image_quality_request" } });
  }
}

async function handleClinicalScreening(req, res) {
  try {
    const body = await readJson(req);
    return json(res, 200, createClinicalScreeningCase(body, body.options || {}));
  } catch (error) {
    json(res, 400, { error: { message: error.message, type: "invalid_clinical_screening_request" } });
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
        backend: backendDescription(config),
        backendMode: config.backend,
        llamaUrl: config.llamaUrl,
        ikLlama: {
          server: config.ikLlamaServer || null,
          model: config.ikLlamaModel || null,
          host: config.ikLlamaHost,
          port: config.ikLlamaPort,
          autoStart: config.ikLlamaAutoStart,
          managedPid: managedIkLlamaProcess?.pid || null
        },
        cpuPerformance: resolveCpuPerformanceConfig(config),
        nativeModel: config.nativeModel || null,
        nativeCandidate: nativeModelPath(config, { model: config.model }),
        nativeMaxTokens: config.nativeMaxTokens,
        nativeTimeoutMs: config.nativeTimeoutMs,
        kvDiskDir: config.kvDiskDir,
        kvDiskSpaceMb: config.kvDiskSpaceMb,
        contextStore: await contextStoreStats(config)
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

    if (req.method === "POST" && url.pathname === "/v1/responses") {
      return handleResponses(req, res, config);
    }

    if (req.method === "POST" && url.pathname === "/v1/messages") {
      return handleAnthropicMessages(req, res, config);
    }

    if (req.method === "POST" && url.pathname === "/v1/clinical/triage") {
      return handleClinicalTriage(req, res);
    }

    if (req.method === "POST" && url.pathname === "/v1/clinical/image/quality") {
      return handleClinicalImageQuality(req, res);
    }

    if (req.method === "POST" && url.pathname === "/v1/clinical/screening") {
      return handleClinicalScreening(req, res);
    }

    if (req.method === "GET") return serveStatic(req, res);
    return text(res, 405, "method not allowed");
  });
}

async function doctor(config) {
  console.log("IronMind doctor");
  console.log(`  model:   ${config.model}`);
  console.log(`  context: ${config.ctx}`);
  console.log(`  kv disk: ${config.kvDiskDir}`);
  console.log(`  kv cap:  ${config.kvDiskSpaceMb} MB`);
  console.log(`  runtime: ${backendDescription(config)}`);
  console.log(`  ollama:  ${config.ollamaUrl}`);
  console.log(`  llama:   ${config.llamaUrl}`);
  const cpu = resolveCpuPerformanceConfig(config);
  console.log(`  cpu mode: ${cpu.cpuOnly ? "CPU-only (num_gpu=0)" : "GPU allowed by config"}`);
  console.log(`  cpu profile: ${cpu.profile}, ctx=${cpu.interactiveContext || "full"}, threads=${cpu.threads}, batch=${cpu.batch}, max_tokens=${cpu.maxTokens}`);

  if (shouldUseNativeBackend(config, { model: config.model })) {
    const runner = nativeRunnerPath();
    const modelPath = nativeModelPath(config, { model: config.model });
    const runnerOk = fs.existsSync(runner);
    const modelOk = modelPath ? fs.existsSync(modelPath) : false;
    console.log(`  native runner: ${runnerOk ? runner : "not built"}`);
    console.log(`  native model:  ${modelOk ? modelPath : "not configured"}`);
    if (!runnerOk || !modelOk) process.exitCode = 1;
    return;
  }

  if (isLlamaCompatibleBackend(config)) {
    if (config.backend === "ik_llama") {
      console.log(`  ik server: ${config.ikLlamaServer || "not configured"}`);
      console.log(`  ik model:  ${config.ikLlamaModel || "not configured"}`);
    }
    try {
      await waitForHttp(`${config.llamaUrl}/health`, 1000);
      console.log("  backend: reachable");
    } catch (error) {
      console.log(`  backend: not reachable (${error.message})`);
      process.exitCode = 1;
    }
    return;
  }

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

async function inspectModel(filePath) {
  if (!filePath) {
    console.error("usage: ironmind inspect <model.gguf>");
    process.exitCode = 2;
    return;
  }

  const info = await inspectGguf(filePath);
  const summary = summarizeGguf(info);
  const validation = validateIronMindTarget(info);

  console.log("IronMind GGUF inspector");
  for (const [key, value] of Object.entries(summary)) {
    console.log(`  ${key.padEnd(20)} ${value ?? "-"}`);
  }
  console.log(`  target              ${validation.ok ? "compatible" : "not compatible"}`);
  for (const issue of validation.issues) console.log(`  issue               ${issue}`);
  for (const warning of validation.warnings) console.log(`  warning             ${warning}`);
}

async function tokenizeModel(args) {
  const filePath = args[0];
  const text = args.slice(1).join(" ");
  if (!filePath || !text) {
    console.error("usage: ironmind tokenize <model.gguf> <text>");
    process.exitCode = 2;
    return;
  }
  const tokenizer = await loadQwen3Tokenizer(filePath);
  const ids = tokenizer.encode(text);
  console.log(JSON.stringify({ summary: tokenizer.summary(), ids, count: ids.length, decoded: tokenizer.decode(ids) }, null, 2));
}

async function mapModel(filePath) {
  if (!filePath) {
    console.error("usage: ironmind map <model.gguf>");
    process.exitCode = 2;
    return;
  }
  const map = await loadQwenTensorMap(filePath);
  console.log("IronMind tensor map");
  for (const [key, value] of Object.entries(summarizeTensorMap(map))) {
    console.log(`  ${key.padEnd(16)} ${value}`);
  }
  for (const missing of map.missing.slice(0, 20)) console.log(`  missing          ${missing}`);
  if (map.missing.length > 20) console.log(`  missing          ... ${map.missing.length - 20} more`);
}

function nativeModel(args) {
  const filePath = args[0];
  if (!filePath) {
    console.error("usage: ironmind native <model.gguf> [--decode TOKEN] [--ctx N]");
    process.exitCode = 2;
    return;
  }
  const exe = process.platform === "win32"
    ? path.join(rootDir, "build", "Release", "ironmind-native.exe")
    : path.join(rootDir, "build", "ironmind-native");
  if (!fs.existsSync(exe)) {
    console.error("native runner not built; run `npm run native:build` first");
    process.exitCode = 1;
    return;
  }
  const result = spawnSync(exe, [path.resolve(filePath), ...args.slice(1)], { stdio: "inherit" });
  process.exitCode = result.status ?? 1;
}

async function main() {
  const { command, config } = parseArgs(process.argv.slice(2));

  if (command === "help") return usage();
  if (command === "doctor") return doctor(config);
  if (command === "inspect") return inspectModel(process.argv.slice(3)[0]);
  if (command === "tokenize") return tokenizeModel(process.argv.slice(3));
  if (command === "map") return mapModel(process.argv.slice(3)[0]);
  if (command === "native") return nativeModel(process.argv.slice(3));
  if (command !== "serve") return usage();

  const ikRuntime = await ensureIkLlamaRuntime(config);
  const server = createServer(config);
  server.listen(config.port, config.host, () => {
    console.log(`IronMind listening on http://${config.host}:${config.port}`);
    console.log(`Model: ${config.model}`);
    console.log(`Backend: ${backendDescription(config)}`);
    if (ikRuntime.started) console.log(`ik_llama managed runtime pid: ${ikRuntime.pid}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
