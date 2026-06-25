import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { renderQwen3Chat } from "./qwen3Prompt.mjs";
import { loadQwen3Tokenizer } from "./tokenizer.mjs";
import { extractToolCallsFromText } from "./toolCalls.mjs";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");
const ollamaModelDir = path.join(os.homedir(), ".ollama", "models");

export function nativeRunnerPath() {
  return process.platform === "win32"
    ? path.join(rootDir, "build", "Release", "ironmind-native.exe")
    : path.join(rootDir, "build", "ironmind-native");
}

export function isGgufModel(value) {
  if (typeof value !== "string" || !value) return false;
  if (value.toLowerCase().endsWith(".gguf")) return true;
  try {
    const handle = fsSync.openSync(path.resolve(value), "r");
    try {
      const magic = Buffer.alloc(4);
      return fsSync.readSync(handle, magic, 0, 4, 0) === 4 && magic.toString("ascii") === "GGUF";
    } finally {
      fsSync.closeSync(handle);
    }
  } catch {
    return false;
  }
}

function parseOllamaModelRef(ref = "") {
  const source = String(ref || "");
  if (!source || source.includes("\\") || source.includes(":\\") || source.startsWith(".")) return null;
  const slash = source.lastIndexOf("/");
  const colon = source.lastIndexOf(":");
  const hasTag = colon > slash;
  const withoutTag = hasTag ? source.slice(0, colon) : source;
  const tag = hasTag ? source.slice(colon + 1) : "latest";
  const parts = withoutTag.split("/").filter(Boolean);
  if (!parts.length) return null;
  if (parts.length === 1) {
    return { host: "registry.ollama.ai", namespace: ["library"], name: parts[0], tag };
  }
  if (parts[0].includes(".")) {
    return { host: parts[0], namespace: parts.slice(1, -1), name: parts.at(-1), tag };
  }
  return { host: "registry.ollama.ai", namespace: parts.slice(0, -1), name: parts.at(-1), tag };
}

function ollamaManifestPath(modelRef) {
  const parsed = parseOllamaModelRef(modelRef);
  if (!parsed) return null;
  return path.join(ollamaModelDir, "manifests", parsed.host, ...parsed.namespace, parsed.name, parsed.tag);
}

function blobPathFromDigest(digest) {
  if (typeof digest !== "string" || !digest.startsWith("sha256:")) return null;
  return path.join(ollamaModelDir, "blobs", digest.replace(":", "-"));
}

export function resolveOllamaGgufModel(modelRef) {
  const manifestPath = ollamaManifestPath(modelRef);
  if (!manifestPath || !fsSync.existsSync(manifestPath)) return null;
  try {
    const manifest = JSON.parse(fsSync.readFileSync(manifestPath, "utf8"));
    const modelLayer = (manifest.layers || []).find((layer) => layer.mediaType === "application/vnd.ollama.image.model");
    const blobPath = blobPathFromDigest(modelLayer?.digest);
    return blobPath && isGgufModel(blobPath) ? blobPath : null;
  } catch {
    return null;
  }
}

export function nativeModelPath(config, payload = {}) {
  const explicit = config.nativeModel || payload.nativeModel;
  if (explicit && isGgufModel(explicit)) return path.resolve(explicit);

  const candidate = payload.model || config.model;
  if (isGgufModel(candidate)) return path.resolve(candidate);
  return resolveOllamaGgufModel(candidate);
}

function hasExplicitNativeModel(config, payload = {}) {
  if (config.nativeModel || payload.nativeModel) return true;
  const candidate = payload.model || config.model;
  return isGgufModel(candidate);
}

export function shouldUseNativeBackend(config, payload = {}) {
  if (config.backend === "ollama") return false;
  if (config.backend === "llama") return false;
  if (config.backend === "ik_llama") return false;
  if (config.backend === "ik_worker") return false;
  if (config.backend === "ik_embedded") return false;
  const modelPath = nativeModelPath(config, payload);
  if (config.backend === "native") return true;
  return Boolean(modelPath && hasExplicitNativeModel(config, payload) && fsSync.existsSync(nativeRunnerPath()));
}

export function backendDescription(config) {
  const modelPath = nativeModelPath(config);
  if (config.backend === "native") return `native:${modelPath || "unconfigured"}`;
  if (config.backend === "ik_embedded") return config.ikEmbeddedPersistent === false
    ? `ik_embedded:${config.ikEmbeddedRunner || "build-ik/ironmind-ik-native"}`
    : `ik_embedded:persistent:${config.ikEmbeddedDaemon || config.ikEmbeddedRunner || "build-ik/ironmind-ik-daemon"}`;
  if (config.backend === "ik_worker") return `ik_worker:${config.ikLlamaWorker || config.ikLlamaServer || "unconfigured"}`;
  if (config.backend === "ik_llama") return `ik_llama:${config.llamaUrl || "http://127.0.0.1:8080"}`;
  if (config.backend === "llama") return `llama:${config.llamaUrl || "http://127.0.0.1:8080"}`;
  if (config.backend === "auto" && modelPath && hasExplicitNativeModel(config) && fsSync.existsSync(nativeRunnerPath())) return `native:${modelPath}`;
  return `ollama:${config.ollamaUrl}`;
}

function maxCompletionTokens(payload, fallback = 16) {
  const requested = payload.max_completion_tokens ?? payload.max_tokens ?? payload.options?.num_predict ?? fallback;
  const n = Number(requested);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(Math.floor(n), 2048);
}

function nativeContextSize(config, payload, promptTokenCount, completionTokens) {
  const requested = Number(payload.ctx || config.ctx || 0);
  const needed = Math.max(1, promptTokenCount + completionTokens + 1);
  if (!Number.isFinite(requested) || requested <= 0) return needed;
  return Math.min(Math.max(needed, 1), requested < needed ? needed : requested);
}

function runNative(args, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    const child = spawn(nativeRunnerPath(), args, {
      cwd: rootDir,
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = timeoutMs > 0 ? setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs) : null;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      if (timedOut) return reject(new Error(`native runner timed out after ${timeoutMs}ms`));
      if (code === 0) return resolve(stdout);
      reject(new Error((stderr || stdout || `native runner exited ${code}`).trim()));
    });
  });
}

export async function nativeChat(config, payload, options = {}) {
  const modelPath = nativeModelPath(config, payload);
  if (!modelPath) throw new Error("Native backend requires a GGUF model path or an Ollama model backed by GGUF.");
  if (!fsSync.existsSync(nativeRunnerPath())) {
    throw new Error("Native runner not built; run `npm run native:build` first.");
  }

  const tokenizer = await loadQwen3Tokenizer(modelPath);
  const renderedPrompt = renderQwen3Chat(payload.messages || [], {
    tools: payload.tools || [],
    think: payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort)
  });
  const promptTokens = tokenizer.encode(renderedPrompt);
  const completionTokens = maxCompletionTokens(payload, Number(config.nativeMaxTokens || 16));
  const nativeCtx = nativeContextSize(config, payload, promptTokens.length, completionTokens);
  const tokenFile = path.join(os.tmpdir(), `ironmind-${process.pid}-${Date.now()}.tokens`);
  await fs.writeFile(tokenFile, promptTokens.join(","), "utf8");

  const args = [
    modelPath,
    "--tokens-file", tokenFile,
    "--generate", String(completionTokens),
    "--ctx", String(nativeCtx),
    "--json"
  ];
  if (options.saveKvPath) args.push("--save-kv", options.saveKvPath);

  try {
    const output = await runNative(args, Number(config.nativeTimeoutMs || 300000));
    const native = JSON.parse(output);
    native.contextSize = nativeCtx;
    native.modelPath = modelPath;
    const generated = Array.isArray(native.tokenIds) ? native.tokenIds : [];
    const text = tokenizer.decode(generated);
    const parsed = extractToolCallsFromText(text);
    const message = {
      role: "assistant",
      content: parsed.content
    };
    if (parsed.tool_calls.length) message.tool_calls = parsed.tool_calls;

    return {
      model: payload.model || config.model,
      message,
      prompt_eval_count: native.promptTokens ?? promptTokens.length,
      eval_count: native.generatedTokens ?? generated.length,
      done: true,
      native
    };
  } finally {
    await fs.rm(tokenFile, { force: true });
  }
}
