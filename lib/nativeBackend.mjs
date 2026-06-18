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

export function nativeRunnerPath() {
  return process.platform === "win32"
    ? path.join(rootDir, "build", "Release", "ironmind-native.exe")
    : path.join(rootDir, "build", "ironmind-native");
}

export function isGgufModel(value) {
  return typeof value === "string" && value.toLowerCase().endsWith(".gguf");
}

export function nativeModelPath(config, payload = {}) {
  const candidate = config.nativeModel || payload.model || config.model;
  return isGgufModel(candidate) ? path.resolve(candidate) : null;
}

export function shouldUseNativeBackend(config, payload = {}) {
  if (config.backend === "ollama") return false;
  const modelPath = nativeModelPath(config, payload);
  if (config.backend === "native") return true;
  return Boolean(modelPath && fsSync.existsSync(nativeRunnerPath()));
}

export function backendDescription(config) {
  const modelPath = nativeModelPath(config);
  if (config.backend === "native") return `native:${modelPath || "unconfigured"}`;
  if (config.backend === "auto" && modelPath && fsSync.existsSync(nativeRunnerPath())) return `native:${modelPath}`;
  return `ollama:${config.ollamaUrl}`;
}

function maxCompletionTokens(payload) {
  const requested = payload.max_completion_tokens ?? payload.max_tokens ?? payload.options?.num_predict ?? 128;
  const n = Number(requested);
  if (!Number.isFinite(n) || n <= 0) return 128;
  return Math.min(Math.floor(n), 2048);
}

function runNative(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(nativeRunnerPath(), args, {
      cwd: rootDir,
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) return resolve(stdout);
      reject(new Error((stderr || stdout || `native runner exited ${code}`).trim()));
    });
  });
}

export async function nativeChat(config, payload, options = {}) {
  const modelPath = nativeModelPath(config, payload);
  if (!modelPath) throw new Error("Native backend requires a .gguf model path.");
  if (!fsSync.existsSync(nativeRunnerPath())) {
    throw new Error("Native runner not built; run `npm run native:build` first.");
  }

  const tokenizer = await loadQwen3Tokenizer(modelPath);
  const renderedPrompt = renderQwen3Chat(payload.messages || [], {
    tools: payload.tools || [],
    think: payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort)
  });
  const promptTokens = tokenizer.encode(renderedPrompt);
  const tokenFile = path.join(os.tmpdir(), `ironmind-${process.pid}-${Date.now()}.tokens`);
  await fs.writeFile(tokenFile, promptTokens.join(","), "utf8");

  const args = [
    modelPath,
    "--tokens-file", tokenFile,
    "--generate", String(maxCompletionTokens(payload)),
    "--ctx", String(Number(payload.ctx || config.ctx)),
    "--json"
  ];
  if (options.saveKvPath) args.push("--save-kv", options.saveKvPath);

  try {
    const output = await runNative(args);
    const native = JSON.parse(output);
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
