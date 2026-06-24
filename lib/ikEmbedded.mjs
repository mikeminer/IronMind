import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { renderQwen3Chat } from "./qwen3Prompt.mjs";
import { loadQwen3Tokenizer } from "./tokenizer.mjs";
import { extractToolCallsFromText } from "./toolCalls.mjs";
import { applyCpuPerformanceOptions } from "./cpuPerformance.mjs";
import { addQwen3ThinkingDirective, stripQwenThinking } from "./qwenThinking.mjs";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");

function platformExe(name) {
  return process.platform === "win32" ? `${name}.exe` : name;
}

export function ikEmbeddedPath(config = {}) {
  if (config.ikEmbeddedRunner) return path.resolve(config.ikEmbeddedRunner);
  const release = path.join(rootDir, "build-ik", "Release", platformExe("ironmind-ik-native"));
  if (fsSync.existsSync(release)) return release;
  return path.join(rootDir, "build-ik", platformExe("ironmind-ik-native"));
}

export function isIkEmbeddedBackend(config = {}) {
  return config.backend === "ik_embedded";
}

export function ikEmbeddedBackendDescription(config = {}) {
  return `ik_embedded:${ikEmbeddedPath(config)}`;
}

function runtimeLibraryEnv(exe) {
  const libraryPathKey = process.platform === "win32"
    ? "PATH"
    : process.platform === "darwin"
      ? "DYLD_LIBRARY_PATH"
      : "LD_LIBRARY_PATH";
  const candidates = [
    path.dirname(exe),
    path.join(rootDir, "build-ik", "bin", "Release"),
    path.join(rootDir, "build-ik", "bin")
  ].filter((dir, index, all) => fsSync.existsSync(dir) && all.indexOf(dir) === index);
  return {
    ...process.env,
    [libraryPathKey]: [
      ...candidates,
      process.env[libraryPathKey] || ""
    ].filter(Boolean).join(path.delimiter)
  };
}

function maxCompletionTokens(payload, fallback) {
  const requested = payload.max_completion_tokens ?? payload.max_tokens ?? payload.options?.num_predict ?? fallback;
  const n = Number(requested);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(Math.floor(n), 2048);
}

function runEmbedded(exe, args, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, {
      cwd: rootDir,
      env: runtimeLibraryEnv(exe),
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
      if (timedOut) return reject(new Error(`ik_embedded timed out after ${timeoutMs}ms`));
      if (code === 0) return resolve(stdout);
      reject(new Error((stderr || stdout || `ik_embedded exited ${code}`).trim()));
    });
  });
}

export async function ikEmbeddedChat(config, payload = {}, options = {}) {
  const runner = ikEmbeddedPath(config);
  if (!runner || !fsSync.existsSync(runner)) {
    throw new Error(`ik_embedded runner not built: ${runner}. Run npm run native:ik:build.`);
  }
  if (!config.ikLlamaModel || !fsSync.existsSync(config.ikLlamaModel)) {
    throw new Error(`ik_embedded model not found: ${config.ikLlamaModel || "not configured"}`);
  }

  const performance = applyCpuPerformanceOptions(config, payload);
  const completionTokens = maxCompletionTokens(payload, performance.performance.maxTokens);
  const preparedPayload = addQwen3ThinkingDirective(payload, config.model, { force: true });
  const renderedPrompt = renderQwen3Chat(preparedPayload.messages || [], {
    tools: preparedPayload.tools || [],
    think: preparedPayload.think ?? Boolean(preparedPayload.reasoning || preparedPayload.reasoning_effort)
  });
  const tmpPrompt = path.join(os.tmpdir(), `ironmind-ik-embedded-${process.pid}-${Date.now()}.prompt.txt`);
  await fs.writeFile(tmpPrompt, renderedPrompt, "utf8");

  const sessionPath = options.contextSnapshot?.ironKvPath
    ? `${options.contextSnapshot.ironKvPath}.iksession`
    : "";
  if (sessionPath) await fs.mkdir(path.dirname(sessionPath), { recursive: true });

  const args = [
    "--model", config.ikLlamaModel,
    "--file", tmpPrompt,
    "--json",
    "--predict", String(completionTokens),
    "--ctx-size", String(performance.performance.effectiveContext || performance.performance.interactiveContext || config.ctx),
    "--batch-size", String(performance.performance.batch),
    "--threads", String(performance.performance.threads)
  ];
  if (sessionPath) args.push("--save-session", sessionPath);

  try {
    const raw = await runEmbedded(runner, args, config.ikEmbeddedTimeoutMs || config.nativeTimeoutMs);
    const parsed = JSON.parse(raw);
    const content = stripQwenThinking(parsed.text || "").trimStart();
    const toolParsed = extractToolCallsFromText(content);
    const tokenizer = await loadQwen3Tokenizer(config.ikLlamaModel);
    const message = {
      role: "assistant",
      content: toolParsed.content
    };
    if (toolParsed.tool_calls.length) message.tool_calls = toolParsed.tool_calls;
    return {
      model: payload.model || config.model,
      message,
      prompt_eval_count: parsed.promptTokens || tokenizer.encode(renderedPrompt).length,
      eval_count: parsed.generatedTokens || (content ? tokenizer.encode(content).length : 0),
      total_duration_ms: parsed.elapsedMs || null,
      done: true,
      native: {
        transport: "ik_embedded",
        runner,
        modelPath: config.ikLlamaModel,
        sessionPath: sessionPath || null
      }
    };
  } finally {
    await fs.rm(tmpPrompt, { force: true });
  }
}
