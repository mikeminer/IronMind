import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { renderQwen3Chat } from "./qwen3Prompt.mjs";
import { loadQwen3Tokenizer } from "./tokenizer.mjs";
import { extractToolCallsFromText } from "./toolCalls.mjs";
import { applyCpuPerformanceOptions } from "./cpuPerformance.mjs";
import { addQwen3ThinkingDirective, stripQwenThinking } from "./qwenThinking.mjs";

function platformExe(name) {
  return process.platform === "win32" ? `${name}.exe` : name;
}

export function ikWorkerPath(config = {}) {
  if (config.ikLlamaWorker) return path.resolve(config.ikLlamaWorker);
  if (config.ikLlamaServer) return path.join(path.dirname(path.resolve(config.ikLlamaServer)), platformExe("llama-cli"));
  return "";
}

export function isIkWorkerBackend(config = {}) {
  return config.backend === "ik_worker";
}

export function ikWorkerBackendDescription(config = {}) {
  const worker = ikWorkerPath(config);
  return `ik_worker:${worker || "unconfigured"}`;
}

function maxCompletionTokens(payload, fallback) {
  const requested = payload.max_completion_tokens ?? payload.max_tokens ?? payload.options?.num_predict ?? fallback;
  const n = Number(requested);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(Math.floor(n), 2048);
}

function runWorker(exe, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, {
      cwd: path.dirname(exe),
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timeoutMs = Number(options.timeoutMs || 300000);
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
      if (timedOut) return reject(new Error(`ik_worker timed out after ${timeoutMs}ms`));
      if (code === 0) return resolve({ stdout, stderr });
      reject(new Error((stderr || stdout || `ik_worker exited ${code}`).trim()));
    });
  });
}

function cleanWorkerOutput(text) {
  return stripQwenThinking(String(text || ""))
    .replace(/<\|im_end\|>/g, "")
    .replace(/<\|endoftext\|>/g, "")
    .replace(/<\/s>/g, "")
    .trimStart();
}

export async function ikWorkerChat(config, payload = {}, options = {}) {
  const worker = ikWorkerPath(config);
  if (!worker || !fsSync.existsSync(worker)) {
    throw new Error(`ik_worker executable not found: ${worker || "not configured"}`);
  }
  if (!config.ikLlamaModel || !fsSync.existsSync(config.ikLlamaModel)) {
    throw new Error(`ik_worker model not found: ${config.ikLlamaModel || "not configured"}`);
  }

  const performance = applyCpuPerformanceOptions(config, payload);
  const completionTokens = maxCompletionTokens(payload, performance.performance.maxTokens);
  const preparedPayload = addQwen3ThinkingDirective(payload, config.model, { force: true });
  const renderedPrompt = renderQwen3Chat(preparedPayload.messages || [], {
    tools: preparedPayload.tools || [],
    think: preparedPayload.think ?? Boolean(preparedPayload.reasoning || preparedPayload.reasoning_effort)
  });

  const tmpPrompt = path.join(os.tmpdir(), `ironmind-ik-worker-${process.pid}-${Date.now()}.prompt.txt`);
  await fs.writeFile(tmpPrompt, renderedPrompt, "utf8");

  const promptCache = options.promptCachePath || (
    options.contextSnapshot?.ironKvPath ? `${options.contextSnapshot.ironKvPath}.llamacache` : ""
  );
  if (promptCache) await fs.mkdir(path.dirname(promptCache), { recursive: true });

  const args = [
    "--model", config.ikLlamaModel,
    "--file", tmpPrompt,
    "--no-display-prompt",
    "--predict", String(completionTokens),
    "--ctx-size", String(performance.performance.effectiveContext || performance.performance.interactiveContext || config.ctx),
    "--threads", String(performance.performance.threads),
    "--threads-batch", String(performance.performance.threads),
    "--batch-size", String(performance.performance.batch),
    "--temp", String(performance.options.temperature ?? 0.15),
    "--top-p", String(performance.options.top_p ?? 0.85),
    "--repeat-penalty", String(performance.options.repeat_penalty ?? 1.05),
    "--n-gpu-layers", "0",
    "--no-warmup"
  ];
  if (promptCache) args.push("--prompt-cache", promptCache, "--prompt-cache-all");

  try {
    const started = Date.now();
    const { stdout } = await runWorker(worker, args, { timeoutMs: config.ikLlamaWorkerTimeoutMs || config.nativeTimeoutMs });
    const content = cleanWorkerOutput(stdout);
    const parsed = extractToolCallsFromText(content);
    const tokenizer = await loadQwen3Tokenizer(config.ikLlamaModel);
    const promptTokens = tokenizer.encode(renderedPrompt).length;
    const outputTokens = content ? tokenizer.encode(content).length : 0;
    const message = {
      role: "assistant",
      content: parsed.content
    };
    if (parsed.tool_calls.length) message.tool_calls = parsed.tool_calls;

    return {
      model: payload.model || config.model,
      message,
      prompt_eval_count: promptTokens,
      eval_count: outputTokens,
      total_duration_ms: Date.now() - started,
      done: true,
      native: {
        transport: "ik_worker",
        worker,
        modelPath: config.ikLlamaModel,
        promptCachePath: promptCache || null,
        contextSnapshot: options.contextSnapshot || null
      }
    };
  } finally {
    await fs.rm(tmpPrompt, { force: true });
  }
}
