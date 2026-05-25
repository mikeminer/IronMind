import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";
import { renderQwen3Chat } from "./qwen3Prompt.mjs";

export function defaultContextDir() {
  return path.join(os.homedir(), ".ironmind", "kvcache");
}

function sha1(input) {
  return crypto.createHash("sha1").update(input).digest("hex");
}

function estimateTokens(text) {
  return Math.max(1, Math.ceil(Buffer.byteLength(text, "utf8") / 4));
}

async function writeJsonAtomic(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${process.pid}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value, null, 2), "utf8");
  await fs.rename(tmp, filePath);
}

async function listCacheFiles(dir) {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const files = [];
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...await listCacheFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith(".ironctx.json")) {
        const stat = await fs.stat(fullPath);
        files.push({ path: fullPath, size: stat.size, mtimeMs: stat.mtimeMs });
      }
    }
    return files;
  } catch {
    return [];
  }
}

export async function pruneContextStore(dir, limitMb) {
  const limitBytes = Number(limitMb || 0) * 1024 * 1024;
  if (!limitBytes) return { removed: 0, bytes: 0 };

  const files = await listCacheFiles(dir);
  let total = files.reduce((sum, file) => sum + file.size, 0);
  let removed = 0;
  files.sort((a, b) => a.mtimeMs - b.mtimeMs);

  for (const file of files) {
    if (total <= limitBytes) break;
    await fs.rm(file.path, { force: true });
    total -= file.size;
    removed += 1;
  }

  return { removed, bytes: total };
}

export async function saveContextSnapshot(config, payload) {
  const messages = payload.messages || [];
  const renderedPrompt = renderQwen3Chat(messages, {
    tools: payload.tools,
    think: payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort)
  });
  const model = payload.model || config.model;
  const ctx = Number(payload.ctx || config.ctx);
  const hash = sha1(`${model}\0${ctx}\0${renderedPrompt}`);
  const shard = hash.slice(0, 2);
  const filePath = path.join(config.kvDiskDir, shard, `${hash}.ironctx.json`);
  const now = new Date().toISOString();
  const snapshot = {
    kind: "ironmind.prompt-prefix",
    version: 1,
    model,
    contextSize: ctx,
    hash,
    createdAt: now,
    updatedAt: now,
    renderedPromptBytes: Buffer.byteLength(renderedPrompt, "utf8"),
    estimatedTokens: estimateTokens(renderedPrompt),
    nativeKvPayload: false,
    renderedPrompt
  };

  await writeJsonAtomic(filePath, snapshot);
  await pruneContextStore(config.kvDiskDir, config.kvDiskSpaceMb);
  return {
    hash,
    path: filePath,
    bytes: snapshot.renderedPromptBytes,
    estimatedTokens: snapshot.estimatedTokens,
    nativeKvPayload: false
  };
}

export async function contextStoreStats(config) {
  const files = await listCacheFiles(config.kvDiskDir);
  return {
    dir: config.kvDiskDir,
    limitMb: config.kvDiskSpaceMb,
    files: files.length,
    bytes: files.reduce((sum, file) => sum + file.size, 0)
  };
}
