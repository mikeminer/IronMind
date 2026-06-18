import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";
import { renderQwen3Chat } from "./qwen3Prompt.mjs";
import { writeIronKv, readIronKv } from "./ironkv.mjs";
import { canonicalizeMessages, canonicalizeTools } from "./toolCalls.mjs";

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

async function statSize(filePath) {
  try {
    return (await fs.stat(filePath)).size;
  } catch {
    return 0;
  }
}

async function listContextRecords(dir) {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const records = [];
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        records.push(...await listContextRecords(fullPath));
      } else if (entry.isFile() && entry.name.endsWith(".ironctx.json")) {
        const stat = await fs.stat(fullPath);
        const ironKvPath = fullPath.replace(/\.ironctx\.json$/, ".ironkv");
        records.push({
          path: fullPath,
          ironKvPath,
          size: stat.size + await statSize(ironKvPath),
          mtimeMs: stat.mtimeMs
        });
      }
    }
    return records;
  } catch {
    return [];
  }
}

export async function pruneContextStore(dir, limitMb) {
  const limitBytes = Number(limitMb || 0) * 1024 * 1024;
  if (!limitBytes) return { removed: 0, bytes: 0 };

  const records = await listContextRecords(dir);
  let total = records.reduce((sum, record) => sum + record.size, 0);
  let removed = 0;
  records.sort((a, b) => a.mtimeMs - b.mtimeMs);

  for (const record of records) {
    if (total <= limitBytes) break;
    await fs.rm(record.path, { force: true });
    await fs.rm(record.ironKvPath, { force: true });
    total -= record.size;
    removed += 1;
  }

  return { removed, bytes: total };
}

export function createContextSnapshot(config, payload) {
  const messages = canonicalizeMessages(payload.messages || []);
  const tools = canonicalizeTools(payload.tools || []);
  const renderedPrompt = renderQwen3Chat(messages, {
    tools,
    think: payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort)
  });
  const model = payload.model || config.model;
  const ctx = Number(payload.ctx || config.ctx);
  const hash = sha1(`${model}\0${ctx}\0${renderedPrompt}`);
  const shard = hash.slice(0, 2);
  const indexPath = path.join(config.kvDiskDir, shard, `${hash}.ironctx.json`);
  const ironKvPath = path.join(config.kvDiskDir, shard, `${hash}.ironkv`);
  const now = new Date().toISOString();
  const renderedPromptBytes = Buffer.byteLength(renderedPrompt, "utf8");
  const estimatedTokens = estimateTokens(renderedPrompt);
  const header = {
    kind: "ironmind.session",
    version: 1,
    model,
    contextSize: ctx,
    hash,
    createdAt: now,
    renderedPromptBytes,
    estimatedTokens,
    promptSha1: sha1(renderedPrompt),
    toolsSha1: sha1(JSON.stringify(tools)),
    nativeKvPayload: false
  };

  return {
    kind: "ironmind.prompt-prefix",
    version: 1,
    model,
    contextSize: ctx,
    hash,
    indexPath,
    ironKvPath,
    header,
    createdAt: now,
    updatedAt: now,
    renderedPromptBytes,
    estimatedTokens,
    nativeKvPayload: false,
    ironKvPayloadBytes: 0,
    renderedPrompt
  };
}

function publicSnapshot(snapshot, overrides = {}) {
  return {
    hash: snapshot.hash,
    path: snapshot.indexPath,
    ironKvPath: snapshot.ironKvPath,
    bytes: snapshot.renderedPromptBytes,
    estimatedTokens: snapshot.estimatedTokens,
    nativeKvPayload: Boolean(overrides.nativeKvPayload),
    ironKvPayloadBytes: Number(overrides.ironKvPayloadBytes || 0)
  };
}

export async function writeContextSnapshot(snapshot, options = {}) {
  const payload = options.payload
    ? (Buffer.isBuffer(options.payload) ? options.payload : Buffer.from(options.payload))
    : Buffer.alloc(0);
  const nativeKvPayload = Boolean(options.nativeKvPayload || payload.length);
  const ironKvHeader = {
    ...snapshot.header,
    ...(options.ironKvHeader || {}),
    nativeKvPayload
  };

  if (!options.skipIronKvWrite) {
    await writeIronKv(snapshot.ironKvPath, ironKvHeader, payload);
  }

  const index = {
    ...snapshot,
    nativeKvPayload,
    ironKvPayloadBytes: Number(options.ironKvPayloadBytes ?? payload.length),
    ironKvPath: snapshot.ironKvPath,
    header: ironKvHeader
  };
  await writeJsonAtomic(snapshot.indexPath, index);
  return publicSnapshot(snapshot, {
    nativeKvPayload,
    ironKvPayloadBytes: index.ironKvPayloadBytes
  });
}

export async function saveContextSnapshot(config, payload, options = {}) {
  const snapshot = createContextSnapshot(config, payload);
  const written = await writeContextSnapshot(snapshot, options);
  await pruneContextStore(config.kvDiskDir, config.kvDiskSpaceMb);
  return written;
}

export async function finalizeNativeContextSnapshot(config, snapshot, details = {}) {
  const kv = await readIronKv(snapshot.ironKvPath);
  const written = await writeContextSnapshot(snapshot, {
    skipIronKvWrite: true,
    nativeKvPayload: true,
    ironKvPayloadBytes: kv.payloadLength,
    ironKvHeader: {
      ...(kv.header || {}),
      native: details.native || true
    }
  });
  await pruneContextStore(config.kvDiskDir, config.kvDiskSpaceMb);
  return written;
}

export async function contextStoreStats(config) {
  const records = await listContextRecords(config.kvDiskDir);
  return {
    dir: config.kvDiskDir,
    limitMb: config.kvDiskSpaceMb,
    files: records.length,
    bytes: records.reduce((sum, record) => sum + record.size, 0)
  };
}
