#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderQwen3Chat } from "../lib/qwen3Prompt.mjs";
import { writeIronKv, readIronKv } from "../lib/ironkv.mjs";
import { saveContextSnapshot, contextStoreStats } from "../lib/contextStore.mjs";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");

const prompt = renderQwen3Chat([
  { role: "system", content: "You are IronMind." },
  { role: "user", content: "Ciao" }
], { think: false });

assert.equal(prompt, "<|im_start|>system\nYou are IronMind.\n<|im_end|>\n<|im_start|>user\nCiao /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ironmind-"));
const kvPath = path.join(dir, "test.ironkv");
await writeIronKv(kvPath, {
  modelFingerprint: "abc",
  tokenCount: 3,
  contextSize: 32768
}, Buffer.from([1, 2, 3, 4]));

const kv = await readIronKv(kvPath);
assert.equal(kv.header.modelFingerprint, "abc");
assert.equal(kv.header.tokenCount, 3);
assert.equal(kv.payloadLength, 4);

const context = await saveContextSnapshot({
  model: "qwen3-coder:30b",
  ctx: 131072,
  kvDiskDir: path.join(dir, "kvcache"),
  kvDiskSpaceMb: 16
}, {
  messages: [{ role: "user", content: "Persist this prefix." }],
  think: false
});
assert.equal(context.nativeKvPayload, false);
assert.ok(context.estimatedTokens > 0);

const stats = await contextStoreStats({
  kvDiskDir: path.join(dir, "kvcache"),
  kvDiskSpaceMb: 16
});
assert.equal(stats.files, 1);

const evalSuite = JSON.parse(await fs.readFile(path.join(rootDir, "eval", "ironmind-100.json"), "utf8"));
assert.equal(evalSuite.items.length, 100);
assert.equal(evalSuite.items.filter((item) => item.category === "physics").length, 34);
assert.equal(evalSuite.items.filter((item) => item.category === "mathematics").length, 33);
assert.equal(evalSuite.items.filter((item) => item.category === "security").length, 33);

await fs.rm(dir, { recursive: true, force: true });
console.log("foundation tests passed");
