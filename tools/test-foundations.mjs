#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderQwen3Chat } from "../lib/qwen3Prompt.mjs";
import { writeIronKv, readIronKv, readIronKvPayload } from "../lib/ironkv.mjs";
import { saveContextSnapshot, contextStoreStats } from "../lib/contextStore.mjs";
import { rmsNorm, applyRoPE, softmax, causalAttention } from "../lib/mathCore.mjs";
import { tensorSizeBytes } from "../lib/tensorMap.mjs";
import { canonicalJson, canonicalizeToolCalls, extractToolCallsFromText } from "../lib/toolCalls.mjs";
import { isGgufModel } from "../lib/nativeBackend.mjs";
import {
  aggregateModelOutputs,
  createClinicalTriage,
  scoreImageQuality,
  scoreModelAgreement
} from "../lib/clinicalScoring.mjs";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");

const prompt = renderQwen3Chat([
  { role: "system", content: "You are IronMind." },
  { role: "user", content: "Ciao" }
], { think: false });

assert.equal(prompt, "<|im_start|>system\nYou are IronMind.\n<|im_end|>\n<|im_start|>user\nCiao /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ironmind-"));
const extensionlessGguf = path.join(dir, "ollama-blob");
await fs.writeFile(extensionlessGguf, Buffer.from("GGUF"));
assert.equal(isGgufModel(extensionlessGguf), true);

assert.equal(scoreImageQuality({ score: 1.2 }), 1);
assert.ok(scoreImageQuality({
  resolutionScore: 0.9,
  contrastScore: 0.8,
  noiseScore: 0.7,
  artifactScore: 0.8,
  modalityFitScore: 0.9
}) > 0.75);
assert.ok(scoreModelAgreement([{ riskScore: 0.5 }, { riskScore: 0.52 }, { riskScore: 0.48 }]) > 0.95);
assert.ok(scoreModelAgreement([{ riskScore: 0.1 }, { riskScore: 0.9 }]) < 0.3);

const aggregate = aggregateModelOutputs([
  { modelId: "a", riskScore: 0.8, confidenceScore: 0.7, uncertaintyScore: 0.2, weight: 2 },
  { modelId: "b", riskScore: 0.6, confidenceScore: 0.8, uncertaintyScore: 0.3, weight: 1 }
]);
assert.ok(aggregate.riskScore > 0.72 && aggregate.riskScore < 0.74);
assert.equal(aggregate.modelCount, 2);

const urgentTriage = createClinicalTriage({
  modality: "xray",
  bodyRegion: "chest",
  imageQuality: { score: 0.92 },
  explainability: { score: 0.7, refs: ["heatmap://case-1"] },
  modelOutputs: [
    { modelId: "cxr-risk-a", riskScore: 0.93, confidenceScore: 0.82, uncertaintyScore: 0.18 },
    { modelId: "cxr-risk-b", riskScore: 0.88, confidenceScore: 0.78, uncertaintyScore: 0.22 }
  ]
});
assert.equal(urgentTriage.humanReviewRequired, true);
assert.equal(urgentTriage.reviewPriority, "urgent");
assert.equal(urgentTriage.evidence.modelIds.length, 2);

const lowQualityTriage = createClinicalTriage({
  imageQuality: { score: 0.2 },
  modelOutputs: [{ riskScore: 0.2, confidenceScore: 0.9, uncertaintyScore: 0.1 }]
});
assert.equal(lowQualityTriage.recommendation, "repeat_or_manual_image_review");

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
const kvPayload = await readIronKvPayload(kvPath);
assert.deepEqual([...kvPayload.payload], [1, 2, 3, 4]);

assert.equal(canonicalJson({ b: 2, a: { d: 4, c: 3 } }), '{"a":{"c":3,"d":4},"b":2}');
const toolCalls = canonicalizeToolCalls([
  { function: { name: "search", arguments: "{\"b\":2,\"a\":1}" } }
]);
assert.equal(toolCalls[0].function.arguments, '{"a":1,"b":2}');
const generatedTool = extractToolCallsFromText('ok\n<tool_call>\n{"arguments":{"z":1,"a":2},"name":"lookup"}\n</tool_call>');
assert.equal(generatedTool.content, "ok");
assert.equal(generatedTool.tool_calls[0].function.arguments, '{"a":2,"z":1}');

const toolPrompt = renderQwen3Chat([
  { role: "user", content: "Find it" },
  { role: "assistant", content: "", tool_calls: [{ function: { name: "lookup", arguments: { b: 2, a: 1 } } }] },
  { role: "tool", tool_call_id: "call_1", name: "lookup", content: "done" }
], {
  tools: [{ function: { name: "lookup", parameters: { type: "object", properties: { b: { type: "number" }, a: { type: "number" } } } } }],
  think: false
});
assert.ok(toolPrompt.includes('{"arguments":{"a":1,"b":2},"name":"lookup"}'));
assert.ok(toolPrompt.includes('{"content":"done","name":"lookup","tool_call_id":"call_1"}'));

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
assert.ok(context.ironKvPath.endsWith(".ironkv"));
const contextKv = await readIronKv(context.ironKvPath);
assert.equal(contextKv.header.kind, "ironmind.session");
assert.equal(contextKv.payloadLength, 0);

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

const normed = rmsNorm([3, 4], [1, 1], 0);
assert.ok(Math.abs(normed[0] - 0.8485281374) < 1e-6);
assert.ok(Math.abs(normed[1] - 1.1313708499) < 1e-6);

const rotated = applyRoPE([1, 0], Math.PI / 2, { headDim: 2, freqBase: 10000 });
assert.ok(Math.abs(rotated[0]) < 1e-12);
assert.ok(Math.abs(rotated[1] - 1) < 1e-12);

const probs = softmax([1, 2, 3]);
assert.ok(Math.abs(probs.reduce((acc, value) => acc + value, 0) - 1) < 1e-12);

const attn = causalAttention([1, 0], [[1, 0], [0, 1]], [[10, 0], [0, 10]], { scale: 100 });
assert.ok(attn.output[0] > 9.999);
assert.ok(attn.output[1] < 0.001);
assert.equal(tensorSizeBytes({ type: 0, dims: ["2", "3"] }), 24);
assert.equal(tensorSizeBytes({ type: 12, dims: ["256", "2"] }), 288);

await fs.rm(dir, { recursive: true, force: true });
console.log("foundation tests passed");
