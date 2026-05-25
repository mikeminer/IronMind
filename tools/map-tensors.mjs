#!/usr/bin/env node
import { loadQwenTensorMap, summarizeTensorMap } from "../lib/tensorMap.mjs";

const file = process.argv[2];
const asJson = process.argv.includes("--json");

if (!file) {
  console.error("usage: node tools/map-tensors.mjs <model.gguf> [--json]");
  process.exit(2);
}

const map = await loadQwenTensorMap(file);
const summary = summarizeTensorMap(map);

if (asJson) {
  console.log(JSON.stringify({ summary, hparams: map.hparams, missing: map.missing }, null, 2));
} else {
  console.log("IronMind tensor map");
  for (const [key, value] of Object.entries(summary)) console.log(`  ${key.padEnd(16)} ${value}`);
  if (map.missing.length) {
    for (const name of map.missing.slice(0, 20)) console.log(`  missing          ${name}`);
    if (map.missing.length > 20) console.log(`  missing          ... ${map.missing.length - 20} more`);
  }
}
