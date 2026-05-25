#!/usr/bin/env node
import { inspectGguf, summarizeGguf } from "../lib/gguf.mjs";
import { validateIronMindTarget } from "../lib/target.mjs";

const file = process.argv[2];
const asJson = process.argv.includes("--json");

if (!file) {
  console.error("usage: node tools/inspect-gguf.mjs <model.gguf> [--json]");
  process.exit(2);
}

const info = await inspectGguf(file);
const summary = summarizeGguf(info);
const validation = validateIronMindTarget(info);

if (asJson) {
  console.log(JSON.stringify({ summary, validation }, null, 2));
} else {
  console.log("IronMind GGUF inspector");
  for (const [key, value] of Object.entries(summary)) {
    console.log(`  ${key.padEnd(20)} ${value ?? "-"}`);
  }
  console.log(`  target              ${validation.ok ? "compatible" : "not compatible"}`);
  for (const issue of validation.issues) console.log(`  issue               ${issue}`);
  for (const warning of validation.warnings) console.log(`  warning             ${warning}`);
}
