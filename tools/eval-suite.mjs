#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");
const suitePath = path.join(rootDir, "eval", "ironmind-100.json");

function parseArgs(argv) {
  const args = [...argv];
  const command = args[0] && !args[0].startsWith("-") ? args.shift() : "stats";
  const options = {
    command,
    baseUrl: process.env.IRONMIND_EVAL_BASE_URL || "http://127.0.0.1:4141/v1",
    model: process.env.IRONMIND_EVAL_MODEL || process.env.IRONMIND_MODEL || "qwen3-coder:30b",
    limit: null,
    category: null,
    out: null
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    if (arg === "--base-url" && next) options.baseUrl = next.replace(/\/+$/, ""), i += 1;
    else if (arg === "--model" && next) options.model = next, i += 1;
    else if (arg === "--limit" && next) options.limit = Number(next), i += 1;
    else if (arg === "--category" && next) options.category = next, i += 1;
    else if (arg === "--out" && next) options.out = next, i += 1;
    else if (arg === "--help" || arg === "-h") options.command = "help";
  }

  return options;
}

async function loadSuite() {
  return JSON.parse(await fs.readFile(suitePath, "utf8"));
}

function selectedItems(suite, options) {
  let items = suite.items;
  if (options.category) items = items.filter((item) => item.category === options.category);
  if (options.limit !== null) items = items.slice(0, options.limit);
  return items;
}

function usage() {
  console.log(`IronMind Eval 100

Usage:
  node tools/eval-suite.mjs stats
  node tools/eval-suite.mjs list [--category physics|mathematics|security]
  node tools/eval-suite.mjs run [--model qwen3:14b] [--limit 10] [--category security] [--out results.json]

Environment:
  IRONMIND_EVAL_BASE_URL=http://127.0.0.1:4141/v1
  IRONMIND_EVAL_MODEL=qwen3-coder:30b
`);
}

function stats(items) {
  const counts = new Map();
  for (const item of items) counts.set(item.category, (counts.get(item.category) || 0) + 1);
  return Object.fromEntries([...counts.entries()].sort());
}

function renderQuestion(item) {
  return [
    "Answer this IronMind evaluation question.",
    "Return exactly one uppercase letter: A, B, C, or D.",
    "",
    `ID: ${item.id}`,
    `Category: ${item.category}`,
    "",
    item.question,
    "",
    `A) ${item.choices.A}`,
    `B) ${item.choices.B}`,
    `C) ${item.choices.C}`,
    `D) ${item.choices.D}`
  ].join("\n");
}

function extractChoice(text) {
  const normalized = String(text || "").toUpperCase();
  const direct = normalized.trim().match(/^[ABCD]$/);
  if (direct) return direct[0];
  const answer = normalized.match(/\b(?:ANSWER|RISPOSTA)\s*[:\-]?\s*([ABCD])\b/);
  if (answer) return answer[1];
  const any = normalized.match(/\b([ABCD])\b/);
  return any ? any[1] : null;
}

async function askModel(item, options) {
  const response = await fetch(`${options.baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: options.model,
      stream: false,
      temperature: 0,
      max_tokens: 8,
      think: false,
      messages: [
        {
          role: "system",
          content: "You are a strict multiple-choice evaluator. Reply with only one uppercase letter."
        },
        { role: "user", content: renderQuestion(item) }
      ]
    })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content || "";
  return {
    text,
    choice: extractChoice(text)
  };
}

async function runSuite(suite, options) {
  const items = selectedItems(suite, options);
  const results = [];
  const totals = { correct: 0, wrong: 0, errors: 0 };
  const byCategory = {};

  for (const item of items) {
    if (!byCategory[item.category]) byCategory[item.category] = { correct: 0, total: 0 };
    byCategory[item.category].total += 1;

    try {
      const model = await askModel(item, options);
      const correct = model.choice === item.answer;
      if (correct) {
        totals.correct += 1;
        byCategory[item.category].correct += 1;
      } else {
        totals.wrong += 1;
      }
      results.push({ id: item.id, category: item.category, answer: item.answer, modelChoice: model.choice, correct, text: model.text });
      console.log(`${correct ? "PASS" : "FAIL"} ${item.id} expected=${item.answer} got=${model.choice || "-"} ${model.text.replace(/\s+/g, " ").trim()}`);
    } catch (error) {
      totals.errors += 1;
      results.push({ id: item.id, category: item.category, answer: item.answer, error: error.message });
      console.log(`ERR  ${item.id} ${error.message}`);
    }
  }

  const report = {
    suite: suite.name,
    model: options.model,
    baseUrl: options.baseUrl,
    total: items.length,
    correct: totals.correct,
    wrong: totals.wrong,
    errors: totals.errors,
    score: items.length ? totals.correct / items.length : 0,
    byCategory,
    results
  };

  console.log("");
  console.log(`Score: ${report.correct}/${report.total} (${(report.score * 100).toFixed(1)}%) errors=${report.errors}`);
  for (const [category, row] of Object.entries(byCategory)) {
    console.log(`  ${category}: ${row.correct}/${row.total}`);
  }

  if (options.out) {
    await fs.writeFile(options.out, JSON.stringify(report, null, 2), "utf8");
    console.log(`Report written to ${options.out}`);
  }
}

function validateSuite(suite) {
  const ids = new Set();
  if (suite.items.length !== 100) throw new Error(`Expected 100 items, got ${suite.items.length}.`);
  for (const item of suite.items) {
    if (ids.has(item.id)) throw new Error(`Duplicate eval id ${item.id}.`);
    ids.add(item.id);
    for (const key of ["A", "B", "C", "D"]) {
      if (!item.choices?.[key]) throw new Error(`${item.id} is missing choice ${key}.`);
    }
    if (!["A", "B", "C", "D"].includes(item.answer)) throw new Error(`${item.id} has invalid answer.`);
  }
  const counts = stats(suite.items);
  if (counts.physics !== 34 || counts.mathematics !== 33 || counts.security !== 33) {
    throw new Error(`Unexpected category counts: ${JSON.stringify(counts)}`);
  }
}

const options = parseArgs(process.argv.slice(2));
const suite = await loadSuite();
validateSuite(suite);

if (options.command === "help") usage();
else if (options.command === "stats") {
  console.log(`${suite.name} v${suite.version}`);
  console.log(JSON.stringify({ total: suite.items.length, categories: stats(suite.items) }, null, 2));
} else if (options.command === "list") {
  for (const item of selectedItems(suite, options)) {
    console.log(`${item.id} ${item.category} answer=${item.answer} ${item.question}`);
  }
} else if (options.command === "run") {
  await runSuite(suite, options);
} else {
  usage();
  process.exitCode = 2;
}
