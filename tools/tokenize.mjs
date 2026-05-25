#!/usr/bin/env node
import { loadQwen3Tokenizer } from "../lib/tokenizer.mjs";

const file = process.argv[2];
const text = process.argv.slice(3).join(" ");

if (!file || !text) {
  console.error("usage: node tools/tokenize.mjs <model.gguf> <text>");
  process.exit(2);
}

const tokenizer = await loadQwen3Tokenizer(file);
const ids = tokenizer.encode(text);
console.log(JSON.stringify({
  summary: tokenizer.summary(),
  text,
  ids,
  count: ids.length,
  decoded: tokenizer.decode(ids)
}, null, 2));
