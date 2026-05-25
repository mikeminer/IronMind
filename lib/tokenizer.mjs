import { inspectGguf } from "./gguf.mjs";

const QWEN2_PATTERN = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const encoder = new Map();
  const decoder = new Map();
  for (let i = 0; i < bs.length; i += 1) {
    const ch = String.fromCodePoint(cs[i]);
    encoder.set(bs[i], ch);
    decoder.set(ch, bs[i]);
  }
  return { encoder, decoder };
}

const BYTE_CODEC = bytesToUnicode();
const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: false });

function encodeBytes(text) {
  let out = "";
  for (const byte of TEXT_ENCODER.encode(text)) out += BYTE_CODEC.encoder.get(byte);
  return out;
}

function splitQwen2(text) {
  const out = [];
  let cursor = 0;
  for (const match of text.matchAll(QWEN2_PATTERN)) {
    if (match.index > cursor) out.push(text.slice(cursor, match.index));
    out.push(match[0]);
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) out.push(text.slice(cursor));
  return out.filter(Boolean);
}

function mergeKey(a, b) {
  return `${a}\u0000${b}`;
}

function parseMerge(merge) {
  const sep = merge.indexOf(" ");
  if (sep <= 0) return null;
  return [merge.slice(0, sep), merge.slice(sep + 1)];
}

function bestPair(parts, ranks) {
  let best = null;
  let bestRank = Infinity;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const rank = ranks.get(mergeKey(parts[i], parts[i + 1]));
    if (rank !== undefined && rank < bestRank) {
      best = [parts[i], parts[i + 1]];
      bestRank = rank;
    }
  }
  return best;
}

function applyBpe(piece, ranks) {
  if (!piece) return [];
  let parts = Array.from(piece);
  if (parts.length < 2) return parts;

  while (parts.length > 1) {
    const pair = bestPair(parts, ranks);
    if (!pair) break;

    const next = [];
    for (let i = 0; i < parts.length; i += 1) {
      if (i < parts.length - 1 && parts[i] === pair[0] && parts[i + 1] === pair[1]) {
        next.push(parts[i] + parts[i + 1]);
        i += 1;
      } else {
        next.push(parts[i]);
      }
    }
    parts = next;
  }

  return parts;
}

function isSpecialToken(token, type) {
  return (typeof token === "string" && token.startsWith("<|") && token.endsWith("|>")) || type === 3 || type === 4;
}

export class Qwen3Tokenizer {
  constructor({ tokens, tokenTypes, merges, metadata }) {
    this.tokens = tokens;
    this.tokenTypes = tokenTypes || [];
    this.metadata = metadata;
    this.tokenToId = new Map(tokens.map((token, id) => [token, id]));
    this.idToToken = tokens;
    this.mergeRanks = new Map();

    for (let i = 0; i < merges.length; i += 1) {
      const pair = parseMerge(merges[i]);
      if (pair) this.mergeRanks.set(mergeKey(pair[0], pair[1]), i);
    }

    this.specialTokenToId = new Map();
    for (let id = 0; id < tokens.length; id += 1) {
      if (isSpecialToken(tokens[id], this.tokenTypes[id])) this.specialTokenToId.set(tokens[id], id);
    }
    this.specialTokens = [...this.specialTokenToId.keys()].sort((a, b) => b.length - a.length);
  }

  encode(text, options = {}) {
    const allowSpecial = options.special !== false;
    const ids = [];
    let pos = 0;

    while (pos < text.length) {
      let matchedSpecial = null;
      if (allowSpecial && text[pos] === "<") {
        matchedSpecial = this.specialTokens.find((token) => text.startsWith(token, pos));
      }

      if (matchedSpecial) {
        ids.push(this.specialTokenToId.get(matchedSpecial));
        pos += matchedSpecial.length;
        continue;
      }

      const nextSpecialPos = allowSpecial
        ? this.specialTokens.reduce((best, token) => {
            const index = text.indexOf(token, pos + 1);
            return index >= 0 && index < best ? index : best;
          }, text.length)
        : text.length;
      const chunk = text.slice(pos, nextSpecialPos);
      ids.push(...this.encodeOrdinary(chunk));
      pos = nextSpecialPos;
    }

    return ids;
  }

  encodeOrdinary(text) {
    const ids = [];
    for (const part of splitQwen2(text)) {
      const encoded = encodeBytes(part);
      const pieces = applyBpe(encoded, this.mergeRanks);
      for (const piece of pieces) {
        const id = this.tokenToId.get(piece);
        if (id !== undefined) ids.push(id);
        else {
          for (const ch of Array.from(piece)) {
            const fallback = this.tokenToId.get(ch);
            if (fallback === undefined) throw new Error(`Tokenizer cannot map byte piece ${JSON.stringify(ch)}.`);
            ids.push(fallback);
          }
        }
      }
    }
    return ids;
  }

  decode(ids) {
    const bytes = [];
    let out = "";
    const flush = () => {
      if (bytes.length) {
        out += TEXT_DECODER.decode(Uint8Array.from(bytes));
        bytes.length = 0;
      }
    };

    for (const id of ids) {
      const token = this.idToToken[id];
      if (token === undefined) throw new Error(`Unknown token id ${id}.`);
      if (this.specialTokenToId.has(token)) {
        flush();
        out += token;
        continue;
      }
      for (const ch of Array.from(token)) {
        const byte = BYTE_CODEC.decoder.get(ch);
        if (byte === undefined) {
          flush();
          out += ch;
        } else {
          bytes.push(byte);
        }
      }
    }

    flush();
    return out;
  }

  summary() {
    return {
      model: this.metadata["tokenizer.ggml.model"]?.value,
      pre: this.metadata["tokenizer.ggml.pre"]?.value,
      vocabSize: this.tokens.length,
      merges: this.mergeRanks.size,
      specialTokens: this.specialTokenToId.size,
      bosTokenId: this.metadata["tokenizer.ggml.bos_token_id"]?.value,
      eosTokenId: this.metadata["tokenizer.ggml.eos_token_id"]?.value,
      padTokenId: this.metadata["tokenizer.ggml.padding_token_id"]?.value
    };
  }
}

export async function loadQwen3Tokenizer(filePath) {
  const info = await inspectGguf(filePath, {
    keepArrays: [
      "tokenizer.ggml.tokens",
      "tokenizer.ggml.token_type",
      "tokenizer.ggml.merges"
    ],
    tensorLimit: 0
  });

  const model = info.metadata["tokenizer.ggml.model"]?.value;
  const pre = info.metadata["tokenizer.ggml.pre"]?.value;
  if (model !== "gpt2" || pre !== "qwen2") {
    throw new Error(`Unsupported tokenizer model/pre ${model}/${pre}; IronMind currently supports Qwen3 gpt2/qwen2.`);
  }

  return new Qwen3Tokenizer({
    tokens: info.metadata["tokenizer.ggml.tokens"].values,
    tokenTypes: info.metadata["tokenizer.ggml.token_type"].values,
    merges: info.metadata["tokenizer.ggml.merges"].values,
    metadata: info.metadata
  });
}
