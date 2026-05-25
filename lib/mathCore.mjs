export function rmsNorm(input, weight, eps = 1e-6) {
  if (input.length !== weight.length) throw new Error("rmsNorm input/weight length mismatch.");
  let sumSq = 0;
  for (const value of input) sumSq += value * value;
  const scale = 1 / Math.sqrt(sumSq / input.length + eps);
  return input.map((value, index) => value * scale * weight[index]);
}

export function silu(x) {
  return x / (1 + Math.exp(-x));
}

export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

export function dot(a, b) {
  if (a.length !== b.length) throw new Error("dot length mismatch.");
  let out = 0;
  for (let i = 0; i < a.length; i += 1) out += a[i] * b[i];
  return out;
}

export function applyRoPE(vector, position, options = {}) {
  const headDim = options.headDim || vector.length;
  const freqBase = options.freqBase || 1000000;
  if (headDim % 2 !== 0) throw new Error("RoPE head dimension must be even.");
  if (vector.length % headDim !== 0) throw new Error("RoPE vector length must be a multiple of headDim.");

  const out = vector.slice();
  for (let offset = 0; offset < out.length; offset += headDim) {
    for (let i = 0; i < headDim; i += 2) {
      const theta = position * Math.pow(freqBase, -i / headDim);
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const x0 = out[offset + i];
      const x1 = out[offset + i + 1];
      out[offset + i] = x0 * cos - x1 * sin;
      out[offset + i + 1] = x0 * sin + x1 * cos;
    }
  }
  return out;
}

export function causalAttention(query, keys, values, options = {}) {
  if (keys.length !== values.length) throw new Error("attention key/value length mismatch.");
  const scale = options.scale ?? (1 / Math.sqrt(query.length));
  const logits = keys.map((key) => dot(query, key) * scale);
  const probs = softmax(logits);
  const out = new Array(values[0]?.length || 0).fill(0);

  for (let row = 0; row < values.length; row += 1) {
    for (let col = 0; col < out.length; col += 1) {
      out[col] += probs[row] * values[row][col];
    }
  }

  return { output: out, probabilities: probs, logits };
}

export function denseMatVec(weightRows, input) {
  return weightRows.map((row) => dot(row, input));
}

export function qwenDenseFfn(input, gateRows, upRows, downRows) {
  const gate = denseMatVec(gateRows, input).map(silu);
  const up = denseMatVec(upRows, input);
  const hidden = gate.map((value, index) => value * up[index]);
  return denseMatVec(downRows, hidden);
}
