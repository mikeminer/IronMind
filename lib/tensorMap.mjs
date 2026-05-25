import { inspectGguf } from "./gguf.mjs";
import { validateIronMindTarget } from "./target.mjs";

export const GGML_TYPE_INFO = new Map([
  [0, { name: "F32", blockSize: 1, typeSize: 4, quantized: false }],
  [1, { name: "F16", blockSize: 1, typeSize: 2, quantized: false }],
  [2, { name: "Q4_0", blockSize: 32, typeSize: 18, quantized: true }],
  [3, { name: "Q4_1", blockSize: 32, typeSize: 20, quantized: true }],
  [6, { name: "Q5_0", blockSize: 32, typeSize: 22, quantized: true }],
  [7, { name: "Q5_1", blockSize: 32, typeSize: 24, quantized: true }],
  [8, { name: "Q8_0", blockSize: 32, typeSize: 34, quantized: true }],
  [9, { name: "Q8_1", blockSize: 32, typeSize: 40, quantized: true }],
  [10, { name: "Q2_K", blockSize: 256, typeSize: 84, quantized: true }],
  [11, { name: "Q3_K", blockSize: 256, typeSize: 110, quantized: true }],
  [12, { name: "Q4_K", blockSize: 256, typeSize: 144, quantized: true }],
  [13, { name: "Q5_K", blockSize: 256, typeSize: 176, quantized: true }],
  [14, { name: "Q6_K", blockSize: 256, typeSize: 210, quantized: true }],
  [15, { name: "Q8_K", blockSize: 256, typeSize: 292, quantized: true }],
  [24, { name: "I8", blockSize: 1, typeSize: 1, quantized: false }],
  [25, { name: "I16", blockSize: 1, typeSize: 2, quantized: false }],
  [26, { name: "I32", blockSize: 1, typeSize: 4, quantized: false }],
  [27, { name: "I64", blockSize: 1, typeSize: 8, quantized: false }],
  [28, { name: "F64", blockSize: 1, typeSize: 8, quantized: false }],
  [30, { name: "BF16", blockSize: 1, typeSize: 2, quantized: false }]
]);

function getValue(metadata, key, fallback = 0) {
  return Number(metadata[key]?.value ?? metadata[key]?.length ?? fallback);
}

function dimsToNumbers(tensor) {
  return tensor.dims.map((dim) => Number(dim));
}

export function tensorSizeBytes(tensor) {
  const type = GGML_TYPE_INFO.get(tensor.type);
  if (!type) return null;
  const dims = dimsToNumbers(tensor);
  if (!dims.length) return 0;
  const rows = dims.slice(1).reduce((acc, dim) => acc * dim, 1);
  const rowBlocks = Math.ceil(dims[0] / type.blockSize);
  return rowBlocks * type.typeSize * rows;
}

function enrichTensor(info, tensor, role) {
  if (!tensor) return null;
  const type = GGML_TYPE_INFO.get(tensor.type);
  const sizeBytes = tensorSizeBytes(tensor);
  return {
    role,
    name: tensor.name,
    dims: dimsToNumbers(tensor),
    type: tensor.type,
    typeName: type?.name || `GGML_${tensor.type}`,
    quantized: Boolean(type?.quantized),
    relativeOffset: Number(tensor.offset),
    absoluteOffset: info.dataOffset + Number(tensor.offset),
    sizeBytes
  };
}

function layerTensor(byName, info, layer, suffix, role) {
  return enrichTensor(info, byName.get(`blk.${layer}.${suffix}`), role);
}

function requiredDenseLayerNames(layer) {
  return [
    `blk.${layer}.attn_norm.weight`,
    `blk.${layer}.attn_q.weight`,
    `blk.${layer}.attn_k.weight`,
    `blk.${layer}.attn_v.weight`,
    `blk.${layer}.attn_output.weight`,
    `blk.${layer}.attn_q_norm.weight`,
    `blk.${layer}.attn_k_norm.weight`,
    `blk.${layer}.ffn_norm.weight`,
    `blk.${layer}.ffn_gate.weight`,
    `blk.${layer}.ffn_up.weight`,
    `blk.${layer}.ffn_down.weight`
  ];
}

function requiredMoeLayerNames(layer) {
  return [
    `blk.${layer}.attn_norm.weight`,
    `blk.${layer}.attn_q.weight`,
    `blk.${layer}.attn_k.weight`,
    `blk.${layer}.attn_v.weight`,
    `blk.${layer}.attn_output.weight`,
    `blk.${layer}.attn_q_norm.weight`,
    `blk.${layer}.attn_k_norm.weight`,
    `blk.${layer}.ffn_norm.weight`,
    `blk.${layer}.ffn_gate_inp.weight`,
    `blk.${layer}.ffn_gate_exps.weight`,
    `blk.${layer}.ffn_up_exps.weight`,
    `blk.${layer}.ffn_down_exps.weight`
  ];
}

export function mapQwenTensors(info) {
  const validation = validateIronMindTarget(info);
  const arch = info.architecture;
  const metadata = info.metadata;
  const byName = new Map(info.tensors.map((tensor) => [tensor.name, tensor]));
  const nLayer = getValue(metadata, `${arch}.block_count`);
  const nEmb = getValue(metadata, `${arch}.embedding_length`);
  const nCtx = getValue(metadata, `${arch}.context_length`);
  const nHead = getValue(metadata, `${arch}.attention.head_count`);
  const nHeadKv = getValue(metadata, `${arch}.attention.head_count_kv`);
  const headDim = getValue(metadata, `${arch}.attention.key_length`, nEmb / Math.max(1, nHead));
  const nFf = getValue(metadata, `${arch}.feed_forward_length`);
  const nFfExp = getValue(metadata, `${arch}.expert_feed_forward_length`);
  const nExpert = getValue(metadata, `${arch}.expert_count`);
  const nExpertUsed = getValue(metadata, `${arch}.expert_used_count`);
  const isMoe = arch === "qwen3moe";

  const missing = [];
  const globals = {
    tokenEmbd: enrichTensor(info, byName.get("token_embd.weight"), "token_embd"),
    outputNorm: enrichTensor(info, byName.get("output_norm.weight"), "output_norm"),
    output: enrichTensor(info, byName.get("output.weight") || byName.get("token_embd.weight"), "output")
  };

  for (const [role, tensor] of Object.entries(globals)) {
    if (!tensor) missing.push(role);
  }

  const layers = [];
  for (let layer = 0; layer < nLayer; layer += 1) {
    const required = isMoe ? requiredMoeLayerNames(layer) : requiredDenseLayerNames(layer);
    for (const name of required) {
      if (!byName.has(name)) missing.push(name);
    }

    layers.push({
      index: layer,
      attnNorm: layerTensor(byName, info, layer, "attn_norm.weight", "attn_norm"),
      q: layerTensor(byName, info, layer, "attn_q.weight", "attn_q"),
      k: layerTensor(byName, info, layer, "attn_k.weight", "attn_k"),
      v: layerTensor(byName, info, layer, "attn_v.weight", "attn_v"),
      o: layerTensor(byName, info, layer, "attn_output.weight", "attn_output"),
      qNorm: layerTensor(byName, info, layer, "attn_q_norm.weight", "attn_q_norm"),
      kNorm: layerTensor(byName, info, layer, "attn_k_norm.weight", "attn_k_norm"),
      ffnNorm: layerTensor(byName, info, layer, "ffn_norm.weight", "ffn_norm"),
      ffnGate: layerTensor(byName, info, layer, "ffn_gate.weight", "ffn_gate"),
      ffnUp: layerTensor(byName, info, layer, "ffn_up.weight", "ffn_up"),
      ffnDown: layerTensor(byName, info, layer, "ffn_down.weight", "ffn_down"),
      moeGateInp: layerTensor(byName, info, layer, "ffn_gate_inp.weight", "ffn_gate_inp"),
      moeGateExperts: layerTensor(byName, info, layer, "ffn_gate_exps.weight", "ffn_gate_exps"),
      moeUpExperts: layerTensor(byName, info, layer, "ffn_up_exps.weight", "ffn_up_exps"),
      moeDownExperts: layerTensor(byName, info, layer, "ffn_down_exps.weight", "ffn_down_exps")
    });
  }

  const mappedTensors = [globals.tokenEmbd, globals.outputNorm, globals.output, ...layers.flatMap((layer) => Object.values(layer).filter((value) => value && typeof value === "object"))];
  const mappedBytes = mappedTensors.reduce((sum, tensor) => sum + (tensor.sizeBytes || 0), 0);

  return {
    ok: validation.ok && missing.length === 0,
    architecture: arch,
    hparams: {
      nCtx,
      nLayer,
      nEmb,
      nHead,
      nHeadKv,
      headDim,
      nFf,
      nFfExp,
      nExpert,
      nExpertUsed,
      ropeFreqBase: getValue(metadata, `${arch}.rope.freq_base`, 1000000),
      rmsNormEps: Number(metadata[`${arch}.attention.layer_norm_rms_epsilon`]?.value ?? 1e-6)
    },
    globals,
    layers,
    missing,
    validation,
    tensorCount: info.tensorCount,
    mappedTensorCount: mappedTensors.length,
    mappedBytes
  };
}

export async function loadQwenTensorMap(filePath) {
  const info = await inspectGguf(filePath);
  return mapQwenTensors(info);
}

export function summarizeTensorMap(map) {
  return {
    ok: map.ok,
    architecture: map.architecture,
    layers: map.hparams.nLayer,
    embedding: map.hparams.nEmb,
    heads: map.hparams.nHead,
    kvHeads: map.hparams.nHeadKv,
    headDim: map.hparams.headDim,
    moeExperts: map.hparams.nExpert,
    tensors: map.tensorCount,
    mappedTensors: map.mappedTensorCount,
    mappedGb: Number((map.mappedBytes / 1024 ** 3).toFixed(3)),
    missing: map.missing.length
  };
}
