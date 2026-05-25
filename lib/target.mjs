export const IRONMIND_TARGET = {
  id: "qwen3-coder-30b-a3b",
  label: "Qwen3-Coder 30B A3B",
  architectures: ["qwen3moe", "qwen3"],
  preferredArchitecture: "qwen3moe",
  minContext: 131072,
  requiresTokenizerModel: true
};

function has(metadata, key) {
  return Object.prototype.hasOwnProperty.call(metadata, key);
}

function value(metadata, key) {
  return metadata[key]?.value ?? metadata[key]?.length;
}

function key(prefix, suffix) {
  return `${prefix}.${suffix}`;
}

export function validateIronMindTarget(info, target = IRONMIND_TARGET) {
  const issues = [];
  const warnings = [];
  const arch = info.architecture;

  if (!arch) {
    issues.push("Missing general.architecture.");
    return { ok: false, target: target.id, architecture: null, issues, warnings };
  }

  if (!target.architectures.includes(arch)) {
    issues.push(`Unsupported architecture ${arch}; expected ${target.architectures.join(" or ")}.`);
  }

  const required = [
    "general.architecture",
    key(arch, "context_length"),
    key(arch, "embedding_length"),
    key(arch, "block_count"),
    key(arch, "attention.head_count"),
    key(arch, "attention.head_count_kv"),
    key(arch, "attention.layer_norm_rms_epsilon"),
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens"
  ];

  for (const requiredKey of required) {
    if (!has(info.metadata, requiredKey)) issues.push(`Missing required GGUF key ${requiredKey}.`);
  }

  const contextLength = Number(value(info.metadata, key(arch, "context_length")) || 0);
  if (contextLength && contextLength < target.minContext) {
    warnings.push(`Context length ${contextLength} is below IronMind's initial ${target.minContext} target.`);
  }

  if (arch.includes("moe")) {
    for (const moeKey of ["expert_count", "expert_used_count", "expert_feed_forward_length"]) {
      const fullKey = key(arch, moeKey);
      if (!has(info.metadata, fullKey)) issues.push(`Missing MoE key ${fullKey}.`);
    }
  } else {
    warnings.push(`${arch} is dense; MoE routing will stay disabled for this model.`);
  }

  if (!has(info.metadata, "tokenizer.chat_template")) {
    warnings.push("Missing tokenizer.chat_template; IronMind will use its built-in Qwen3 renderer.");
  }

  const tensorNames = new Set(info.tensors.map((tensor) => tensor.name));
  for (const tensorName of ["token_embd.weight", "output_norm.weight"]) {
    if (!tensorNames.has(tensorName)) warnings.push(`Tensor ${tensorName} was not found in the inspected tensor directory.`);
  }

  return {
    ok: issues.length === 0,
    target: target.id,
    architecture: arch,
    issues,
    warnings
  };
}
