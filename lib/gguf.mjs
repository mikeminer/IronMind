import fs from "node:fs/promises";
import crypto from "node:crypto";

export const GGUF_TYPES = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12
};

export const GGUF_TYPE_NAMES = new Map([
  [0, "u8"],
  [1, "i8"],
  [2, "u16"],
  [3, "i16"],
  [4, "u32"],
  [5, "i32"],
  [6, "f32"],
  [7, "bool"],
  [8, "str"],
  [9, "arr"],
  [10, "u64"],
  [11, "i64"],
  [12, "f64"]
]);

const SCALAR_SIZES = new Map([
  [GGUF_TYPES.UINT8, 1],
  [GGUF_TYPES.INT8, 1],
  [GGUF_TYPES.UINT16, 2],
  [GGUF_TYPES.INT16, 2],
  [GGUF_TYPES.UINT32, 4],
  [GGUF_TYPES.INT32, 4],
  [GGUF_TYPES.FLOAT32, 4],
  [GGUF_TYPES.BOOL, 1],
  [GGUF_TYPES.UINT64, 8],
  [GGUF_TYPES.INT64, 8],
  [GGUF_TYPES.FLOAT64, 8]
]);

class GgufReader {
  constructor(handle, size) {
    this.handle = handle;
    this.size = size;
    this.offset = 0n;
  }

  async readBytes(length) {
    if (this.offset + BigInt(length) > BigInt(this.size)) {
      throw new Error(`Unexpected EOF at ${this.offset}`);
    }
    const buffer = Buffer.allocUnsafe(length);
    const { bytesRead } = await this.handle.read(buffer, 0, length, Number(this.offset));
    if (bytesRead !== length) throw new Error(`Short read at ${this.offset}`);
    this.offset += BigInt(length);
    return buffer;
  }

  async skip(length) {
    this.offset += BigInt(length);
    if (this.offset > BigInt(this.size)) throw new Error(`Unexpected EOF at ${this.offset}`);
  }

  async u32() {
    return (await this.readBytes(4)).readUInt32LE(0);
  }

  async i32() {
    return (await this.readBytes(4)).readInt32LE(0);
  }

  async u64() {
    return (await this.readBytes(8)).readBigUInt64LE(0);
  }

  async i64() {
    return (await this.readBytes(8)).readBigInt64LE(0);
  }

  async f32() {
    return (await this.readBytes(4)).readFloatLE(0);
  }

  async f64() {
    return (await this.readBytes(8)).readDoubleLE(0);
  }

  async string(maxKeepBytes = 65536) {
    const length = await this.u64();
    const n = toSafeNumber(length, "string length");
    if (n > maxKeepBytes) {
      await this.skip(n);
      return { value: null, bytes: n, truncated: true };
    }
    return { value: (await this.readBytes(n)).toString("utf8"), bytes: n, truncated: false };
  }
}

function toSafeNumber(value, label) {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} is too large for this inspector`);
  }
  return Number(value);
}

function align(value, alignment) {
  const mask = BigInt(alignment - 1);
  return Number((value + mask) & ~mask);
}

function scalarValueToJson(value) {
  return typeof value === "bigint" ? value.toString() : value;
}

async function readScalar(reader, type) {
  switch (type) {
    case GGUF_TYPES.UINT8:
      return (await reader.readBytes(1)).readUInt8(0);
    case GGUF_TYPES.INT8:
      return (await reader.readBytes(1)).readInt8(0);
    case GGUF_TYPES.UINT16:
      return (await reader.readBytes(2)).readUInt16LE(0);
    case GGUF_TYPES.INT16:
      return (await reader.readBytes(2)).readInt16LE(0);
    case GGUF_TYPES.UINT32:
      return reader.u32();
    case GGUF_TYPES.INT32:
      return reader.i32();
    case GGUF_TYPES.FLOAT32:
      return reader.f32();
    case GGUF_TYPES.BOOL:
      return Boolean((await reader.readBytes(1)).readUInt8(0));
    case GGUF_TYPES.STRING:
      return (await reader.string()).value;
    case GGUF_TYPES.UINT64:
      return reader.u64();
    case GGUF_TYPES.INT64:
      return reader.i64();
    case GGUF_TYPES.FLOAT64:
      return reader.f64();
    default:
      throw new Error(`Unsupported GGUF scalar type ${type}`);
  }
}

async function skipArray(reader, innerType, count) {
  const n = toSafeNumber(count, "array length");
  if (innerType === GGUF_TYPES.STRING) {
    for (let i = 0; i < n; i += 1) {
      const length = await reader.u64();
      await reader.skip(toSafeNumber(length, "array string length"));
    }
    return;
  }

  const size = SCALAR_SIZES.get(innerType);
  if (!size) throw new Error(`Unsupported GGUF array type ${innerType}`);
  await reader.skip(n * size);
}

async function readArray(reader, innerType, count) {
  const n = toSafeNumber(count, "array length");
  const values = new Array(n);
  for (let i = 0; i < n; i += 1) {
    values[i] = scalarValueToJson(await readScalar(reader, innerType));
  }
  return values;
}

async function readKv(reader, options = {}) {
  const key = (await reader.string()).value;
  const type = await reader.i32();

  if (type === GGUF_TYPES.ARRAY) {
    const innerType = await reader.i32();
    const count = await reader.u64();
    const keepArrays = options.keepArrays instanceof Set ? options.keepArrays : new Set(options.keepArrays || []);
    const keep = keepArrays.has(key);
    const item = {
      key,
      type: "array",
      innerType,
      innerTypeName: GGUF_TYPE_NAMES.get(innerType) || `unknown:${innerType}`,
      length: toSafeNumber(count, `array length for ${key}`)
    };
    if (keep) item.values = await readArray(reader, innerType, count);
    else await skipArray(reader, innerType, count);
    return item;
  }

  const value = await readScalar(reader, type);
  return {
    key,
    type,
    typeName: GGUF_TYPE_NAMES.get(type) || `unknown:${type}`,
    value: scalarValueToJson(value)
  };
}

async function readTensorInfo(reader) {
  const name = (await reader.string()).value;
  const nDims = await reader.u32();
  const dims = [];
  for (let i = 0; i < nDims; i += 1) {
    dims.push((await reader.u64()).toString());
  }
  const type = await reader.i32();
  const offset = await reader.u64();
  return {
    name,
    dims,
    type,
    typeName: `ggml:${type}`,
    offset: offset.toString()
  };
}

function metadataObject(kv) {
  const out = {};
  for (const item of kv) {
    if (item.type === "array") {
      out[item.key] = {
        type: "array",
        innerType: item.innerTypeName,
        length: item.length,
        values: item.values
      };
    } else {
      out[item.key] = {
        type: item.typeName,
        value: item.value
      };
    }
  }
  return out;
}

function fingerprint(info) {
  const hash = crypto.createHash("sha256");
  hash.update(`gguf:${info.version}:${info.tensorCount}:${info.kvCount}\n`);
  for (const key of [
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    `${info.architecture}.context_length`,
    `${info.architecture}.embedding_length`,
    `${info.architecture}.block_count`
  ]) {
    const entry = info.metadata[key];
    if (entry) hash.update(`${key}=${JSON.stringify(entry)}\n`);
  }
  for (const tensor of info.tensors) {
    hash.update(`${tensor.name}:${tensor.type}:${tensor.dims.join("x")}\n`);
  }
  return hash.digest("hex");
}

export async function inspectGguf(filePath, options = {}) {
  const handle = await fs.open(filePath, "r");
  try {
    const stat = await handle.stat();
    const reader = new GgufReader(handle, stat.size);
    const magic = (await reader.readBytes(4)).toString("ascii");
    if (magic !== "GGUF") throw new Error(`Invalid GGUF magic ${JSON.stringify(magic)}`);

    const version = await reader.u32();
    if (version < 2 || version > 3) {
      throw new Error(`Unsupported GGUF version ${version}`);
    }

    const tensorCount = toSafeNumber(await reader.u64(), "tensor count");
    const kvCount = toSafeNumber(await reader.u64(), "kv count");
    const kv = [];
    const kvOptions = {
      keepArrays: options.keepArrays instanceof Set ? options.keepArrays : new Set(options.keepArrays || [])
    };
    for (let i = 0; i < kvCount; i += 1) {
      kv.push(await readKv(reader, kvOptions));
    }

    const metadata = metadataObject(kv);
    const alignment = Number(metadata["general.alignment"]?.value || 32);
    if (alignment <= 0 || (alignment & (alignment - 1)) !== 0) {
      throw new Error(`Invalid GGUF alignment ${alignment}`);
    }

    const tensors = [];
    for (let i = 0; i < tensorCount; i += 1) {
      const tensor = await readTensorInfo(reader);
      if (options.tensorLimit === undefined || tensors.length < options.tensorLimit) tensors.push(tensor);
    }

    const dataOffset = align(reader.offset, alignment);
    const architecture = metadata["general.architecture"]?.value || null;
    const info = {
      path: filePath,
      size: stat.size,
      version,
      kvCount,
      tensorCount,
      alignment,
      dataOffset,
      architecture,
      metadata,
      tensors
    };
    info.fingerprint = fingerprint(info);
    return info;
  } finally {
    await handle.close();
  }
}

export function summarizeGguf(info) {
  const prefix = info.architecture || "unknown";
  const get = (key) => info.metadata[key]?.value ?? info.metadata[key]?.length ?? null;
  return {
    architecture: info.architecture,
    name: get("general.name"),
    fileType: get("general.file_type"),
    quantizationVersion: get("general.quantization_version"),
    contextLength: get(`${prefix}.context_length`),
    embeddingLength: get(`${prefix}.embedding_length`),
    blockCount: get(`${prefix}.block_count`),
    headCount: get(`${prefix}.attention.head_count`),
    headCountKv: get(`${prefix}.attention.head_count_kv`),
    expertCount: get(`${prefix}.expert_count`),
    expertUsedCount: get(`${prefix}.expert_used_count`),
    tokenizer: get("tokenizer.ggml.model"),
    vocabItems: get("tokenizer.ggml.tokens"),
    tensorCount: info.tensorCount,
    kvCount: info.kvCount,
    dataOffset: info.dataOffset,
    fingerprint: info.fingerprint.slice(0, 16)
  };
}
