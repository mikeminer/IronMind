import fs from "node:fs/promises";
import path from "node:path";

const MAGIC = Buffer.from("IRONKV1\0", "ascii");
const HEADER_SIZE = 32;

function u64(value) {
  const out = Buffer.alloc(8);
  out.writeBigUInt64LE(BigInt(value), 0);
  return out;
}

export async function writeIronKv(filePath, header, payload = Buffer.alloc(0)) {
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const payloadBytes = Buffer.isBuffer(payload) ? payload : Buffer.from(payload || "");
  const fixed = Buffer.concat([
    MAGIC,
    u64(headerBytes.length),
    u64(payloadBytes.length),
    Buffer.alloc(8)
  ]);
  if (fixed.length !== HEADER_SIZE) throw new Error("Invalid IronKV fixed header size.");
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.concat([fixed, headerBytes, payloadBytes]));
}

export async function readIronKv(filePath) {
  const handle = await fs.open(filePath, "r");
  try {
    const fixed = Buffer.alloc(HEADER_SIZE);
    const { bytesRead } = await handle.read(fixed, 0, HEADER_SIZE, 0);
    if (bytesRead !== HEADER_SIZE) throw new Error("Short IronKV header.");
    if (!fixed.subarray(0, 8).equals(MAGIC)) throw new Error("Invalid IronKV magic.");

    const headerLength = Number(fixed.readBigUInt64LE(8));
    const payloadLength = Number(fixed.readBigUInt64LE(16));
    const headerBytes = Buffer.alloc(headerLength);
    await handle.read(headerBytes, 0, headerLength, HEADER_SIZE);
    return {
      header: JSON.parse(headerBytes.toString("utf8")),
      payloadOffset: HEADER_SIZE + headerLength,
      payloadLength
    };
  } finally {
    await handle.close();
  }
}

export async function readIronKvPayload(filePath) {
  const kv = await readIronKv(filePath);
  const handle = await fs.open(filePath, "r");
  try {
    const payload = Buffer.alloc(kv.payloadLength);
    if (kv.payloadLength) {
      const { bytesRead } = await handle.read(payload, 0, kv.payloadLength, kv.payloadOffset);
      if (bytesRead !== kv.payloadLength) throw new Error("Short IronKV payload.");
    }
    return { ...kv, payload };
  } finally {
    await handle.close();
  }
}
