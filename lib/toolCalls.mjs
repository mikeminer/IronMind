import crypto from "node:crypto";

export function stableValue(value) {
  if (value === undefined) return undefined;
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((item) => stableValue(item));

  const out = {};
  for (const key of Object.keys(value).sort()) {
    const normalized = stableValue(value[key]);
    if (normalized !== undefined) out[key] = normalized;
  }
  return out;
}

export function canonicalJson(value) {
  return JSON.stringify(stableValue(value));
}

export function normalizeContent(content) {
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === "string") return part;
      if (part?.type === "text" || part?.type === "output_text") return part.text || "";
      if (part?.text) return part.text;
      return "";
    }).join("");
  }
  return content == null ? "" : String(content);
}

function parseJsonMaybe(value) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed) return {};
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

export function canonicalToolArguments(value) {
  const parsed = parseJsonMaybe(value);
  if (parsed == null) return {};
  if (typeof parsed === "object" && !Array.isArray(parsed)) return stableValue(parsed);
  return { value: parsed };
}

export function canonicalToolArgumentsJson(value) {
  return canonicalJson(canonicalToolArguments(value));
}

function toolCallSeed(call, index) {
  const fn = call?.function || call || {};
  return canonicalJson({
    index,
    name: fn.name || call?.name || "",
    arguments: canonicalToolArguments(fn.arguments ?? call?.arguments)
  });
}

export function canonicalizeToolCall(call, index = 0) {
  const fn = call?.function || call || {};
  const name = fn.name || call?.name;
  if (!name) return null;
  const seed = toolCallSeed(call, index);
  const id = call?.id || `call_${crypto.createHash("sha1").update(seed).digest("hex").slice(0, 12)}`;
  return {
    id,
    type: "function",
    function: {
      name: String(name),
      arguments: canonicalToolArgumentsJson(fn.arguments ?? call?.arguments)
    }
  };
}

export function canonicalizeToolCalls(calls = []) {
  return calls
    .map((call, index) => canonicalizeToolCall(call, index))
    .filter(Boolean);
}

export function qwenToolCallObject(call) {
  const canonical = canonicalizeToolCall(call);
  if (!canonical) return null;
  return {
    name: canonical.function.name,
    arguments: canonicalToolArguments(canonical.function.arguments)
  };
}

export function canonicalizeToolDefinition(tool) {
  const fn = tool?.type === "function" ? tool.function : (tool?.function || tool);
  if (!fn?.name) return null;
  const out = {
    type: "function",
    function: {
      name: String(fn.name)
    }
  };
  if (fn.description) out.function.description = String(fn.description);
  out.function.parameters = stableValue(fn.parameters || { type: "object", properties: {} });
  return out;
}

export function canonicalizeTools(tools = []) {
  return tools
    .map(canonicalizeToolDefinition)
    .filter(Boolean)
    .sort((a, b) => a.function.name.localeCompare(b.function.name));
}

export function canonicalizeMessages(messages = []) {
  return messages.filter(Boolean).map((message) => {
    const role = message.role || "user";
    const out = { role };
    if (message.name) out.name = String(message.name);
    if (message.tool_call_id) out.tool_call_id = String(message.tool_call_id);
    if (message.thinking) out.thinking = normalizeContent(message.thinking);
    if (message.content !== undefined) out.content = normalizeContent(message.content);
    if (message.tool_calls?.length) out.tool_calls = canonicalizeToolCalls(message.tool_calls);
    return out;
  });
}

function parseToolCallBlock(block) {
  const trimmed = block.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    const calls = [];
    for (const line of trimmed.split(/\r?\n/)) {
      const text = line.trim();
      if (!text) continue;
      try {
        calls.push(JSON.parse(text));
      } catch {
        return [];
      }
    }
    return calls;
  }
}

export function extractToolCallsFromText(text) {
  const source = normalizeContent(text);
  const calls = [];
  const content = source.replace(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g, (_match, body) => {
    calls.push(...parseToolCallBlock(body));
    return "";
  }).trim();
  return {
    content,
    tool_calls: canonicalizeToolCalls(calls)
  };
}
