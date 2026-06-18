import {
  canonicalJson,
  canonicalizeMessages,
  canonicalizeTools,
  normalizeContent,
  qwenToolCallObject
} from "./toolCalls.mjs";

function renderTools(tools) {
  const canonical = canonicalizeTools(tools);
  if (!canonical.length) return "";
  const body = canonical.map((tool) => canonicalJson(tool)).join("\n");
  return `
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${body}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>`;
}

export function renderQwen3Chat(messages, options = {}) {
  const out = [];
  const canonicalMessages = canonicalizeMessages(messages);
  const system = options.system || canonicalMessages.find((msg) => msg.role === "system")?.content;
  const conversation = canonicalMessages.filter((msg) => msg.role !== "system");
  const lastUserIndex = conversation.map((msg, index) => msg.role === "user" ? index : -1).filter((index) => index >= 0).pop();

  if (system || options.tools?.length) {
    out.push("<|im_start|>system\n");
    if (system) out.push(normalizeContent(system).trim(), "\n");
    out.push(renderTools(options.tools));
    out.push("<|im_end|>\n");
  }

  for (let i = 0; i < conversation.length; i += 1) {
    const msg = conversation[i];
    if (msg.role === "user") {
      out.push("<|im_start|>user\n", normalizeContent(msg.content));
      if (options.think !== undefined && i === lastUserIndex) {
        out.push(options.think ? " /think" : " /no_think");
      }
      out.push("<|im_end|>\n");
    } else if (msg.role === "assistant") {
      out.push("<|im_start|>assistant\n");
      if (msg.thinking && options.think) out.push("<think>", normalizeContent(msg.thinking), "</think>\n");
      if (msg.content) out.push(normalizeContent(msg.content));
      if (msg.tool_calls?.length) {
        out.push("<tool_call>\n");
        for (const call of msg.tool_calls) {
          const rendered = qwenToolCallObject(call);
          if (rendered) out.push(canonicalJson(rendered), "\n");
        }
        out.push("</tool_call>");
      }
      out.push("<|im_end|>\n");
    } else if (msg.role === "tool") {
      const response = msg.tool_call_id
        ? canonicalJson({ tool_call_id: msg.tool_call_id, name: msg.name, content: normalizeContent(msg.content) })
        : normalizeContent(msg.content);
      out.push("<|im_start|>user\n<tool_response>\n", response, "\n</tool_response><|im_end|>\n");
    }
  }

  out.push("<|im_start|>assistant\n");
  if (options.think === false) out.push("<think>\n\n</think>\n\n");
  return out.join("");
}
