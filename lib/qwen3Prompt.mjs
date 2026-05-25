function normalizeContent(content) {
  if (Array.isArray(content)) {
    return content.map((part) => typeof part === "string" ? part : part.text || "").join("");
  }
  return content == null ? "" : String(content);
}

function renderTools(tools) {
  if (!tools?.length) return "";
  const body = tools.map((tool) => JSON.stringify({ type: "function", function: tool.function || tool })).join("\n");
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
  const system = options.system || messages.find((msg) => msg.role === "system")?.content;
  const conversation = messages.filter((msg) => msg.role !== "system");
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
          out.push(JSON.stringify({
            name: call.function?.name || call.name,
            arguments: call.function?.arguments || call.arguments || {}
          }), "\n");
        }
        out.push("</tool_call>");
      }
      out.push("<|im_end|>\n");
    } else if (msg.role === "tool") {
      out.push("<|im_start|>user\n<tool_response>\n", normalizeContent(msg.content), "\n</tool_response><|im_end|>\n");
    }
  }

  out.push("<|im_start|>assistant\n");
  if (options.think === false) out.push("<think>\n\n</think>\n\n");
  return out.join("");
}
