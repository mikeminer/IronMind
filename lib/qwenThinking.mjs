const THINKING_DIRECTIVE_RE = /(^|\s)\/(?:no_)?think(?=\s|$)/i;

export function usesQwen3Thinking(model, options = {}) {
  return Boolean(options.force) || /qwen3/i.test(String(model || ""));
}

export function wantsThinking(payload = {}) {
  return payload.think ?? Boolean(payload.reasoning || payload.reasoning_effort);
}

export function addQwen3ThinkingDirective(payload = {}, fallbackModel = "", options = {}) {
  const model = payload.model || fallbackModel;
  if (!usesQwen3Thinking(model, options) || !Array.isArray(payload.messages)) return payload;

  const lastUserIndex = payload.messages
    .map((message, index) => message?.role === "user" ? index : -1)
    .filter((index) => index >= 0)
    .pop();

  if (lastUserIndex === undefined) return payload;

  const directive = wantsThinking(payload) ? "/think" : "/no_think";
  const messages = payload.messages.map((message, index) => {
    if (index !== lastUserIndex) return message;
    const content = message?.content == null ? "" : String(message.content);
    if (THINKING_DIRECTIVE_RE.test(content)) return message;
    const nextContent = `${content.trimEnd()} ${directive}`.trimStart();
    return { ...message, content: nextContent };
  });

  return { ...payload, messages };
}

export function stripQwenThinking(content) {
  let visible = content == null ? "" : String(content);
  visible = visible.replace(/<think>[\s\S]*?<\/think>\s*/gi, "");
  visible = visible.replace(/\s*<think>[\s\S]*$/i, "");
  visible = visible.replace(/^\s*<\/think>\s*/i, "");
  return visible.trimStart();
}

export function stripQwenThinkingFromMessage(message = {}) {
  if (message.content === undefined) return message;
  return { ...message, content: stripQwenThinking(message.content) };
}
