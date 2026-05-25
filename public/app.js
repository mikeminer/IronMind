const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendEl = document.querySelector("#send");
const clearEl = document.querySelector("#clear");
const statusEl = document.querySelector("#status");
const modelEl = document.querySelector("#model");
const ctxEl = document.querySelector("#ctx");
const thinkEl = document.querySelector("#think");

const messages = [
  {
    role: "assistant",
    content: "IronMind is ready. Start Ollama and pull qwen3-coder:30b if the backend is not loaded yet."
  }
];

function setStatus(text, danger = false) {
  statusEl.textContent = text;
  statusEl.style.color = danger ? "var(--danger)" : "var(--muted)";
}

function render() {
  messagesEl.innerHTML = "";
  for (const message of messages) {
    const item = document.createElement("article");
    item.className = `message ${message.role}`;

    const role = document.createElement("span");
    role.className = "role";
    role.textContent = message.role;

    const content = document.createElement("div");
    content.textContent = message.content;

    item.append(role, content);
    messagesEl.append(item);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function readNdjson(response, onObject) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) onObject(JSON.parse(trimmed));
    }
  }

  const tail = buffer.trim();
  if (tail) onObject(JSON.parse(tail));
}

async function sendPrompt(text) {
  messages.push({ role: "user", content: text });
  const assistant = { role: "assistant", content: "" };
  messages.push(assistant);
  render();

  sendEl.disabled = true;
  setStatus("Generating...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: modelEl.value.trim(),
        ctx: Number(ctxEl.value),
        think: thinkEl.checked,
        messages: messages
          .filter((message) => message.content.trim())
          .map((message) => ({ role: message.role, content: message.content }))
      })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    await readNdjson(response, (event) => {
      if (event.type === "delta") {
        assistant.content += event.content;
        render();
      }
      if (event.type === "done") {
        setStatus(`Done. prompt=${event.promptEvalCount || 0}, output=${event.evalCount || 0}`);
      }
    });
  } catch (error) {
    assistant.content = `Backend error: ${error.message}`;
    setStatus("Backend unavailable", true);
    render();
  } finally {
    sendEl.disabled = false;
    promptEl.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";
  sendPrompt(text);
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    composer.requestSubmit();
  }
});

clearEl.addEventListener("click", () => {
  messages.splice(0, messages.length, {
    role: "assistant",
    content: "New IronMind session."
  });
  setStatus("Ready");
  render();
});

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    modelEl.value = health.model;
    ctxEl.value = health.context;
    setStatus(`Backend: ${health.backend}`);
  } catch {
    setStatus("Health check failed", true);
  }
}

render();
loadHealth();
