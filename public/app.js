const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendEl = document.querySelector("#send");
const clearEl = document.querySelector("#clear");
const statusEl = document.querySelector("#status");
const modelEl = document.querySelector("#model");
const ctxEl = document.querySelector("#ctx");
const kvDiskEl = document.querySelector("#kvDisk");
const cpuModeEl = document.querySelector("#cpuMode");
const thinkEl = document.querySelector("#think");

const messages = [
  {
    role: "assistant",
    content: "Sto controllando il backend locale."
  }
];

function setStatus(text, danger = false) {
  statusEl.textContent = text;
  statusEl.style.color = danger ? "var(--danger)" : "var(--muted)";
}

function roleLabel(role) {
  if (role === "user") return "Tu";
  if (role === "assistant") return "IronMind";
  return role;
}

function render() {
  messagesEl.innerHTML = "";
  for (const message of messages) {
    const item = document.createElement("article");
    item.className = `message ${message.role}`;

    const role = document.createElement("span");
    role.className = "role";
    role.textContent = roleLabel(message.role);

    const content = document.createElement("div");
    content.textContent = message.content;

    item.append(role, content);
    messagesEl.append(item);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function cpuModeText(performance = {}) {
  const mode = performance.cpuOnly ? "solo CPU" : "GPU consentita";
  const ctx = performance.interactiveContext || "completo";
  return `${mode}, ${performance.profile || "sconosciuto"}, ctx=${ctx}, thread=${performance.threads || "-"}`;
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
  setStatus("Sto generando...");

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
        const snapshot = event.contextSnapshot;
        const cached = snapshot ? `, disco=${snapshot.estimatedTokens}t` : "";
        const speed = event.tokensPerSecond ? `, ${event.tokensPerSecond} tok/s` : "";
        const latency = event.totalDurationMs ? `, ${Math.round(event.totalDurationMs / 100) / 10}s` : "";
        setStatus(`Fatto. prompt=${event.promptEvalCount || 0}, output=${event.evalCount || 0}${speed}${latency}${cached}`);
      }
    });
  } catch (error) {
    assistant.content = `Errore backend: ${error.message}`;
    setStatus("Backend non disponibile", true);
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
    content: "Nuova chat pronta. Scrivimi pure."
  });
  setStatus("Pronto");
  render();
});

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    modelEl.value = health.model;
    ctxEl.value = health.cpuPerformance?.interactiveContext || health.context;
    kvDiskEl.value = health.kvDiskDir;
    cpuModeEl.value = cpuModeText(health.cpuPerformance);
    if (messages.length === 1 && messages[0].role === "assistant") {
      messages[0].content = `${health.displayName || health.model} e pronto. Parlo italiano.`;
      render();
    }
    const files = health.contextStore?.files || 0;
    setStatus(`Backend: ${health.backend}; ${cpuModeText(health.cpuPerformance)}; file disco=${files}`);
  } catch {
    setStatus("Controllo health fallito", true);
  }
}

render();
loadHealth();
