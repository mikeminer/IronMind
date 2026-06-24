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
const docsEl = document.querySelector("#docs");
const uploadDocsEl = document.querySelector("#uploadDocs");
const documentsEl = document.querySelector("#documents");
const useDocsEl = document.querySelector("#useDocs");
const docModeEl = document.querySelector("#docMode");

const messages = [
  {
    role: "assistant",
    content: "Sto controllando il backend locale di Iurexa."
  }
];
let documents = [];

function setStatus(text, danger = false) {
  statusEl.textContent = text;
  statusEl.style.color = danger ? "var(--danger)" : "var(--muted)";
}

function roleLabel(role) {
  if (role === "user") return "Tu";
  if (role === "assistant") return "Iurexa";
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
    if (useDocsEl.checked) {
      const response = await fetch("/api/documents/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: modelEl.value.trim(),
          ctx: Number(ctxEl.value),
          mode: docModeEl.value,
          question: text
        })
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
      assistant.content = data.answer || "";
      if (data.citations?.length) {
        assistant.content += "\n\nFonti:\n" + data.citations
          .map((citation) => `[${citation.id}] ${citation.source}`)
          .join("\n");
      }
      render();
      const cited = data.citations?.length || 0;
      setStatus(`Analisi documenti completata. fonti=${cited}, backend=${data.ironmind?.backend || "-"}`);
      return;
    }

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
    content: "Nuova chat pronta. Indicami giurisdizione, fatti e obiettivo."
  });
  setStatus("Pronto");
  render();
});

function renderDocuments() {
  documentsEl.innerHTML = "";
  if (!documents.length) {
    documentsEl.textContent = "Nessun documento caricato";
    return;
  }
  for (const doc of documents) {
    const item = document.createElement("div");
    item.className = "document";
    const title = document.createElement("strong");
    title.textContent = doc.title || doc.fileName;
    const meta = document.createElement("span");
    meta.textContent = `${doc.chunks || 0} chunk, ${doc.sections || 0} sezioni`;
    item.append(title, meta);
    documentsEl.append(item);
  }
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    const data = await response.json();
    documents = data.data || [];
    renderDocuments();
  } catch {
    documents = [];
    renderDocuments();
  }
}

uploadDocsEl.addEventListener("click", async () => {
  const files = [...docsEl.files];
  if (!files.length) return;
  uploadDocsEl.disabled = true;
  setStatus("Estraggo testo dai documenti...");
  try {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    const response = await fetch("/api/documents/upload", {
      method: "POST",
      body: form
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
    docsEl.value = "";
    await loadDocuments();
    useDocsEl.checked = documents.length > 0;
    setStatus(`Documenti caricati: ${data.uploaded.length}`);
  } catch (error) {
    setStatus(`Upload fallito: ${error.message}`, true);
  } finally {
    uploadDocsEl.disabled = false;
  }
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
      messages[0].content = `${health.displayName || health.model} e pronta. Posso aiutarti con analisi, bozze e orientamento legale in italiano.`;
      render();
    }
    const files = health.contextStore?.files || 0;
    setStatus(`Backend: ${health.backend}; ${cpuModeText(health.cpuPerformance)}; file disco=${files}`);
    await loadDocuments();
  } catch {
    setStatus("Controllo health fallito", true);
  }
}

render();
loadHealth();
