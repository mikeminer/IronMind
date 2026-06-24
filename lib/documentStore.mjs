import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const rootDir = path.resolve(path.dirname(__filename), "..");
const extractorPath = path.join(rootDir, "tools", "extract-document.py");

const STOPWORDS = new Set([
  "a", "ad", "al", "alla", "allo", "ai", "agli", "alle", "anche", "che", "chi", "con",
  "da", "dal", "dalla", "dei", "del", "della", "di", "e", "gli", "ha", "il", "in",
  "la", "le", "lo", "ma", "nel", "nella", "non", "o", "per", "piu", "puo",
  "se", "si", "sono", "su", "tra", "un", "una", "uno", "e'", "dell", "all"
]);

export function defaultDocumentDir() {
  return path.join(os.homedir(), ".ironmind", "documents");
}

export function bundledPythonPath() {
  return path.join(
    os.homedir(),
    ".cache",
    "codex-runtimes",
    "codex-primary-runtime",
    "dependencies",
    "python",
    process.platform === "win32" ? "python.exe" : "python"
  );
}

function pythonPath(config = {}) {
  if (config.documentPython) return path.resolve(config.documentPython);
  if (process.env.IRONMIND_DOCUMENT_PYTHON) return path.resolve(process.env.IRONMIND_DOCUMENT_PYTHON);
  const bundled = bundledPythonPath();
  if (fsSync.existsSync(bundled)) return bundled;
  return "python";
}

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function sanitizeName(name = "document") {
  return String(name)
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 160) || "document";
}

function writeJsonAtomic(filePath, value) {
  const tmp = `${filePath}.${process.pid}.tmp`;
  return fs.writeFile(tmp, JSON.stringify(value, null, 2), "utf8")
    .then(() => fs.rename(tmp, filePath));
}

function runExtractor(config, filePath) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath(config), [extractorPath, filePath], {
      cwd: rootDir,
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      let parsed = null;
      try {
        parsed = JSON.parse(stdout || "{}");
      } catch {
        parsed = null;
      }
      if (code === 0 && parsed?.ok) return resolve(parsed);
      reject(new Error(parsed?.error || stderr || `extractor exited ${code}`));
    });
  });
}

function normalizeText(text = "") {
  return String(text)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function terms(text = "") {
  const seen = new Set();
  for (const match of normalizeText(text).matchAll(/[a-z0-9]{3,}/g)) {
    const term = match[0];
    if (!STOPWORDS.has(term)) seen.add(term);
  }
  return [...seen];
}

function sourceLabel(chunk) {
  const parts = [chunk.title || chunk.fileName || chunk.docId];
  if (chunk.pageStart || chunk.pageEnd) {
    parts.push(chunk.pageStart === chunk.pageEnd ? `p. ${chunk.pageStart}` : `pp. ${chunk.pageStart}-${chunk.pageEnd}`);
  } else if (chunk.paragraphStart || chunk.paragraphEnd) {
    parts.push(chunk.paragraphStart === chunk.paragraphEnd
      ? `par. ${chunk.paragraphStart}`
      : `par. ${chunk.paragraphStart}-${chunk.paragraphEnd}`);
  }
  return parts.filter(Boolean).join(", ");
}

export function chunkDocument(document, options = {}) {
  const maxChars = Number(options.maxChars || 1800);
  const chunks = [];
  let current = [];
  let currentChars = 0;

  function flush() {
    if (!current.length) return;
    const first = current[0];
    const last = current.at(-1);
    const text = current.map((section) => section.text).join("\n\n");
    const chunk = {
      id: `${document.id}#c${chunks.length + 1}`,
      docId: document.id,
      title: document.title,
      fileName: document.fileName,
      pageStart: first.page || null,
      pageEnd: last.page || first.page || null,
      paragraphStart: first.paragraph || null,
      paragraphEnd: last.paragraph || first.paragraph || null,
      text,
      terms: terms(text)
    };
    chunk.source = sourceLabel(chunk);
    chunks.push(chunk);
    current = [];
    currentChars = 0;
  }

  for (const section of document.sections || []) {
    const text = String(section.text || "").trim();
    if (!text) continue;
    if (current.length && currentChars + text.length > maxChars) flush();
    current.push({ ...section, text });
    currentChars += text.length;
  }
  flush();
  return chunks;
}

export async function saveUploadedDocument(config, upload) {
  const dir = path.resolve(config.documentStoreDir || defaultDocumentDir());
  await fs.mkdir(path.join(dir, "files"), { recursive: true });
  await fs.mkdir(path.join(dir, "index"), { recursive: true });

  const fileName = sanitizeName(upload.filename);
  const digest = sha256(upload.data);
  const ext = path.extname(fileName).toLowerCase();
  const id = digest.slice(0, 16);
  const storedName = `${id}${ext || ".bin"}`;
  const filePath = path.join(dir, "files", storedName);
  await fs.writeFile(filePath, upload.data);

  const extracted = await runExtractor(config, filePath);
  const document = {
    id,
    title: path.basename(fileName, ext) || fileName,
    fileName,
    mimeType: upload.mimeType || "",
    filePath,
    hash: digest,
    uploadedAt: new Date().toISOString(),
    sections: extracted.sections || [],
    warnings: extracted.warnings || []
  };
  document.characters = document.sections.reduce((sum, section) => sum + String(section.text || "").length, 0);
  document.chunks = chunkDocument(document);
  await writeJsonAtomic(path.join(dir, "index", `${id}.json`), document);
  return publicDocument(document);
}

export function publicDocument(document) {
  return {
    id: document.id,
    title: document.title,
    fileName: document.fileName,
    mimeType: document.mimeType,
    uploadedAt: document.uploadedAt,
    characters: document.characters,
    sections: document.sections?.length || 0,
    chunks: document.chunks?.length || 0,
    warnings: document.warnings || []
  };
}

export async function listDocuments(config) {
  const dir = path.resolve(config.documentStoreDir || defaultDocumentDir());
  const indexDir = path.join(dir, "index");
  try {
    const files = await fs.readdir(indexDir);
    const documents = [];
    for (const file of files.filter((name) => name.endsWith(".json"))) {
      try {
        const doc = JSON.parse(await fs.readFile(path.join(indexDir, file), "utf8"));
        documents.push(publicDocument(doc));
      } catch {
        // Ignore corrupt indexes; the original upload remains on disk for recovery.
      }
    }
    documents.sort((a, b) => String(b.uploadedAt).localeCompare(String(a.uploadedAt)));
    return documents;
  } catch {
    return [];
  }
}

export async function loadDocument(config, id) {
  const dir = path.resolve(config.documentStoreDir || defaultDocumentDir());
  const safeId = String(id || "").replace(/[^a-f0-9]/gi, "");
  if (!safeId) return null;
  try {
    return JSON.parse(await fs.readFile(path.join(dir, "index", `${safeId}.json`), "utf8"));
  } catch {
    return null;
  }
}

export async function loadDocuments(config, ids = []) {
  const requested = ids.filter(Boolean);
  if (!requested.length) {
    const listed = await listDocuments(config);
    return Promise.all(listed.map((doc) => loadDocument(config, doc.id)));
  }
  return Promise.all(requested.map((id) => loadDocument(config, id)));
}

export function rankChunks(query, documents, options = {}) {
  const queryTerms = terms(query);
  const queryTermSet = new Set(queryTerms);
  const limit = Number(options.limit || 8);
  const chunks = documents
    .filter(Boolean)
    .flatMap((doc) => (doc.chunks || []).map((chunk) => ({ ...chunk, title: doc.title, fileName: doc.fileName })));

  const ranked = chunks.map((chunk) => {
    let score = 0;
    const chunkTerms = new Set(chunk.terms || terms(chunk.text));
    for (const term of queryTermSet) {
      if (chunkTerms.has(term)) score += 3;
      else if (normalizeText(chunk.text).includes(term)) score += 1;
    }
    return { ...chunk, score };
  }).sort((a, b) => b.score - a.score);

  const useful = ranked.filter((chunk) => chunk.score > 0);
  return (useful.length ? useful : ranked).slice(0, limit);
}

export function buildRagPrompt({ question, mode = "answer", chunks = [], documents = [] }) {
  const modeText = {
    answer: "Rispondi alla domanda usando solo le fonti fornite.",
    summary: "Prepara un riassunto professionale dei documenti.",
    compare: "Confronta i documenti, evidenziando convergenze, differenze, rischi e lacune.",
    report: "Prepara un report strutturato con executive summary, punti chiave, rischi, citazioni e prossimi passi."
  }[mode] || "Rispondi alla domanda usando solo le fonti fornite.";

  const sourceBlock = chunks.map((chunk, index) => {
    const label = `F${index + 1}`;
    return `[${label}] ${chunk.source}\n${chunk.text}`;
  }).join("\n\n---\n\n");

  const docList = documents.filter(Boolean).map((doc) => `- ${doc.title} (${doc.fileName})`).join("\n");
  return [
    "Sei Iurexa, assistente legale locale. Devi ragionare sui testi caricati.",
    "Usa solo le fonti riportate sotto. Se una risposta non e' supportata dai testi, dichiaralo.",
    "Inserisci citazioni brevi usando gli ID fonte tra parentesi quadre, ad esempio [F1].",
    "Non inventare norme, pagine o paragrafi non presenti.",
    "",
    `Modalita: ${modeText}`,
    "",
    "Documenti disponibili:",
    docList || "- nessun documento",
    "",
    "Fonti rilevanti:",
    sourceBlock || "Nessuna fonte rilevante trovata.",
    "",
    `Domanda: ${question}`,
    "",
    "Formato richiesto: risposta, evidenze citate, limiti/incertezze, prossimi passi."
  ].join("\n");
}
