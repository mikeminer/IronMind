import os from "node:os";

export const CPU_PERFORMANCE_PROFILES = Object.freeze({
  "low-latency": Object.freeze({
    interactiveContext: 4096,
    maxTokens: 128,
    batch: 128,
    keepAlive: "30m"
  }),
  balanced: Object.freeze({
    interactiveContext: 16384,
    maxTokens: 384,
    batch: 256,
    keepAlive: "30m"
  }),
  "full-context": Object.freeze({
    interactiveContext: null,
    maxTokens: 512,
    batch: 256,
    keepAlive: "30m"
  })
});

export const IRONMIND_CPU_SYSTEM_PROMPT = [
  "Sei Iurexa, assistente virtuale locale specializzata in supporto legale.",
  "Rispondi sempre in italiano, salvo richiesta esplicita diversa.",
  "Agisci con il rigore, l'ordine e l'attenzione ai dettagli di un avvocato esperto.",
  "Lavora con precisione, metodo e prudenza: inquadra i fatti, la giurisdizione, le norme rilevanti, i rischi, le opzioni operative e i prossimi passi.",
  "Puoi aiutare con spiegazioni giuridiche, bozze, checklist, analisi di clausole, lettere, contratti e strategie di preparazione.",
  "Se l'utente non indica la giurisdizione, assumi l'Italia come ipotesi iniziale e dichiaralo brevemente.",
  "Non dichiararti avvocato abilitato, non garantire esiti e non sostituire una consulenza professionale su decisioni vincolanti, scadenze, contenziosi o dati incompleti.",
  "Se mancano paese, materia, date o documenti rilevanti, chiedi prima i dettagli necessari.",
  "In questo runtime Iurexa e CPU-only: non dire mai che viene usata accelerazione GPU.",
  "Preferisci risposte chiare, naturali e concise."
].join(" ");

function numberOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function boolOr(value, fallback) {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value;
  return !["0", "false", "no", "off"].includes(String(value).toLowerCase());
}

export function defaultCpuThreads(parallelism = os.availableParallelism?.() || os.cpus().length || 1) {
  const usable = Math.max(1, Number(parallelism) || 1);
  if (usable <= 4) return usable;
  return Math.max(1, Math.min(12, usable - 2));
}

export function normalizeCpuProfile(profile) {
  return CPU_PERFORMANCE_PROFILES[profile] ? profile : "low-latency";
}

export function resolveCpuPerformanceConfig(config = {}) {
  const profile = normalizeCpuProfile(config.cpuProfile);
  const preset = CPU_PERFORMANCE_PROFILES[profile];
  const cpuOnly = boolOr(config.cpuOnly, true);
  const threads = Math.max(1, Math.floor(numberOr(config.cpuThreads, defaultCpuThreads())));
  const batch = Math.max(1, Math.floor(numberOr(config.cpuBatch, preset.batch)));
  const maxTokens = Math.max(1, Math.floor(numberOr(config.cpuMaxTokens, preset.maxTokens)));
  const interactiveContext = profile === "full-context"
    ? null
    : Math.max(512, Math.floor(numberOr(config.cpuInteractiveCtx, preset.interactiveContext)));

  return {
    profile,
    cpuOnly,
    forceGpuLayers: cpuOnly ? 0 : null,
    threads,
    batch,
    interactiveContext,
    maxTokens,
    keepAlive: config.cpuKeepAlive || preset.keepAlive
  };
}

export function applyCpuPerformanceOptions(config = {}, payload = {}) {
  const performance = resolveCpuPerformanceConfig(config);
  const incomingOptions = payload.options || {};
  const requestedContext = numberOr(incomingOptions.num_ctx, numberOr(payload.ctx, config.ctx || 8192));
  const effectiveContext = performance.interactiveContext
    ? Math.min(requestedContext, performance.interactiveContext)
    : requestedContext;
  const options = {
    ...incomingOptions,
    num_ctx: effectiveContext,
    num_thread: incomingOptions.num_thread ?? performance.threads,
    num_batch: incomingOptions.num_batch ?? performance.batch
  };

  if (performance.cpuOnly) options.num_gpu = 0;
  else if (incomingOptions.num_gpu !== undefined) options.num_gpu = incomingOptions.num_gpu;

  if (payload.temperature === undefined && incomingOptions.temperature === undefined) options.temperature = 0.2;
  if (payload.top_p === undefined && incomingOptions.top_p === undefined) options.top_p = 0.9;
  if (incomingOptions.repeat_penalty === undefined) options.repeat_penalty = 1.05;
  if (
    payload.max_tokens === undefined &&
    payload.max_completion_tokens === undefined &&
    incomingOptions.num_predict === undefined
  ) {
    options.num_predict = performance.maxTokens;
  }

  if (payload.temperature !== undefined) options.temperature = payload.temperature;
  if (payload.top_p !== undefined) options.top_p = payload.top_p;
  if (payload.max_tokens !== undefined) options.num_predict = Number(payload.max_tokens);
  if (payload.max_completion_tokens !== undefined) options.num_predict = Number(payload.max_completion_tokens);

  return {
    options,
    keepAlive: payload.keep_alive || payload.keepAlive || performance.keepAlive,
    performance: {
      ...performance,
      requestedContext,
      effectiveContext
    }
  };
}

export function ensureCpuSystemMessage(messages = []) {
  const hasPolicy = messages.some((message) =>
    message?.role === "system" &&
    String(message.content || "").includes("assistente virtuale locale specializzata in supporto legale")
  );
  if (hasPolicy) return messages;
  return [
    { role: "system", content: IRONMIND_CPU_SYSTEM_PROMPT },
    ...messages
  ];
}
