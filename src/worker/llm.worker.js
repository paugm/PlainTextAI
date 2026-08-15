/**
 * Loads SmolLM2 and generates tokens off the UI thread.
 * Speaks to LlmClient through the typed worker protocol.
 */
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  LogitsProcessorList,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import { CaptureLogitsProcessor } from "../lib/capture-logits.js";
import {
  MAX_NEW_TOKENS,
  MIN_NEW_TOKENS,
  MODEL_ID,
  NO_REPEAT_NGRAM_SIZE,
  REPETITION_PENALTY,
  TOP_K,
} from "../lib/config.js";
import { withContinuePrefix, withOpeningPrefix } from "../lib/prompt.js";
import { clipStepsToSentence, endsWithSentence } from "../lib/sentences.js";
import { tokenStep } from "../lib/token-step.js";
import {
  assertNever,
  workerCompleteOut,
  workerErrorOut,
  workerProgressOut,
  workerReadyOut,
  workerTokenOut,
  WORKER_IN,
} from "../lib/worker-protocol.js";

/** @typedef {import("../lib/types.js").TokenStep} TokenStep */
/** @typedef {import("../lib/types.js").WorkerGenerateIn} WorkerGenerateIn */
/** @typedef {import("../lib/types.js").WorkerInMessage} WorkerInMessage */
/** @typedef {"wasm" | "webgpu"} InferenceDevice */
/** @typedef {"q4" | "q4f16"} InferenceDtype */

env.allowLocalModels = false;
env.useBrowserCache = true;

/** @type {any} */
let tokenizer = null;
/** @type {any} */
let model = null;
/** @type {InferenceDevice} */
let device = "wasm";
/** @type {InferenceDtype} */
let dtype = "q4";
/** @type {Promise<void> | null} */
let loading = null;
let work = Promise.resolve();
const stopping = new InterruptableStoppingCriteria();

self.addEventListener("message", (event) => {
  const data = /** @type {WorkerInMessage} */ (event.data);
  switch (data.type) {
    case WORKER_IN.ABORT:
      stopping.interrupt();
      return;
    case WORKER_IN.LOAD:
      enqueue(() => load());
      return;
    case WORKER_IN.GENERATE:
      enqueue(() => generate(data));
      return;
    default:
      assertNever(data);
  }
});

/**
 * Run load/generate one at a time so abort cannot overlap a second generate.
 *
 * @param {() => Promise<void>} task
 */
function enqueue(task) {
  work = work.then(async () => {
    try {
      await task();
    } catch (error) {
      self.postMessage(
        workerErrorOut(error instanceof Error ? error.message : "Something went wrong.")
      );
    }
  });
}

async function load() {
  if (model && tokenizer) {
    self.postMessage(
      workerReadyOut({
        device,
        dtype,
        message: "Ready.",
      })
    );
    return;
  }
  if (loading) {
    await loading;
    return;
  }
  loading = loadModel();
  try {
    await loading;
  } finally {
    loading = null;
  }
}

async function loadModel() {
  const attempts = await deviceAttempts();
  /** @type {unknown} */
  let lastError = null;

  for (const attempt of attempts) {
    try {
      self.postMessage(
        workerProgressOut({
          percent: 0,
          message: "Getting ready…",
        })
      );
      tokenizer =
        tokenizer ||
        (await AutoTokenizer.from_pretrained(MODEL_ID, {
          progress_callback: onHubProgress,
        }));
      model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        device: attempt.device,
        dtype: attempt.dtype,
        progress_callback: onHubProgress,
      });
      self.postMessage(
        workerProgressOut({
          percent: 97,
          message: "Warming up…",
        })
      );
      const warmup = tokenizer("Hello");
      await model.generate({ ...warmup, max_new_tokens: 1 });
      device = attempt.device;
      dtype = attempt.dtype;
      self.postMessage(
        workerReadyOut({
          device,
          dtype,
          message: "Ready.",
        })
      );
      return;
    } catch (error) {
      lastError = error;
      model = null;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Could not get ready in this browser.");
}

/** @returns {Promise<Array<{ device: InferenceDevice, dtype: InferenceDtype }>>} */
async function deviceAttempts() {
  const gpu = await webgpuAttempt();
  const cpu = /** @type {const} */ ({ device: "wasm", dtype: "q4" });
  return gpu ? [gpu, cpu] : [cpu];
}

/** @returns {Promise<{ device: InferenceDevice, dtype: InferenceDtype } | null>} */
async function webgpuAttempt() {
  try {
    const gpu = /** @type {{ gpu?: GPU }} */ (navigator).gpu;
    if (!gpu) {
      return null;
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return null;
    }
    const fp16 = adapter.features.has("shader-f16");
    return { device: "webgpu", dtype: fp16 ? "q4f16" : "q4" };
  } catch {
    return null;
  }
}

/**
 * @param {{
 *   status?: string,
 *   progress?: number,
 *   loaded?: number,
 *   total?: number,
 *   file?: string,
 * }} info
 */
function onHubProgress(info) {
  if (info.status === "progress_total") {
    const percent = Number(info.progress) || 0;
    const size = formatSize(info.loaded, info.total);
    self.postMessage(
      workerProgressOut({
        percent,
        loaded: info.loaded,
        total: info.total,
        message: size
          ? `Getting ready… ${Math.round(percent)}% · ${size}`
          : `Getting ready… ${Math.round(percent)}%`,
      })
    );
    return;
  }
  if (info.status === "progress") {
    self.postMessage(
      workerProgressOut({
        message: `Fetching ${shortFileName(info.file)}…`,
      })
    );
  }
}

/**
 * @param {number | undefined} loaded
 * @param {number | undefined} total
 */
function formatSize(loaded, total) {
  if (
    typeof loaded !== "number" ||
    typeof total !== "number" ||
    !Number.isFinite(loaded) ||
    !Number.isFinite(total) ||
    total <= 0
  ) {
    return "";
  }
  return `${mb(loaded)} of ${mb(total)}`;
}

/** @param {number} bytes */
function mb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** @param {string | undefined} file */
function shortFileName(file) {
  if (!file) {
    return "files";
  }
  return String(file).split("/").pop() || "files";
}

/** @param {WorkerGenerateIn} request */
async function generate({ id, messages, prefix, temperature, maxNewTokens, continueFrom }) {
  if (!model || !tokenizer) {
    await load();
  }

  stopping.reset();
  const capture = new CaptureLogitsProcessor();
  const processors = new LogitsProcessorList();
  processors.push(capture);

  const chatPrompt = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  const promptText = continueFrom
    ? withContinuePrefix(chatPrompt, prefix)
    : withOpeningPrefix(chatPrompt, prefix);
  const inputs = tokenizer(promptText, {
    add_special_tokens: false,
  });

  /** @type {TokenStep[]} */
  const steps = [];
  let text = "";
  const specialIds = new Set((tokenizer.all_special_ids || []).map(Number));

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: () => {},
    token_callback_function: (tokens) => {
      const tokenId = Number(tokens[0]);
      if (specialIds.has(tokenId)) {
        capture.pending = null;
        return;
      }
      const step = tokenStep(tokenizer, capture.pending, tokenId);
      capture.pending = null;
      if (!step) {
        return;
      }
      steps.push(step);
      text += step.token;
      self.postMessage(workerTokenOut({ id, step }));
      if (steps.length >= MIN_NEW_TOKENS && endsWithSentence(text)) {
        stopping.interrupt();
      }
    },
  });

  try {
    await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens || MAX_NEW_TOKENS,
      do_sample: true,
      temperature: clampTemperature(temperature),
      top_k: TOP_K,
      repetition_penalty: REPETITION_PENALTY,
      no_repeat_ngram_size: NO_REPEAT_NGRAM_SIZE,
      logits_processor: processors,
      stopping_criteria: stopping,
      streamer,
    });
  } catch (error) {
    if (!stopping.interrupted) {
      throw error;
    }
  }

  const clipped = stopping.interrupted ? { text, steps } : clipStepsToSentence(steps);
  self.postMessage(
    workerCompleteOut({ id, text: clipped.text, steps: clipped.steps })
  );
}

/** @param {number} value */
function clampTemperature(value) {
  const temperature = Number(value);
  if (!Number.isFinite(temperature)) {
    return 1;
  }
  return Math.min(2, Math.max(0.1, temperature));
}
