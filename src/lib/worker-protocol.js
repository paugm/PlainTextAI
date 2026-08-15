/**
 * Typed worker RPC. The page and the worker both import this module so
 * message `type` strings are never invented at the call site.
 */

/** @typedef {import("./types.js").ChatMessage} ChatMessage */
/** @typedef {import("./types.js").TokenStep} TokenStep */
/** @typedef {import("./types.js").WorkerAbortIn} WorkerAbortIn */
/** @typedef {import("./types.js").WorkerCompleteOut} WorkerCompleteOut */
/** @typedef {import("./types.js").WorkerErrorOut} WorkerErrorOut */
/** @typedef {import("./types.js").WorkerGenerateIn} WorkerGenerateIn */
/** @typedef {import("./types.js").WorkerLoadIn} WorkerLoadIn */
/** @typedef {import("./types.js").WorkerProgressOut} WorkerProgressOut */
/** @typedef {import("./types.js").WorkerReadyOut} WorkerReadyOut */
/** @typedef {import("./types.js").WorkerTokenOut} WorkerTokenOut */

export const WORKER_IN = Object.freeze(
  /** @type {const} */ ({
    LOAD: "load",
    GENERATE: "generate",
    ABORT: "abort",
  })
);

export const WORKER_OUT = Object.freeze(
  /** @type {const} */ ({
    PROGRESS: "progress",
    READY: "ready",
    TOKEN: "token",
    COMPLETE: "complete",
    ERROR: "error",
  })
);

/** @returns {WorkerLoadIn} */
export function workerLoadIn() {
  return { type: WORKER_IN.LOAD };
}

/**
 * @param {object} fields
 * @param {number} fields.id
 * @param {ChatMessage[]} fields.messages
 * @param {string} fields.prefix
 * @param {number} fields.temperature
 * @param {number} fields.maxNewTokens
 * @param {boolean} [fields.continueFrom]
 * @returns {WorkerGenerateIn}
 */
export function workerGenerateIn({
  id,
  messages,
  prefix,
  temperature,
  maxNewTokens,
  continueFrom = false,
}) {
  return {
    type: WORKER_IN.GENERATE,
    id,
    messages,
    prefix,
    temperature,
    maxNewTokens,
    continueFrom: Boolean(continueFrom),
  };
}

/** @returns {WorkerAbortIn} */
export function workerAbortIn() {
  return { type: WORKER_IN.ABORT };
}

/**
 * @param {object} [fields]
 * @param {number} [fields.percent]
 * @param {number} [fields.loaded]
 * @param {number} [fields.total]
 * @param {string} [fields.message]
 * @returns {WorkerProgressOut}
 */
export function workerProgressOut({ percent, loaded, total, message } = {}) {
  return { type: WORKER_OUT.PROGRESS, percent, loaded, total, message };
}

/**
 * @param {object} fields
 * @param {string} fields.device
 * @param {string} fields.dtype
 * @param {string} [fields.message]
 * @returns {WorkerReadyOut}
 */
export function workerReadyOut({ device, dtype, message }) {
  return { type: WORKER_OUT.READY, device, dtype, message };
}

/**
 * @param {object} fields
 * @param {number} fields.id
 * @param {TokenStep} fields.step
 * @returns {WorkerTokenOut}
 */
export function workerTokenOut({ id, step }) {
  return { type: WORKER_OUT.TOKEN, id, step };
}

/**
 * @param {object} fields
 * @param {number} fields.id
 * @param {string} fields.text
 * @param {TokenStep[]} fields.steps
 * @returns {WorkerCompleteOut}
 */
export function workerCompleteOut({ id, text, steps }) {
  return { type: WORKER_OUT.COMPLETE, id, text, steps };
}

/**
 * @param {string} message
 * @returns {WorkerErrorOut}
 */
export function workerErrorOut(message) {
  return { type: WORKER_OUT.ERROR, message };
}

/**
 * Generation aborted by the user (Stop, or a newer run).
 */
export class CancelledError extends Error {
  /** @type {true} */
  cancelled = true;

  constructor() {
    super("cancelled");
    this.name = "CancelledError";
  }
}

/**
 * Marks a switch over a worker message union as exhaustive.
 *
 * @param {never} value
 * @returns {never}
 */
export function assertNever(value) {
  throw new Error(`Unhandled value: ${JSON.stringify(value)}`);
}
