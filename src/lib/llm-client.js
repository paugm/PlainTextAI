import {
  assertNever,
  CancelledError,
  workerAbortIn,
  workerGenerateIn,
  workerLoadIn,
  WORKER_OUT,
} from "./worker-protocol.js";

/** @typedef {import("./types.js").EngineStatus} EngineStatus */
/** @typedef {import("./types.js").GenerateRequest} GenerateRequest */
/** @typedef {import("./types.js").GenerationResult} GenerationResult */
/** @typedef {import("./types.js").TokenStep} TokenStep */
/** @typedef {import("./types.js").WorkerOutMessage} WorkerOutMessage */

/**
 * Talks to the inference worker. The UI never imports Transformers.js.
 */
export class LlmClient {
  #requestId = 0;
  /** @type {import("./types.js").Deferred<{ device: string | null, dtype: string | null }> | null} */
  #load = null;
  /** @type {import("./types.js").Deferred<GenerationResult> | null} */
  #generate = null;

  /** @type {Worker} */
  worker = new Worker(new URL("../worker/llm.worker.js", import.meta.url), {
    type: "module",
  });
  /** @type {EngineStatus} */
  status = "idle";
  /** @type {string | null} */
  device = null;
  /** @type {string | null} */
  dtype = null;
  progress = 0;
  loaded = 0;
  total = 0;
  message = "";
  /** @type {(() => void) | null} */
  onStatus = null;
  /** @type {((step: TokenStep) => void) | null} */
  onToken = null;

  constructor() {
    this.worker.addEventListener("message", (event) => {
      this.#onMessage(/** @type {WorkerOutMessage} */ (event.data));
    });
    this.worker.addEventListener("error", (event) => {
      this.#fail(event.message || "Something went wrong.");
    });
  }

  get ready() {
    return this.status === "ready";
  }

  load() {
    if (this.ready) {
      return Promise.resolve({ device: this.device, dtype: this.dtype });
    }
    if (this.#load) {
      return this.#load.promise;
    }

    this.status = "loading";
    this.progress = 0;
    this.loaded = 0;
    this.total = 0;
    this.message = "Getting ready…";
    this.#load = deferred();
    this.onStatus?.();
    this.worker.postMessage(workerLoadIn());
    return this.#load.promise;
  }

  /** @param {GenerateRequest} request */
  generate({ messages, prefix, temperature, maxNewTokens, onToken, continueFrom = false }) {
    this.abort();
    const id = ++this.#requestId;
    this.onToken = onToken ?? null;
    this.#generate = deferred();
    this.worker.postMessage(
      workerGenerateIn({
        id,
        messages,
        prefix,
        temperature,
        maxNewTokens,
        continueFrom,
      })
    );
    return this.#generate.promise;
  }

  abort() {
    if (!this.#generate) {
      return;
    }
    this.worker.postMessage(workerAbortIn());
    const pending = this.#generate;
    this.#generate = null;
    this.onToken = null;
    pending.reject(new CancelledError());
  }

  /** @param {WorkerOutMessage} data */
  #onMessage(data) {
    switch (data.type) {
      case WORKER_OUT.PROGRESS:
        this.status = "loading";
        this.progress = data.percent ?? this.progress;
        this.loaded = data.loaded ?? this.loaded;
        this.total = data.total ?? this.total;
        this.message = data.message || this.message;
        this.onStatus?.();
        break;
      case WORKER_OUT.READY:
        this.status = "ready";
        this.device = data.device;
        this.dtype = data.dtype;
        this.progress = 100;
        this.message = data.message || "Ready.";
        this.onStatus?.();
        this.#load?.resolve({ device: data.device, dtype: data.dtype });
        this.#load = null;
        break;
      case WORKER_OUT.TOKEN:
        if (data.id === this.#requestId) {
          this.onToken?.(data.step);
        }
        break;
      case WORKER_OUT.COMPLETE:
        if (data.id === this.#requestId && this.#generate) {
          this.#generate.resolve({
            text: data.text,
            steps: data.steps,
          });
          this.#generate = null;
          this.onToken = null;
        }
        break;
      case WORKER_OUT.ERROR:
        this.#fail(data.message);
        break;
      default:
        assertNever(data);
    }
  }

  /** @param {string} message */
  #fail(message) {
    const generating = this.#generate;
    this.#generate = null;
    this.onToken = null;
    generating?.reject(new Error(message));

    if (generating && this.status === "ready") {
      return;
    }

    this.status = "error";
    this.message = message;
    this.onStatus?.();
    this.#load?.reject(new Error(message));
    this.#load = null;
  }
}

/**
 * @template T
 * @returns {import("./types.js").Deferred<T>}
 */
function deferred() {
  /** @type {(value: T) => void} */
  let resolve = () => {};
  /** @type {(reason?: unknown) => void} */
  let reject = () => {};
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
