import { EXPLANATION_DELAY } from "./config.js";
import { branchSteps, continuePrefix, continuationsFrom } from "./continuations.js";
import { createTokenGraph, drawTokenGraph } from "./token-graph.js";
import { formatToken, splitTokenSpaces } from "./tokens.js";

/** @typedef {import("./types.js").AppElements} AppElements */
/** @typedef {import("./types.js").BranchRequest} BranchRequest */
/** @typedef {import("./types.js").ContinuationOption} ContinuationOption */
/** @typedef {import("./types.js").TokenStep} TokenStep */

/**
 * @typedef {object} WalkthroughState
 * @property {number} id
 * @property {string} prompt
 * @property {TokenStep[]} steps
 * @property {number} index
 * @property {boolean} paused
 * @property {boolean} done
 * @property {number | null} pinIndex
 * @property {ReturnType<typeof setTimeout> | null} timer
 */

/**
 * Live output and next-token walkthrough in one block:
 * the writing, the highlight, and the next pieces that were possible.
 */
export class Walkthrough {
  /** @type {WalkthroughState | null} */
  #state = null;
  #generation = 0;
  #runStart = 0;
  #captureRunStart = true;
  /** @type {import("cytoscape").Core | null} */
  #cy = null;

  /** @type {AppElements} */
  elements;
  /** @type {((event: BranchRequest) => void) | null} */
  onBranch = null;

  /** @param {AppElements} elements */
  constructor(elements) {
    this.elements = elements;
  }

  get isOpen() {
    return Boolean(this.#state) && !this.elements.outputCard.classList.contains("is-hidden");
  }

  stop() {
    this.clearTimer();
    this.destroyGraph();
    this.#state = null;
    this.elements.generatedText.replaceChildren();
    this.elements.explainCaption.textContent = "";
    this.elements.explainCaption.classList.remove("animate-in");
    this.elements.explainProgress.textContent = "";
    this.elements.outputCard.classList.add("is-hidden");
    this.elements.generatedText.classList.remove("is-typing");
  }

  /** @param {string} prompt */
  begin(prompt) {
    this.stop();
    this.#generation += 1;
    this.#runStart = 0;
    this.#captureRunStart = true;
    this.#state = {
      id: this.#generation,
      prompt,
      steps: [],
      index: -1,
      paused: true,
      done: false,
      pinIndex: null,
      timer: null,
    };
    this.renderPrompt(prompt);
    this.elements.outputCard.classList.remove("is-hidden");
  }

  /** @param {TokenStep} step */
  append(step) {
    if (!this.#state) {
      return;
    }
    this.#state.steps.push(step);
    const index = this.#state.steps.length - 1;
    if (this.#captureRunStart) {
      this.#runStart = index;
      this.#captureRunStart = false;
    }
    this.appendWord(step, index);
    this.#state.done = false;
    if (this.#state.pinIndex != null) {
      this.elements.explainProgress.textContent =
        `${this.#state.pinIndex + 1} / ${this.#state.steps.length}`;
      return;
    }
    this.#state.index = index;
    this.renderStep(index, { animate: false });
  }

  finish() {
    if (!this.#state) {
      return;
    }
    this.#state.paused = true;
    this.#state.done = this.#state.steps.length > 0;
    this.#state.pinIndex = null;
    this.clearTimer();
    this.elements.generatedText.classList.remove("is-typing");
    if (this.#state.steps.length === 0) {
      return;
    }
    const start = Math.min(this.#runStart, this.#state.steps.length - 1);
    this.goto(start, { pause: true });
  }

  /**
   * @param {number} count
   * @param {{ silent?: boolean }} [options]
   */
  keepGeneratedCount(count, { silent = false } = {}) {
    if (!this.#state) {
      return;
    }
    const n = Math.max(0, Number(count) || 0);
    if (n >= this.#state.steps.length) {
      return;
    }
    this.#state.steps = this.#state.steps.slice(0, n);
    this.elements.generatedText.querySelectorAll(".generated-word").forEach((span) => {
      if (Number(span.getAttribute("data-index")) >= n) {
        span.remove();
      }
    });
    if (this.#state.index >= n) {
      this.#state.index = n - 1;
    }
    if (this.#state.pinIndex != null && this.#state.pinIndex >= n) {
      this.#state.pinIndex = Math.max(-1, n - 1);
    }
    if (silent) {
      return;
    }
    if (this.#state.steps.length > 0 && this.#state.index >= 0) {
      this.renderStep(this.#state.index, { animate: false });
    } else {
      this.elements.explainCaption.textContent = "";
      this.destroyGraph();
      this.elements.explainProgress.textContent = "";
    }
  }

  pause() {
    if (!this.#state || this.#state.paused) {
      return;
    }
    this.#state.paused = true;
    this.clearTimer();
  }

  /**
   * @param {number} index
   * @param {{ pause?: boolean }} [options]
   */
  goto(index, { pause = false } = {}) {
    if (!this.#state) {
      return;
    }
    this.#state.pinIndex = null;
    const last = this.#state.steps.length;
    if (index >= last) {
      this.#state.index = Math.max(0, last - 1);
      this.#state.done = true;
      this.#state.paused = true;
      this.clearTimer();
      return;
    }

    this.#state.index = index;
    this.#state.done = false;
    if (pause) {
      this.#state.paused = true;
      this.clearTimer();
    }
    this.renderStep(index);
    if (!this.#state.paused && !this.#state.done) {
      this.scheduleAdvance();
    }
  }

  clearTimer() {
    if (this.#state?.timer) {
      clearTimeout(this.#state.timer);
      this.#state.timer = null;
    }
  }

  scheduleAdvance() {
    if (!this.#state) {
      return;
    }
    this.clearTimer();
    const id = this.#state.id;
    this.#state.timer = setTimeout(() => {
      if (!this.#state || this.#state.id !== id) {
        return;
      }
      this.goto(this.#state.index + 1);
    }, EXPLANATION_DELAY);
  }

  /**
   * @param {number} index
   * @param {{ animate?: boolean }} [options]
   */
  renderStep(index, { animate = true } = {}) {
    if (!this.#state) {
      return;
    }
    const step = this.#state.steps[index];
    if (!step) {
      return;
    }
    this.elements.generatedText.querySelectorAll(".generated-word").forEach((span) => {
      const i = Number(span.getAttribute("data-index"));
      span.classList.toggle("word-highlighted", i === index);
      span.classList.toggle("word-visited", i < index);
    });
    this.elements.explainProgress.textContent = `${index + 1} / ${this.#state.steps.length}`;
    this.renderCaption(index, { animate });
  }

  /** @param {string} prompt */
  renderPrompt(prompt) {
    const host = this.elements.generatedText;
    host.replaceChildren();
    if (!prompt.trim()) {
      return;
    }
    const promptSpan = document.createElement("span");
    promptSpan.className = "word prompt-word";
    promptSpan.textContent = `${prompt.trim()} `;
    host.appendChild(promptSpan);
  }

  /**
   * @param {TokenStep} step
   * @param {number} index
   */
  appendWord(step, index) {
    const { leading, core, trailing } = splitTokenSpaces(step.token);
    const span = document.createElement("span");
    span.className = "word generated-word";
    span.setAttribute("data-index", String(index));
    span.title = "See possible next pieces from here";
    span.addEventListener("click", () => this.goto(index, { pause: true }));
    if (leading) {
      span.append(leading);
    }
    const body = document.createElement("span");
    body.className = "token-core";
    body.textContent = core;
    span.appendChild(body);
    if (trailing) {
      span.append(trailing);
    }
    this.elements.generatedText.appendChild(span);
  }

  /**
   * @param {number} index
   * @param {{ animate?: boolean }} [options]
   */
  renderCaption(index, { animate = true } = {}) {
    if (!this.#state) {
      return;
    }
    const caption = this.elements.explainCaption;
    const { from, options } = continuationsFrom(this.#state.steps, index);
    if (!from) {
      caption.textContent = "";
      this.destroyGraph();
      return;
    }
    const fromLabel = formatToken(from.token);
    const chosen = options.find((row) => row.chosen);

    if (options.length === 0) {
      caption.textContent = this.elements.generatedText.classList.contains("is-typing")
        ? `From "${fromLabel}", the next piece has not been written yet.`
        : `The model stopped after "${fromLabel}", so there is no next piece recorded.`;
    } else if (chosen) {
      caption.textContent = `From "${fromLabel}", these were the most likely next pieces. It continued with "${formatToken(chosen.token)}". Click another piece to rewrite from there.`;
    } else {
      caption.textContent = `From "${fromLabel}", these were the most likely next pieces.`;
    }

    this.drawGraph(from.token, options);

    if (!animate) {
      caption.classList.add("animate-in");
      return;
    }

    caption.classList.remove("animate-in");
    // Force a reflow so the animation can restart.
    void caption.offsetWidth;
    caption.classList.add("animate-in");
  }

  /**
   * @param {string} fromToken
   * @param {ContinuationOption[]} options
   */
  drawGraph(fromToken, options) {
    const host = this.elements.explainOptions;
    if (!this.#cy) {
      this.#cy = createTokenGraph(host);
      this.#cy.on("tap", "node", (event) => this.onGraphTap(event));
    }
    drawTokenGraph(this.#cy, { fromToken, options });
    requestAnimationFrame(() => {
      this.#cy?.resize();
      this.#cy?.fit(undefined, 28);
    });
  }

  /** @param {import("cytoscape").EventObject} event */
  onGraphTap(event) {
    const id = event.target.id();
    if (!this.#state || id === "from") {
      return;
    }
    const tokenId = Number(String(id).replace(/^opt-/, ""));
    const { options } = continuationsFrom(this.#state.steps, this.#state.index);
    const option = options.find((row) => Number(row.tokenId) === tokenId) ||
      options.find((row) => `opt-${row.token}` === id);
    if (!option) {
      return;
    }
    if (option.chosen) {
      this.goto(this.#state.index + 1, { pause: true });
      return;
    }
    this.onBranch?.({
      fromIndex: this.#state.index,
      option,
    });
  }

  /**
   * @param {number} fromIndex
   * @param {ContinuationOption} option
   * @returns {{ prefix: string, steps: TokenStep[] } | null}
   */
  applyBranch(fromIndex, option) {
    if (!this.#state) {
      return null;
    }
    this.pause();
    const branched = branchSteps(this.#state.steps, fromIndex, option);
    this.keepGeneratedCount(fromIndex + 1, { silent: true });
    const next = branched[branched.length - 1];
    if (!next) {
      return null;
    }
    this.#state.steps.push(next);
    this.appendWord(next, this.#state.steps.length - 1);
    this.#runStart = this.#state.steps.length - 1;
    this.#captureRunStart = true;
    this.#state.pinIndex = fromIndex;
    this.#state.index = fromIndex;
    this.#state.done = false;
    this.renderStep(fromIndex, { animate: false });
    return {
      prefix: continuePrefix(this.#state.prompt, this.#state.steps),
      steps: this.#state.steps.slice(),
    };
  }

  destroyGraph() {
    if (this.#cy) {
      this.#cy.destroy();
      this.#cy = null;
    }
  }
}
