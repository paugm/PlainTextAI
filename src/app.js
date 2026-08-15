import {
  DEFAULT_OPENING,
  DEFAULT_TEMPERATURE,
  MAX_FILE_SIZE,
  MAX_NEW_TOKENS,
  MIN_FILE_LENGTH,
} from "./lib/config.js";
import { bindAppElements, conceal, reveal, showToast, wireFileDrop } from "./lib/dom.js";
import { prepareVoice } from "./lib/excerpts.js";
import { LlmClient } from "./lib/llm-client.js";
import { buildMessages } from "./lib/prompt.js";
import { renderSamples } from "./lib/samples-ui.js";
import { Walkthrough } from "./lib/walkthrough.js";
import { CancelledError, assertNever } from "./lib/worker-protocol.js";
import { SAMPLE_CORPORA } from "./samples.js";

/** @typedef {import("./lib/types.js").AppElements} AppElements */
/** @typedef {import("./lib/types.js").BranchRequest} BranchRequest */
/** @typedef {import("./lib/types.js").GenerationResult} GenerationResult */
/** @typedef {import("./lib/types.js").TokenStep} TokenStep */
/** @typedef {import("./lib/types.js").Voice} Voice */

/**
 * Page controller: model download, text choice, generation, and the walkthrough.
 */
export class PlainTextAI {
  #generating = false;
  #runId = 0;

  /** @type {LlmClient} */
  engine = new LlmClient();
  /** @type {Voice | null} */
  voice = null;
  /** @type {GenerationResult | null} */
  generatedResult = null;
  currentStep = 0;
  /** @type {AppElements} */
  elements = bindAppElements();
  /** @type {Walkthrough} */
  walkthrough;

  constructor() {
    this.elements.promptInput.value = DEFAULT_OPENING;
    this.walkthrough = new Walkthrough(this.elements);
    this.walkthrough.onBranch = (event) => {
      void this.continueFromBranch(event);
    };
    renderSamples(this.elements);
    this.bindEvents();
    this.engine.onStatus = () => this.renderEngineStatus();
    this.renderEngineStatus();
    this.showStep(0);
  }

  bindEvents() {
    this.elements.fileInput.addEventListener("change", () => {
      const file = this.elements.fileInput.files?.[0];
      if (file) {
        void this.handleFileUpload(file);
      }
    });
    this.elements.sampleList.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const button = target.closest("[data-sample]");
      if (button instanceof HTMLElement && button.dataset.sample) {
        void this.useSample(button.dataset.sample);
      }
    });
    this.elements.startBtn.addEventListener("click", () => this.onStartClick());
    this.elements.generateBtn.addEventListener("click", () => this.onGenerateClick());
    this.elements.stopBtn.addEventListener("click", () => this.engine.abort());
    this.elements.regenerateBtn.addEventListener("click", () => {
      void this.generateText();
    });
    this.elements.newPromptBtn.addEventListener("click", () => this.focusPrompt());
    this.elements.temperatureInput.addEventListener("input", () => this.updateTemperatureValue());

    document.querySelectorAll("[data-go-step]").forEach((node) => {
      const button = /** @type {HTMLElement} */ (node);
      button.addEventListener("click", (event) => {
        if (button.tagName === "A") {
          event.preventDefault();
        }
        this.closeMobileMenu();
        this.showStep(Number(button.dataset.goStep));
      });
    });

    this.elements.mobileMenuBtn.addEventListener("click", () => this.openMobileMenu());
    this.elements.mobileMenuClose.addEventListener("click", () => this.closeMobileMenu());
    this.elements.mobileMenu.addEventListener("close", () => {
      this.elements.app.inert = false;
      this.elements.mobileMenuBtn.setAttribute("aria-expanded", "false");
    });

    this.elements.promptInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.onGenerateClick();
      }
    });

    wireFileDrop(this.elements.fileUploadArea, (file) => {
      void this.handleFileUpload(file);
    });
  }

  renderEngineStatus() {
    const el = this.elements.engineStatus;
    switch (this.engine.status) {
      case "ready":
        el.textContent = "Ready.";
        break;
      case "error":
        el.textContent = this.engine.message || "Something went wrong. Try again.";
        break;
      case "loading":
        el.textContent = this.engine.message || "Getting ready…";
        break;
      case "idle":
        el.textContent = "Hit Start to begin.";
        break;
      default:
        assertNever(this.engine.status);
    }
    el.classList.toggle("is-hidden", this.engine.status === "loading");

    const percent = this.engine.progress || 0;
    this.elements.progressBar.style.setProperty("--progress", `${percent}%`);
    this.elements.progressBar.setAttribute("aria-valuenow", String(Math.round(percent)));
    this.elements.progressPercent.textContent = `${Math.round(percent)}%`;
    this.elements.progressText.textContent = this.engine.message || "";
    this.elements.progressBarContainer.classList.toggle("is-hidden", this.engine.status !== "loading");
    this.updateStartButton();

    this.updateGenerateEnabled();
    this.updateStudioSummary();
  }

  updateStartButton() {
    const btn = this.elements.startBtn;
    if (this.engine.status === "loading") {
      btn.disabled = true;
      btn.textContent = "Getting ready…";
      return;
    }
    btn.disabled = false;
    if (this.engine.ready) {
      btn.textContent = "Choose a text";
      return;
    }
    if (this.engine.status === "error") {
      btn.textContent = "Try again";
      return;
    }
    btn.textContent = "Start";
  }

  onStartClick() {
    void this.beginDownload({ advance: true });
  }

  /** @param {{ advance?: boolean }} [options] */
  async beginDownload({ advance = false } = {}) {
    try {
      await this.engine.load();
      if (advance && this.currentStep === 0) {
        this.showStep(1);
      }
    } catch (error) {
      this.showError(errorMessage(error, "Something went wrong. Try again."));
    }
  }

  updateGenerateEnabled() {
    this.elements.stopBtn.classList.toggle("is-hidden", !this.#generating);
    if (this.#generating) {
      this.elements.generateBtn.disabled = true;
      this.elements.generateHint.textContent = "";
      return;
    }

    this.elements.generateBtn.textContent = "Launch";
    const ready = Boolean(this.voice) && this.engine.ready;
    this.elements.generateBtn.disabled = !ready;
    if (!this.voice) {
      this.elements.generateHint.textContent = "Pick a text first.";
    } else if (!this.engine.ready) {
      if (this.engine.status === "idle") {
        this.elements.generateHint.textContent = "Hit Start first.";
      } else {
        const percent = Math.round(this.engine.progress || 0);
        this.elements.generateHint.textContent =
          percent > 0
            ? `Getting ready… ${percent}%`
            : this.engine.message || "Getting ready…";
      }
    } else {
      this.elements.generateHint.textContent = "";
    }
  }

  updateStudioSummary() {
    const el = this.elements.studioModelSummary;
    if (!this.voice) {
      el.textContent = "";
      return;
    }
    el.textContent = this.voice.title;
  }

  /** @param {number} stepNumber */
  showStep(stepNumber) {
    let step = stepNumber;
    if (step > 1 && !this.voice) {
      if (this.currentStep !== 0 && this.currentStep !== 1) {
        this.showError("Pick a text first.");
      }
      step = 1;
    }
    if (step === 3 && !this.generatedResult && !this.#generating) {
      step = 2;
    }

    this.currentStep = step;
    if (step < 3 && this.#generating) {
      this.engine.abort();
    }
    conceal(this.elements.introPanel);
    conceal(this.elements.voicePanel);
    conceal(this.elements.writePanel);
    conceal(this.elements.explorePanel);

    if (step === 0) {
      reveal(this.elements.introPanel);
    } else if (step === 1) {
      reveal(this.elements.voicePanel);
      if (this.engine.status === "idle" || this.engine.status === "error") {
        void this.beginDownload();
      }
    } else if (step === 2) {
      reveal(this.elements.writePanel);
    } else {
      reveal(this.elements.explorePanel);
    }

    document.querySelectorAll(".progress-step").forEach((node, index) => {
      const el = /** @type {HTMLElement} */ (node);
      const n = index + 1;
      el.classList.toggle("active", n === step);
      el.classList.toggle("is-complete", n < step);
      if (n === step) {
        el.setAttribute("aria-current", "step");
      } else {
        el.removeAttribute("aria-current");
      }
    });

    if (step === 2) {
      this.elements.promptInput.focus();
    }
    if (step === 3) {
      this.elements.explorePanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (step < 2) {
      this.walkthrough.pause();
    }
  }

  /** @param {File} file */
  async handleFileUpload(file) {
    if (!file.name.endsWith(".txt") && file.type !== "text/plain") {
      this.showError("Use a .txt file.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      this.showError("That file is over 5MB. Try a shorter one.");
      return;
    }

    try {
      const text = await file.text();
      const title = file.name.replace(/\.txt$/i, "") || "Your text";
      this.setVoiceFromText(text, title);
    } catch {
      this.showError("Could not read that file. Try again.");
    }
  }

  /** @param {string} sampleId */
  async useSample(sampleId) {
    const sample = SAMPLE_CORPORA[sampleId];
    if (!sample) {
      this.showError("That sample text is missing.");
      return;
    }
    try {
      const response = await fetch(sample.file);
      if (!response.ok) {
        throw new Error(`Could not load ${sample.file}.`);
      }
      const text = await response.text();
      this.setVoiceFromText(text, sample.title);
    } catch (error) {
      this.showError(errorMessage(error, "Could not load that sample."));
    }
  }

  /**
   * @param {string} text
   * @param {string} title
   */
  setVoiceFromText(text, title) {
    if (!text || text.trim().length < MIN_FILE_LENGTH) {
      this.showError("That file is very short. A few paragraphs work better.");
      return;
    }

    this.voice = prepareVoice(text, { title });
    this.generatedResult = null;
    this.walkthrough.stop();
    this.hideResultActions();
    this.updateStudioSummary();
    this.updateGenerateEnabled();
    this.showStep(2);
  }

  onGenerateClick() {
    void this.generateText();
  }

  async generateText() {
    const prompt = this.elements.promptInput.value.trim();
    if (!prompt) {
      this.showError("Start a sentence first.");
      return;
    }
    if (!this.canGenerate()) {
      return;
    }
    if (this.#generating) {
      return;
    }

    this.walkthrough.begin(prompt);
    await this.runGeneration({
      prefix: prompt,
      continueFrom: false,
      keptSteps: [],
    });
  }

  /** @param {BranchRequest} event */
  async continueFromBranch({ fromIndex, option }) {
    if (!this.canGenerate() || !this.walkthrough.isOpen) {
      return;
    }
    const runId = ++this.#runId;
    this.engine.abort();
    const branched = this.walkthrough.applyBranch(fromIndex, option);
    if (!branched) {
      if (runId === this.#runId) {
        this.#generating = false;
        this.updateGenerateEnabled();
      }
      return;
    }
    await this.runGeneration({
      prefix: branched.prefix,
      continueFrom: true,
      keptSteps: branched.steps,
      runId,
    });
  }

  canGenerate() {
    if (!this.voice) {
      this.showError("Pick a text first.");
      return false;
    }
    if (!this.engine.ready) {
      this.showError("Wait until setup finishes.");
      return false;
    }
    return true;
  }

  /**
   * @param {object} fields
   * @param {string} fields.prefix
   * @param {boolean} fields.continueFrom
   * @param {TokenStep[]} fields.keptSteps
   * @param {number} [fields.runId]
   */
  async runGeneration({ prefix, continueFrom, keptSteps, runId: existingRunId }) {
    const voice = this.voice;
    if (!voice) {
      this.showError("Pick a text first.");
      return;
    }

    const runId = existingRunId ?? ++this.#runId;
    const temperature =
      parseFloat(this.elements.temperatureInput.value) || DEFAULT_TEMPERATURE;

    this.#generating = true;
    this.updateGenerateEnabled();
    this.hideResultActions();
    this.generatedResult = {
      text: keptSteps.map((step) => step.token).join(""),
      steps: keptSteps.slice(),
    };
    this.elements.generatedText.classList.add("is-typing");
    this.elements.generatedText.setAttribute("aria-busy", "true");
    this.showStep(3);

    try {
      const result = await this.engine.generate({
        messages: buildMessages(voice.excerpts),
        prefix,
        continueFrom,
        temperature,
        maxNewTokens: MAX_NEW_TOKENS,
        onToken: (step) => {
          if (runId !== this.#runId) {
            return;
          }
          this.generatedResult?.steps.push(step);
          if (this.generatedResult) {
            this.generatedResult.text += step.token;
          }
          this.walkthrough.append(step);
        },
      });
      if (runId !== this.#runId) {
        return;
      }
      const merged = {
        text: keptSteps.map((step) => step.token).join("") + result.text,
        steps: [...keptSteps, ...result.steps],
      };
      this.generatedResult = merged;
      this.walkthrough.keepGeneratedCount(merged.steps.length);
      this.finishGeneration({ explain: true });
    } catch (error) {
      if (runId !== this.#runId) {
        return;
      }
      if (error instanceof CancelledError) {
        this.finishGeneration({ explain: Boolean(this.generatedResult?.steps?.length) });
        return;
      }
      this.elements.generatedText.classList.remove("is-typing");
      this.elements.generatedText.removeAttribute("aria-busy");
      this.showError(errorMessage(error, "Could not write. Try again."));
      if (!this.generatedResult?.steps?.length) {
        this.showStep(2);
      }
    } finally {
      if (runId === this.#runId) {
        this.#generating = false;
        this.updateGenerateEnabled();
      }
    }
  }

  /** @param {{ explain: boolean }} options */
  finishGeneration({ explain }) {
    this.elements.generatedText.classList.remove("is-typing");
    this.elements.generatedText.removeAttribute("aria-busy");
    this.walkthrough.finish();
    if (!explain) {
      return;
    }
    this.showResultActions();
  }

  showResultActions() {
    this.elements.regenerateBtn.classList.remove("is-hidden");
    this.elements.newPromptBtn.classList.remove("is-hidden");
  }

  hideResultActions() {
    this.elements.regenerateBtn.classList.add("is-hidden");
    this.elements.newPromptBtn.classList.add("is-hidden");
  }

  focusPrompt() {
    this.showStep(2);
    this.elements.promptInput.focus();
    this.elements.promptInput.select();
    this.elements.promptInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  updateTemperatureValue() {
    this.elements.temperatureValue.textContent = this.elements.temperatureInput.value;
  }

  openMobileMenu() {
    this.elements.app.inert = true;
    this.elements.mobileMenuBtn.setAttribute("aria-expanded", "true");
    this.elements.mobileMenu.showModal();
  }

  closeMobileMenu() {
    if (this.elements.mobileMenu.open) {
      this.elements.mobileMenu.close();
    }
  }

  /** @param {string} message */
  showError(message) {
    showToast(this.elements.toastRegion, message);
  }
}

/**
 * @param {unknown} error
 * @param {string} fallback
 */
function errorMessage(error, fallback) {
  return error instanceof Error && error.message ? error.message : fallback;
}
