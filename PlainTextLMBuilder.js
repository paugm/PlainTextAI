// Configuration constants
const CONFIG = {
  MIN_LOADING_TIME: 600,
  MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
  MIN_FILE_LENGTH: 100,
  ERROR_DISPLAY_TIME: 5000,
  TYPEWRITER_SPEED: 110,
  DEFAULT_MAX_LENGTH: 48,
  MIN_GENERATED_TOKENS: 28,
  MIN_SENTENCES: 3,
  SENTENCE_END_GRACE: 20,
  DEFAULT_TEMPERATURE: 1.0,
  DEFAULT_NGRAM_SIZE: 3,
  DEFAULT_ALPHA: 0.1,
  EXPLANATION_DELAY: 2000,
  ANIMATION_DURATION: 300,
  STORAGE_KEY: 'plainTextAI_model',
};

const SENTENCE_END_TOKENS = new Set([".", "!", "?"]);

// Main language model builder class
class PlainTextLMBuilder {
  constructor(config = {}) {
    this.corpus = "";
    this.model = new Map();
    this.config = {
      ngramSize: Math.max(1, config.ngramSize ?? CONFIG.DEFAULT_NGRAM_SIZE),
      alpha: config.alpha ?? CONFIG.DEFAULT_ALPHA,
    };
    this.stats = {
      uniqueNgrams: 0,
      vocabularySize: 0,
      totalTokens: 0,
    };
    this.vocabulary = new Set();
    this.vocabularyArray = []; // Cached array for performance
    this.tokenizer = new OptimizedTokenizer();
    this.maxNgramLength = this.config.ngramSize;
    this.onProgress = null; // Progress callback
  }

  // Train the model with the given text
  async train(text, onProgress = null) {
    this.onProgress = onProgress;
    
    try {
      this.corpus = this.tokenizer.tokenize(text);
      this.model.clear();
      this.vocabulary.clear();
      this.stats.totalTokens = this.corpus.length;

      const totalIterations = this.corpus.length;
      let lastProgressUpdate = 0;

      for (let i = 0; i < this.corpus.length; i++) {
        for (
          let j = 1;
          j <= this.maxNgramLength && i + j < this.corpus.length;
          j++
        ) {
          const gram = this.corpus.slice(i, i + j).join(" ");
          const nextToken = this.corpus[i + j];

          if (!this.model.has(gram)) {
            this.model.set(gram, new Map());
          }
          const currentCount = this.model.get(gram).get(nextToken) || 0;
          this.model.get(gram).set(nextToken, currentCount + 1);

          this.vocabulary.add(nextToken);
        }

        // Report progress every 1% or at least every 100 iterations
        const progress = ((i + 1) / totalIterations) * 100;
        if (progress - lastProgressUpdate >= 1 || i === totalIterations - 1) {
          lastProgressUpdate = progress;
          if (this.onProgress) {
            this.onProgress(progress);
          }
          // Yield to main thread periodically for UI updates
          if (i % 1000 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }
      }

      // Cache vocabulary array for performance
      this.vocabularyArray = Array.from(this.vocabulary);
      
      this.stats.uniqueNgrams = this.model.size;
      this.stats.vocabularySize = this.vocabulary.size;

    } catch (error) {
      console.error("Error during training:", error);
      throw new Error(`Training failed: ${error.message}`);
    }
  }

  // Generate text based on the trained model
  generate(prompt, maxLength = CONFIG.DEFAULT_MAX_LENGTH, temperature = 1.0) {
    if (!this.model.size) {
      throw new Error("The model has not been trained yet. Pick a text first.");
    }
    
    if (!prompt || prompt.trim().length === 0) {
      throw new Error("Type a prompt first.");
    }

    // Clamp temperature to valid range
    temperature = Math.max(0.1, Math.min(2.0, temperature));
    
    const tokens = this.tokenizer.tokenize(prompt);
    const generated = [];
    const steps = [];
    const hardMax = maxLength + CONFIG.SENTENCE_END_GRACE;
    let sentenceCount = 0;

    while (generated.length < hardMax) {
      const context = [...tokens, ...generated]
        .slice(-this.maxNgramLength)
        .join(" ");
      const next = this.getNextTokens(context, temperature);
      if (!next.tokens.length) {
        break;
      }

      const [selectedToken, probability] = this.selectToken(next.tokens);
      generated.push(selectedToken);
      steps.push({
        token: selectedToken,
        probability,
        alternatives: next.tokens,
        orderUsed: next.orderUsed,
        context: next.context,
        fallback: next.fallback,
      });

      if (!SENTENCE_END_TOKENS.has(selectedToken)) {
        continue;
      }

      sentenceCount += 1;
      const longEnough = generated.length >= CONFIG.MIN_GENERATED_TOKENS;
      const enoughSentences = sentenceCount >= CONFIG.MIN_SENTENCES;
      const hitTarget = generated.length >= maxLength;
      if (longEnough && (enoughSentences || hitTarget)) {
        break;
      }
    }

    const generatedText = this.tokenizer.detokenize([...tokens, ...generated]);

    return {
      text: generatedText,
      steps,
    };
  }

  renormalize(pairs) {
    let mass = 0;
    for (const [, probability] of pairs) {
      mass += probability;
    }
    if (mass <= 0) {
      const uniform = pairs.length ? 1 / pairs.length : 0;
      return pairs.map(([token]) => [token, uniform]);
    }
    return pairs.map(([token, probability]) => [token, probability / mass]);
  }

  // Get the next possible tokens based on the given context
  getNextTokens(gram, temperature = 1.0, topK = 12) {
    const gramTokens = gram.split(" ");
    const scores = new Map();
    let orderUsed = 0;
    let matchedContext = "";

    const maxOrder = Math.min(gramTokens.length, this.maxNgramLength);
    for (let i = 1; i <= maxOrder; i++) {
      const subGram = gramTokens.slice(-i).join(" ");
      const possibilities = this.model.get(subGram);
      if (!possibilities || possibilities.size === 0) {
        continue;
      }

      orderUsed = i;
      matchedContext = subGram;

      let total = 0;
      for (const count of possibilities.values()) {
        total += count;
      }
      if (total <= 0) {
        continue;
      }

      // Longer n-grams weigh more, but shorter ones still vote, so a
      // unique 3-gram does not show up as 100% with no alternatives.
      const weight = i * i;
      for (const [token, count] of possibilities.entries()) {
        scores.set(token, (scores.get(token) || 0) + weight * (count / total));
      }
    }

    const vocabArray = this.vocabularyArray;

    if (scores.size === 0) {
      const sampled = [];
      const used = new Set();
      while (sampled.length < Math.min(topK, vocabArray.length)) {
        const idx = Math.floor(Math.random() * vocabArray.length);
        if (!used.has(idx)) {
          used.add(idx);
          sampled.push([vocabArray[idx], 1]);
        }
      }
      return {
        tokens: this.renormalize(sampled),
        orderUsed: 0,
        context: "",
        fallback: true,
      };
    }

    const adjusted = [];
    for (const [token, score] of scores.entries()) {
      adjusted.push([token, Math.pow(score, 1 / temperature)]);
    }

    const ranked = this.renormalize(adjusted).sort((a, b) => b[1] - a[1]);
    const top = ranked.slice(0, topK);

    return {
      tokens: this.renormalize(top),
      orderUsed,
      context: matchedContext,
      fallback: false,
    };
  }
  
  // Select a token based on probabilities
  selectToken(tokens) {
    if (!tokens || tokens.length === 0) {
      throw new Error("Could not pick a next word.");
    }
    const randomValue = Math.random();
    let cumulativeProbability = 0;
    for (const [token, probability] of tokens) {
      cumulativeProbability += probability;
      if (randomValue <= cumulativeProbability) {
        return [token, probability];
      }
    }
    return tokens[tokens.length - 1];
  }

  // Get model statistics
  getStats() {
    return {
      ngramSize: this.config.ngramSize,
      uniqueNgrams: this.stats.uniqueNgrams,
      vocabularySize: this.stats.vocabularySize,
      totalTokens: this.stats.totalTokens,
    };
  }

  // Serialize model for storage
  serialize() {
    const modelData = {};
    for (const [gram, nextTokens] of this.model.entries()) {
      modelData[gram] = Object.fromEntries(nextTokens);
    }
    
    return {
      model: modelData,
      vocabulary: Array.from(this.vocabulary),
      config: this.config,
      stats: this.stats,
    };
  }

  // Deserialize model from storage
  deserialize(data) {
    try {
      if (!data || !data.model || !data.vocabulary) {
        throw new Error("Invalid model data format");
      }

      this.model.clear();
      for (const [gram, nextTokens] of Object.entries(data.model)) {
        this.model.set(gram, new Map(Object.entries(nextTokens)));
      }
      
      this.vocabulary = new Set(data.vocabulary);
      this.vocabularyArray = Array.from(this.vocabulary);
      this.config = {
        ngramSize: Math.max(1, data.config?.ngramSize ?? CONFIG.DEFAULT_NGRAM_SIZE),
        alpha: data.config?.alpha ?? CONFIG.DEFAULT_ALPHA,
      };
      this.stats = data.stats || { uniqueNgrams: 0, vocabularySize: 0, totalTokens: 0 };
      this.maxNgramLength = this.config.ngramSize;
      
      return true;
    } catch (error) {
      console.error("Error deserializing model:", error);
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }
}

// Optimized tokenizer class
class OptimizedTokenizer {
  constructor() {
    this.wordRegex = /\b[\w']+\b|\S/g;
  }

  tokenize(text) {
    // Remove stage directions and speaker names
    text = text.replace(/\[.*?\]/g, "").replace(/^[A-Z]+\.$/gm, "");

    // Convert to lowercase and split into words
    return text.toLowerCase().match(this.wordRegex) || [];
  }

  detokenize(tokens) {
    let result = tokens.join(" ");

    // Capitalize the first letter of sentences
    result = result.replace(/(^\w|\.\s+\w)/g, (l) => l.toUpperCase());

    // Fix spacing around punctuation
    return result.replace(/ ([.,!?;:])/g, "$1");
  }
}

// Main application class
class PlainTextAI {
  constructor() {
    this.llm = new PlainTextLMBuilder();
    this.generatedResult = null;
    this.trainingStartTime = null;
    this.currentStep = 0;
    this.furthestStep = 0;
    this._typeToken = 0;
    this._explainToken = 0;
    this._explain = null;
    this._cy = null;
    this._cyResizeObs = null;
    this.initializeElements();
    this.addEventListeners();
    this.checkForSavedModel();
    this.showStep(0);
  }

  initializeElements() {
    this.elements = {
      app: document.getElementById("app"),
      fileInput: document.getElementById("fileInput"),
      uploadForm: document.getElementById("uploadForm"),
      progressBarContainer: document.getElementById("progressBarContainer"),
      progressBar: document.getElementById("progressBar"),
      progressText: document.getElementById("progressText"),
      modelStatus: document.getElementById("modelStatus"),
      modelStats: document.getElementById("modelStats"),
      continueBtn: document.getElementById("continueBtn"),
      promptInput: document.getElementById("promptInput"),
      generateBtn: document.getElementById("generateBtn"),
      temperatureInput: document.getElementById("temperatureInput"),
      temperatureValue: document.getElementById("temperatureValue"),
      pauseExplainBtn: document.getElementById("pauseExplainBtn"),
      regenerateBtn: document.getElementById("regenerateBtn"),
      newPromptBtn: document.getElementById("newPromptBtn"),
      retrainBtn: document.getElementById("retrainBtn"),
      animatedExplanation: document.getElementById("animatedExplanation"),
      ngramGraph: document.getElementById("ngramGraph"),
      explainWords: document.getElementById("explainWords"),
      explainCaption: document.getElementById("explainCaption"),
      explainOptions: document.getElementById("explainOptions"),
      explainProgress: document.getElementById("explainProgress"),
      generatedText: document.getElementById("generatedText"),
      trainPanel: document.getElementById("step1"),
      writePanel: document.getElementById("step2"),
      explorePanel: document.getElementById("step3"),
      introPanel: document.getElementById("intro"),
      startBtn: document.getElementById("startBtn"),
      loadingContainer: document.getElementById("loadingContainer"),
      fileUploadArea: document.querySelector(".file-upload"),
      savedModelBanner: document.getElementById("savedModelBanner"),
      loadSavedModelBtn: document.getElementById("loadSavedModelBtn"),
      discardModelBtn: document.getElementById("discardModelBtn"),
      saveModelBtn: document.getElementById("saveModelBtn"),
      studioModelSummary: document.getElementById("studioModelSummary"),
      toastRegion: document.getElementById("toast-region"),
      mobileMenu: document.getElementById("mobileMenu"),
      mobileMenuBtn: document.getElementById("mobileMenuBtn"),
      mobileMenuClose: document.getElementById("mobileMenuClose"),
      trainIntro: document.getElementById("trainIntro"),
    };
  }

  addEventListeners() {
    this.elements.fileInput.addEventListener("change", (e) =>
      this.handleFileUpload(e.target.files[0])
    );
    document.querySelectorAll("[data-sample]").forEach((btn) => {
      btn.addEventListener("click", () => this.trainFromSample(btn.dataset.sample));
    });
    this.elements.continueBtn.addEventListener("click", () => this.showStep(2));
    if (this.elements.startBtn) {
      this.elements.startBtn.addEventListener("click", () => this.showStep(1));
    }
    this.elements.generateBtn.addEventListener("click", () =>
      this.generateText()
    );
    this.elements.regenerateBtn.addEventListener("click", () =>
      this.generateText()
    );
    this.elements.newPromptBtn.addEventListener("click", () =>
      this.focusPrompt()
    );
    this.elements.pauseExplainBtn.addEventListener("click", () =>
      this.toggleExplainPlayback()
    );
    this.elements.temperatureInput.addEventListener("input", () =>
      this.updateTemperatureValue()
    );
    if (this.elements.retrainBtn) {
      this.elements.retrainBtn.addEventListener("click", () => this.retrain());
    }
    const chooseDifferentBtn = document.getElementById("chooseDifferentBtn");
    if (chooseDifferentBtn) {
      chooseDifferentBtn.addEventListener("click", () => this.retrain());
    }

    document.querySelectorAll("[data-go-step]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        if (btn.tagName === "A") {
          e.preventDefault();
        }
        this.closeMobileMenu();
        this.showStep(Number(btn.dataset.goStep));
      });
    });

    if (this.elements.mobileMenuBtn && this.elements.mobileMenu) {
      this.elements.mobileMenuBtn.addEventListener("click", () =>
        this.openMobileMenu()
      );
      this.elements.mobileMenuClose.addEventListener("click", () =>
        this.closeMobileMenu()
      );
      this.elements.mobileMenu.addEventListener("close", () => {
        if (this.elements.app) {
          this.elements.app.inert = false;
        }
        this.elements.mobileMenuBtn.setAttribute("aria-expanded", "false");
      });
    }

    if (this.elements.loadSavedModelBtn) {
      this.elements.loadSavedModelBtn.addEventListener("click", () =>
        this.loadSavedModel()
      );
    }
    if (this.elements.discardModelBtn) {
      this.elements.discardModelBtn.addEventListener("click", () =>
        this.discardSavedModel()
      );
    }
    if (this.elements.saveModelBtn) {
      this.elements.saveModelBtn.addEventListener("click", () =>
        this.saveModel()
      );
    }

    this.elements.promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.generateText();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key !== " " && e.code !== "Space") {
        return;
      }
      if (e.target.closest("input, textarea, button, select, a, [contenteditable]")) {
        return;
      }
      if (this.elements.animatedExplanation.classList.contains("is-hidden")) {
        return;
      }
      e.preventDefault();
      this.toggleExplainPlayback();
    });

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      this.elements.fileUploadArea.addEventListener(
        eventName,
        this.preventDefaults,
        false
      );
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      this.elements.fileUploadArea.addEventListener(
        eventName,
        this.highlight.bind(this),
        false
      );
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.elements.fileUploadArea.addEventListener(
        eventName,
        this.unhighlight.bind(this),
        false
      );
    });

    this.elements.fileUploadArea.addEventListener(
      "drop",
      this.handleDrop.bind(this),
      false
    );
  }

  openMobileMenu() {
    if (!this.elements.mobileMenu) {
      return;
    }
    if (this.elements.app) {
      this.elements.app.inert = true;
    }
    this.elements.mobileMenuBtn.setAttribute("aria-expanded", "true");
    this.elements.mobileMenu.showModal();
  }

  closeMobileMenu() {
    if (this.elements.mobileMenu?.open) {
      this.elements.mobileMenu.close();
    }
  }

  focusPrompt() {
    this.elements.promptInput.focus();
    this.elements.promptInput.select();
    this.elements.promptInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  retrain() {
    this.elements.modelStatus.classList.add("is-hidden");
    this.elements.uploadForm.classList.remove("is-hidden");
    this.elements.progressBarContainer.classList.add("is-hidden");
    this.elements.loadingContainer.classList.add("is-hidden");
    if (this.elements.trainIntro) {
      this.elements.trainIntro.classList.remove("is-hidden");
    }
    this.showStep(1);
  }

  // Check for saved model in localStorage
  checkForSavedModel() {
    try {
      const savedModel = localStorage.getItem(CONFIG.STORAGE_KEY);
      if (savedModel && this.elements.savedModelBanner) {
        this.elements.savedModelBanner.classList.remove("is-hidden");
        this.elements.savedModelBanner.classList.add("fade-in");
      }
    } catch (error) {
      console.warn("Could not access localStorage:", error);
    }
  }

  // Load saved model from localStorage
  loadSavedModel() {
    try {
      const savedModel = localStorage.getItem(CONFIG.STORAGE_KEY);
      if (!savedModel) {
        this.showError("No saved model in this browser.");
        return;
      }

      const modelData = JSON.parse(savedModel);
      this.llm.deserialize(modelData);

      if (this.elements.savedModelBanner) {
        this.elements.savedModelBanner.classList.add("is-hidden");
      }
      this.elements.uploadForm.classList.add("is-hidden");
      if (this.elements.trainIntro) {
        this.elements.trainIntro.classList.add("is-hidden");
      }
      this.elements.modelStats.innerHTML = this.renderStats(this.llm.getStats());
      this.elements.modelStatus.classList.remove("is-hidden");
      this.elements.modelStatus.classList.add("fade-in");
      this.furthestStep = Math.max(this.furthestStep, 2);
      this.updateStudioSummary();
      this.showStep(2);
      this.showSuccess("Loaded the saved model.");
    } catch (error) {
      console.error("Error loading saved model:", error);
      this.showError(`Could not load the saved model: ${error.message}`);
    }
  }

  // Save current model to localStorage
  saveModel() {
    try {
      if (!this.llm.model.size) {
        this.showError("There is nothing to save yet. Train a model first.");
        return;
      }

      const modelData = this.llm.serialize();
      const serialized = JSON.stringify(modelData);
      
      // Check storage size (localStorage typically has 5-10MB limit)
      if (serialized.length > 4 * 1024 * 1024) {
        this.showError("This model is too big to save here. Try a shorter text file.");
        return;
      }

      localStorage.setItem(CONFIG.STORAGE_KEY, serialized);
      this.showSuccess("Saved in this browser.");
    } catch (error) {
      console.error("Error saving model:", error);
      if (error.name === 'QuotaExceededError') {
        this.showError("This browser is out of storage space. Clear some site data, or train on a shorter file.");
      } else {
        this.showError(`Could not save the model: ${error.message}`);
      }
    }
  }

  // Discard saved model from localStorage
  discardSavedModel() {
    try {
      localStorage.removeItem(CONFIG.STORAGE_KEY);
      if (this.elements.savedModelBanner) {
        this.elements.savedModelBanner.classList.add("fade-out");
        setTimeout(() => {
          this.elements.savedModelBanner.classList.add("is-hidden");
          this.elements.savedModelBanner.classList.remove("fade-out");
        }, 500);
      }
    } catch (error) {
      console.error("Error discarding saved model:", error);
    }
  }

  showStep(stepNumber) {
    if (stepNumber > 1 && !this.llm.model.size) {
      if (this.currentStep !== 0 && this.currentStep !== 1) {
        this.showError("Pick a training text first.");
      }
      stepNumber = 1;
    }
    if (stepNumber === 3 && !this.generatedResult) {
      stepNumber = 2;
    }

    this.currentStep = stepNumber;
    this.furthestStep = Math.max(this.furthestStep, stepNumber);

    const reveal = (el) => {
      if (!el) return;
      el.classList.remove("is-hidden");
      el.classList.add("fade-in");
    };
    const conceal = (el) => {
      if (!el) return;
      el.classList.add("is-hidden");
      el.classList.remove("fade-in");
    };

    conceal(this.elements.introPanel);
    conceal(this.elements.trainPanel);
    conceal(this.elements.writePanel);
    conceal(this.elements.explorePanel);

    if (stepNumber === 0) {
      reveal(this.elements.introPanel);
    } else if (stepNumber === 1) {
      reveal(this.elements.trainPanel);
    } else {
      reveal(this.elements.writePanel);
      if (stepNumber === 3 || this.generatedResult) {
        reveal(this.elements.explorePanel);
      }
    }

    document.querySelectorAll(".progress-step").forEach((el, index) => {
      const n = index + 1;
      el.classList.toggle("active", n === stepNumber);
      el.classList.toggle("is-complete", n < stepNumber);
      if (n === stepNumber) {
        el.setAttribute("aria-current", "step");
      } else {
        el.removeAttribute("aria-current");
      }
    });

    if (stepNumber === 2) {
      this.elements.promptInput.focus();
    }
    if (stepNumber === 3 && this.elements.explorePanel) {
      this.elements.explorePanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (stepNumber < 2 && this._explain && !this._explain.paused) {
      this._explain.paused = true;
      if (this._explain.timer) {
        clearTimeout(this._explain.timer);
        this._explain.timer = null;
      }
      this.updatePauseButton();
    }
  }

  // Handle file upload with validation
  handleFileUpload(file) {
    if (!file) {
      return;
    }

    // Validate file type
    if (!file.name.endsWith('.txt') && file.type !== 'text/plain') {
      this.showError('Use a .txt file.');
      return;
    }

    // Warn about large files
    if (file.size > CONFIG.MAX_FILE_SIZE) {
      this.showError('That file is over 5MB. Try a shorter one.');
      return;
    }

    const reader = new FileReader();
    
    reader.onerror = () => {
      this.showError('Could not read that file. Try again.');
    };

    reader.onload = async (e) => {
      const text = e.target.result;
      
      // Validate content
      if (!text || text.trim().length === 0) {
        this.showError('That file looks empty.');
        return;
      }

      // Warn about very short files
      if (text.trim().length < CONFIG.MIN_FILE_LENGTH) {
        this.showError('That file is very short. The model does better with more text.');
        return;
      }

      this.showLoadingUI();
      try {
        await this.trainModel(text);
        this.hideLoadingUI();
      } catch (error) {
        this.hideLoadingUI(false);
        this.showError(error.message || 'Training failed. Try again.');
        console.error('Training error:', error);
      }
    };

    reader.readAsText(file);
  }

  // Train from a demo .txt in the samples/ folder (see samples.js)
  async trainFromSample(sampleId) {
    const sample =
      typeof SAMPLE_CORPORA !== "undefined" ? SAMPLE_CORPORA[sampleId] : null;
    if (!sample || !sample.file) {
      this.showError("That sample text is missing.");
      return;
    }

    let text;
    try {
      text = await this.loadSampleText(sample);
    } catch (error) {
      if (error && error.message) {
        this.showError(error.message);
      }
      return;
    }

    this.showLoadingUI();
    try {
      await this.trainModel(text);
      this.hideLoadingUI();
    } catch (error) {
      this.hideLoadingUI(false);
      this.showError(error.message || "Training failed. Try again.");
      console.error("Training error:", error);
    }
  }

  // Hosted pages can fetch the .txt. file:// cannot (Chrome), so pick the file.
  async loadSampleText(sample) {
    if (window.location.protocol !== "file:") {
      const url = new URL(sample.file, window.location.href);
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Could not load ${sample.file}.`);
      }
      return response.text();
    }

    return this.pickSampleFile(sample);
  }

  pickSampleFile(sample) {
    const filename = sample.file.split("/").pop();
    this.showSuccess(`Choose ${filename} from the samples folder.`);

    return new Promise((resolve, reject) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".txt,text/plain";

      let settled = false;
      const finish = (error, text) => {
        if (settled) {
          return;
        }
        settled = true;
        window.removeEventListener("focus", onWindowFocus);
        if (error) {
          reject(error);
        } else {
          resolve(text);
        }
      };

      const onWindowFocus = () => {
        window.setTimeout(() => {
          if (!input.files || input.files.length === 0) {
            finish(new Error("No file selected."));
          }
        }, 400);
      };

      input.addEventListener(
        "change",
        () => {
          const file = input.files && input.files[0];
          if (!file) {
            finish(new Error("No file selected."));
            return;
          }
          const reader = new FileReader();
          reader.onerror = () =>
            finish(new Error("Could not read that file. Try again."));
          reader.onload = () => finish(null, String(reader.result || ""));
          reader.readAsText(file);
        },
        { once: true }
      );

      window.addEventListener("focus", onWindowFocus, { once: true });
      input.click();
    });
  }

  // Show error message to user
  showError(message) {
    this.showNotification(message, 'error');
  }

  // Show success message to user
  showSuccess(message) {
    this.showNotification(message, 'success');
  }

  // Show notification (error or success)
  showNotification(message, type = 'error') {
    const existingNotification = document.getElementById('notification');
    if (existingNotification) {
      existingNotification.remove();
    }

    const notification = document.createElement('div');
    notification.id = 'notification';
    notification.className = `notification notification-${type} fade-in`;
    notification.setAttribute('role', 'alert');
    notification.setAttribute('aria-live', 'polite');
    notification.textContent = message;

    const region = this.elements.toastRegion || document.querySelector("main");
    region.appendChild(notification);

    setTimeout(() => {
      notification.classList.remove('fade-in');
      notification.classList.add('fade-out');
      setTimeout(() => {
        if (notification.parentNode) {
          notification.remove();
        }
      }, 400);
    }, CONFIG.ERROR_DISPLAY_TIME);
  }

  // Show loading UI during model training
  showLoadingUI() {
    this.trainingStartTime = Date.now();
    this._loadingActive = true;
    
    this.elements.uploadForm.classList.add("fade-out");
    setTimeout(() => {
      if (!this._loadingActive) return;
      this.elements.uploadForm.classList.add("is-hidden");
      this.elements.progressBarContainer.classList.remove("is-hidden");
      this.elements.loadingContainer.classList.remove("is-hidden");
      this.elements.progressBarContainer.classList.add("fade-in");
      this.elements.loadingContainer.classList.add("fade-in");
      this.updateProgressBar(0, "Reading the text...");
    }, 500);
  }

  // Hide loading UI after model training
  hideLoadingUI(success = true) {
    this._loadingActive = false;
    this.elements.progressBarContainer.classList.add("fade-out");
    this.elements.loadingContainer.classList.add("fade-out");
    setTimeout(() => {
      this.elements.progressBarContainer.classList.add("is-hidden");
      this.elements.loadingContainer.classList.add("is-hidden");
      this.elements.progressBarContainer.classList.remove("fade-out", "fade-in");
      this.elements.loadingContainer.classList.remove("fade-out", "fade-in");
      this.elements.uploadForm.classList.remove("fade-out");
      if (!success) {
        this.elements.uploadForm.classList.remove("is-hidden");
        if (this.elements.trainIntro) {
          this.elements.trainIntro.classList.remove("is-hidden");
        }
        return;
      }
      if (this.elements.trainIntro) {
        this.elements.trainIntro.classList.add("is-hidden");
      }
      this.elements.modelStats.innerHTML = this.renderStats(this.llm.getStats());
      this.elements.modelStatus.classList.remove("is-hidden");
      this.elements.modelStatus.classList.add("fade-in");
      this.furthestStep = Math.max(this.furthestStep, 2);
      this.updateStudioSummary();
    }, 500);
  }

  updateProgressBar(percent, text = "") {
    this.elements.progressBar.style.setProperty("--progress", `${percent}%`);
    this.elements.progressBar.setAttribute('aria-valuenow', Math.round(percent));
    if (this.elements.progressText && text) {
      this.elements.progressText.textContent = text;
    }
  }

  updateStudioSummary() {
    if (!this.elements.studioModelSummary || !this.llm.model.size) {
      if (this.elements.studioModelSummary) {
        this.elements.studioModelSummary.textContent = "";
      }
      return;
    }
    const stats = this.llm.getStats();
    this.elements.studioModelSummary.textContent =
      `${stats.totalTokens.toLocaleString()} tokens · ${stats.vocabularySize.toLocaleString()} words it knows`;
  }

  // Render model statistics as HTML (UI layer responsibility)
  renderStats(stats) {
    const statItems = [
      {
        name: 'N-gram size',
        value: stats.ngramSize,
        explanation: 'How many previous words it looks at to guess the next one.'
      },
      {
        name: 'Unique n-grams',
        value: stats.uniqueNgrams.toLocaleString(),
        explanation: 'How many different word chunks it stored.'
      },
      {
        name: 'Vocabulary',
        value: stats.vocabularySize.toLocaleString(),
        explanation: 'How many distinct words it can use.'
      },
      {
        name: 'Tokens trained',
        value: stats.totalTokens.toLocaleString(),
        explanation: 'How many words it read.'
      }
    ];

    return `<dl class="stats-grid">${statItems.map(item => `
      <div class="stat-item">
        <dt class="stat-name">${this.escapeHtml(item.name)}</dt>
        <dd class="stat-value">${this.escapeHtml(String(item.value))}</dd>
        <p class="stat-explanation">${this.escapeHtml(item.explanation)}</p>
      </div>
    `).join('')}</dl>`;
  }

  // Escape HTML to prevent XSS
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Train the model with the provided text
  async trainModel(text) {
    const startTime = Date.now();

    // Train the model with progress callback
    await this.llm.train(text, (progress) => {
      // Training phase is 0-50% of progress bar
      this.updateProgressBar(progress * 0.5, `Reading the text... ${Math.round(progress)}%`);
    });

    const trainingTime = Date.now() - startTime;
    const remainingTime = CONFIG.MIN_LOADING_TIME - trainingTime;

    // If training was fast, show a "finalizing" phase
    if (remainingTime > 0) {
      this.updateProgressBar(50, "Finishing up...");
      
      const finalizingSteps = 50;
      const stepDuration = remainingTime / finalizingSteps;
      
      for (let i = 0; i <= finalizingSteps; i++) {
        await new Promise(resolve => setTimeout(resolve, stepDuration));
        const progress = 50 + (i / finalizingSteps) * 50;
        this.updateProgressBar(progress, `Finishing up... ${Math.round(progress)}%`);
      }
    } else {
      this.updateProgressBar(100, "Complete!");
    }
  }

  // Generate text based on the user's prompt
  generateText() {
    const prompt = this.elements.promptInput.value;
    const temperature = parseFloat(this.elements.temperatureInput.value) || CONFIG.DEFAULT_TEMPERATURE;
    
    if (!prompt || prompt.trim().length === 0) {
      this.showError("Type a prompt first.");
      return;
    }

    try {
      this.stopExplain();
      this.generatedResult = this.llm.generate(prompt, CONFIG.DEFAULT_MAX_LENGTH, temperature);
      this.showStep(3);
      this.typewriterEffect(this.generatedResult.text);
      this.showResultActions();
      this.startExplain();
    } catch (error) {
      this.showError(error.message || "Could not generate text. Try again.");
      console.error("Generation error:", error);
    }
  }

  showResultActions() {
    this.elements.pauseExplainBtn.classList.remove("is-hidden");
    this.elements.regenerateBtn.classList.remove("is-hidden");
    this.elements.newPromptBtn.classList.remove("is-hidden");
  }

  typewriterEffect(text) {
    this._typeToken += 1;
    const token = this._typeToken;
    const words = text.split(" ");

    this.elements.generatedText.textContent = "";
    this.elements.generatedText.classList.remove("is-typing");

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) {
      this.elements.generatedText.textContent = text;
      return;
    }

    this.elements.generatedText.classList.add("is-typing");
    let i = 0;
    const typeWord = () => {
      if (token !== this._typeToken) {
        return;
      }
      if (i < words.length) {
        this.elements.generatedText.textContent += words[i] + " ";
        i += 1;
        setTimeout(typeWord, CONFIG.TYPEWRITER_SPEED);
      } else {
        this.elements.generatedText.classList.remove("is-typing");
      }
    };
    typeWord();
  }

  prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  stopExplain() {
    if (this._explain?.timer) {
      clearTimeout(this._explain.timer);
    }
    this._explain = null;
    this.destroyGraph();
    if (this.elements.explainWords) {
      this.elements.explainWords.replaceChildren();
    }
    if (this.elements.explainCaption) {
      this.elements.explainCaption.textContent = "";
      this.elements.explainCaption.classList.remove("animate-in");
    }
    if (this.elements.explainOptions) {
      this.elements.explainOptions.replaceChildren();
      this.elements.explainOptions.classList.remove("animate-in");
    }
    if (this.elements.explainProgress) {
      this.elements.explainProgress.textContent = "";
    }
    this.elements.animatedExplanation.classList.add("is-hidden");
  }

  startExplain() {
    if (!this.generatedResult) {
      return;
    }

    this._explainToken += 1;
    const id = this._explainToken;
    const steps = this.generatedResult.steps || [];
    const reduceMotion = this.prefersReducedMotion();

    this._explain = {
      id,
      steps,
      index: 0,
      paused: reduceMotion,
      done: steps.length === 0,
      timer: null,
    };

    this.displayExplanationWords(this.elements.promptInput.value, steps);

    this.elements.animatedExplanation.classList.remove("is-hidden");
    this.updatePauseButton();

    if (steps.length === 0) {
      return;
    }

    this.renderExplainStep(0);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (!this._explain || this._explain.id !== id) {
          return;
        }
        this.ensureGraph();
        this.updateGraph(this._explain.steps[this._explain.index]);
      });
    });

    if (!this._explain.paused && !this._explain.done) {
      this.scheduleExplainAdvance();
    }
  }

  scheduleExplainAdvance() {
    if (!this._explain) {
      return;
    }
    if (this._explain.timer) {
      clearTimeout(this._explain.timer);
    }
    const id = this._explain.id;
    this._explain.timer = setTimeout(() => {
      if (!this._explain || this._explain.id !== id) {
        return;
      }
      this.gotoExplainStep(this._explain.index + 1);
    }, CONFIG.EXPLANATION_DELAY);
  }

  gotoExplainStep(index, options = {}) {
    if (!this._explain) {
      return;
    }
    const { pause = false } = options;
    const last = this._explain.steps.length;

    if (index >= last) {
      this._explain.index = Math.max(0, last - 1);
      this._explain.done = true;
      this._explain.paused = true;
      if (this._explain.timer) {
        clearTimeout(this._explain.timer);
        this._explain.timer = null;
      }
      this.updatePauseButton();
      return;
    }

    this._explain.index = index;
    this._explain.done = false;
    if (pause) {
      this._explain.paused = true;
      if (this._explain.timer) {
        clearTimeout(this._explain.timer);
        this._explain.timer = null;
      }
    }
    this.renderExplainStep(index);
    this.updateGraph(this._explain.steps[index]);
    this.updatePauseButton();
    if (!this._explain.paused && !this._explain.done) {
      this.scheduleExplainAdvance();
    }
  }

  toggleExplainPlayback() {
    if (!this._explain || !this._explain.steps.length) {
      return;
    }

    if (this._explain.done) {
      this._explain.done = false;
      this._explain.paused = false;
      this.gotoExplainStep(0);
      return;
    }

    this._explain.paused = !this._explain.paused;
    if (this._explain.paused) {
      if (this._explain.timer) {
        clearTimeout(this._explain.timer);
        this._explain.timer = null;
      }
    } else {
      this.scheduleExplainAdvance();
    }
    this.updatePauseButton();
  }

  updatePauseButton() {
    const btn = this.elements.pauseExplainBtn;
    if (!btn) {
      return;
    }
    if (!this._explain) {
      btn.textContent = "Pause";
      btn.setAttribute("aria-pressed", "false");
      return;
    }
    if (this._explain.done) {
      btn.textContent = "Play again";
      btn.setAttribute("aria-pressed", "false");
      return;
    }
    if (this._explain.paused) {
      btn.textContent = "Resume";
      btn.setAttribute("aria-pressed", "true");
      return;
    }
    btn.textContent = "Pause";
    btn.setAttribute("aria-pressed", "false");
  }

  renderExplainStep(index) {
    const step = this._explain.steps[index];
    if (!step) {
      return;
    }

    const words = this.elements.explainWords.querySelectorAll(".generated-word");
    words.forEach((wordSpan) => {
      const i = Number(wordSpan.getAttribute("data-index"));
      wordSpan.classList.toggle("word-highlighted", i === index);
      wordSpan.classList.toggle("word-visited", i < index);
    });

    if (this.elements.explainProgress) {
      this.elements.explainProgress.textContent = `${index + 1} / ${this._explain.steps.length}`;
    }

    this.displayExplanationAndOptions(
      this.formatStepExplanation(step),
      step.alternatives || [],
      step.token
    );
  }

  formatStepExplanation(step) {
    const pct = (step.probability * 100).toFixed(2);
    if (step.fallback) {
      return `Selected "${step.token}" (${pct}%). No matching n-gram, so this was drawn from the whole vocabulary.`;
    }
    const ngramLabel = `${step.orderUsed}-gram`;
    if (step.context) {
      return `Selected "${step.token}" (${pct}%) using a ${ngramLabel} on "${step.context}".`;
    }
    return `Selected "${step.token}" (${pct}%) using a ${ngramLabel}.`;
  }

  explainTokenPrefix(token, isFirst) {
    if (isFirst) {
      return "";
    }
    if (/^[.,!?;:]+$/.test(token)) {
      return "";
    }
    return " ";
  }

  displayExplanationWords(prompt, steps) {
    const host = this.elements.explainWords;
    host.replaceChildren();

    const promptTokens = this.llm.tokenizer.tokenize(prompt || "");
    promptTokens.forEach((token, i) => {
      const span = document.createElement("span");
      span.textContent = this.explainTokenPrefix(token, i === 0) + token;
      span.classList.add("word", "prompt-word");
      host.appendChild(span);
    });

    const promptEnded = promptTokens.length > 0;
    steps.forEach((step, index) => {
      const span = document.createElement("span");
      const isFirst = !promptEnded && index === 0;
      span.textContent = this.explainTokenPrefix(step.token, isFirst) + step.token;
      span.classList.add("word", "generated-word");
      span.setAttribute("data-index", String(index));
      span.title = "Jump to this word";
      span.addEventListener("click", () => this.gotoExplainStep(index, { pause: true }));
      host.appendChild(span);
    });
  }

  displayExplanationAndOptions(explanation, wordOptions, selectedWord) {
    const explanationDiv = this.elements.explainCaption;
    const optionsDiv = this.elements.explainOptions;
    if (!explanationDiv || !optionsDiv) {
      return;
    }

    explanationDiv.textContent = explanation || "No explanation available.";
    optionsDiv.replaceChildren();
    wordOptions.forEach(([token, prob]) => {
      const row = document.createElement("div");
      row.className = "option-row";
      if (token === selectedWord) {
        row.classList.add("is-selected");
      }

      const label = document.createElement("span");
      label.className = "option-label";
      label.textContent = token;

      const track = document.createElement("div");
      track.className = "option-track";
      const bar = document.createElement("div");
      bar.className = "option-bar";
      bar.style.setProperty("--bar", `${(prob * 100).toFixed(1)}%`);
      track.appendChild(bar);

      const pct = document.createElement("span");
      pct.className = "option-pct";
      pct.textContent = `${(prob * 100).toFixed(1)}%`;

      row.append(label, track, pct);
      optionsDiv.appendChild(row);
    });

    explanationDiv.classList.remove("animate-in");
    optionsDiv.classList.remove("animate-in");
    void explanationDiv.offsetWidth;
    explanationDiv.classList.add("animate-in");
    optionsDiv.classList.add("animate-in");
  }

  graphTheme() {
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    if (dark) {
      return {
        label: "#f4f4f5",
        context: "#27272a",
        contextBorder: "#52525b",
        alt: "#18181b",
        altBorder: "#3f3f46",
        chosen: "#ff8904",
        chosenLabel: "#09090b",
        fallback: "#3f3f46",
        edge: "rgba(255,255,255,0.18)",
        edgeChosen: "#ff8904",
      };
    }
    return {
      label: "#09090b",
      context: "#f4f4f5",
      contextBorder: "#d4d4d8",
      alt: "#ffffff",
      altBorder: "#e4e4e7",
      chosen: "#c2410c",
      chosenLabel: "#ffffff",
      fallback: "#e4e4e7",
      edge: "rgba(9,9,11,0.16)",
      edgeChosen: "#c2410c",
    };
  }

  graphStylesheet() {
    const t = this.graphTheme();
    return [
      {
        selector: "node",
        style: {
          label: "data(label)",
          color: t.label,
          "font-family": "VT323, ui-monospace, monospace",
          "font-size": 16,
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "wrap",
          "text-max-width": 72,
          width: 36,
          height: 36,
          "border-width": 1,
          "overlay-padding": 4,
        },
      },
      {
        selector: 'node[kind = "context"]',
        style: {
          shape: "round-rectangle",
          width: 52,
          height: 32,
          "background-color": t.context,
          "border-color": t.contextBorder,
        },
      },
      {
        selector: 'node[kind = "fallback"]',
        style: {
          shape: "diamond",
          "background-color": t.fallback,
          "border-color": t.contextBorder,
          width: 40,
          height: 40,
        },
      },
      {
        selector: 'node[kind = "alt"]',
        style: {
          "background-color": t.alt,
          "border-color": t.altBorder,
          width: "mapData(prob, 0, 0.6, 28, 56)",
          height: "mapData(prob, 0, 0.6, 28, 56)",
        },
      },
      {
        selector: 'node[kind = "chosen"]',
        style: {
          "background-color": t.chosen,
          "border-color": t.chosen,
          color: t.chosenLabel,
          width: "mapData(prob, 0, 0.6, 36, 64)",
          height: "mapData(prob, 0, 0.6, 36, 64)",
          "border-width": 2,
        },
      },
      {
        selector: "edge",
        style: {
          width: "mapData(weight, 0, 0.6, 1, 7)",
          "line-color": t.edge,
          "target-arrow-color": t.edge,
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 0.8,
        },
      },
      {
        selector: 'edge[kind = "context"]',
        style: {
          width: 2,
          "line-color": t.contextBorder,
          "target-arrow-color": t.contextBorder,
          "curve-style": "straight",
        },
      },
      {
        selector: 'edge[kind = "chosen"]',
        style: {
          "line-color": t.edgeChosen,
          "target-arrow-color": t.edgeChosen,
          width: "mapData(weight, 0, 0.6, 3, 8)",
        },
      },
    ];
  }

  destroyGraph() {
    if (this._cyResizeObs) {
      this._cyResizeObs.disconnect();
      this._cyResizeObs = null;
    }
    if (this._graphSchemeListener && this._onGraphScheme) {
      this._graphSchemeListener.removeEventListener("change", this._onGraphScheme);
      this._graphSchemeListener = null;
      this._onGraphScheme = null;
    }
    if (this._cy) {
      this._cy.destroy();
      this._cy = null;
    }
  }

  ensureGraph() {
    const container = this.elements.ngramGraph;
    if (!container) {
      return;
    }
    if (typeof window.cytoscape !== "function") {
      container.classList.add("is-hidden");
      return;
    }
    container.classList.remove("is-hidden");
    if (this._cy) {
      this._cy.resize();
      return;
    }

    this._cy = window.cytoscape({
      container,
      elements: [],
      style: this.graphStylesheet(),
      minZoom: 0.35,
      maxZoom: 2.6,
      wheelSensitivity: 0.25,
      boxSelectionEnabled: false,
      autounselectify: true,
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    this._cyResizeObs = new ResizeObserver(() => {
      if (this._cy) {
        this._cy.resize();
      }
    });
    this._cyResizeObs.observe(container);

    this._onGraphScheme = () => {
      if (!this._cy) {
        return;
      }
      this._cy.style().fromJson(this.graphStylesheet()).update();
    };
    this._graphSchemeListener = window.matchMedia("(prefers-color-scheme: dark)");
    if (this._graphSchemeListener.addEventListener) {
      this._graphSchemeListener.addEventListener("change", this._onGraphScheme);
    }
  }

  graphElementsForStep(step) {
    const elements = [];
    const contextTokens = step.context ? step.context.split(" ").filter(Boolean) : [];
    const alts = step.alternatives || [];

    if (contextTokens.length === 0) {
      elements.push({
        group: "nodes",
        data: { id: "root", label: "?", kind: "fallback" },
        position: { x: 80, y: 160 },
      });
    } else {
      contextTokens.forEach((token, i) => {
        elements.push({
          group: "nodes",
          data: { id: `c${i}`, label: token, kind: "context" },
          position: { x: 70 + i * 108, y: 70 },
        });
        if (i > 0) {
          elements.push({
            group: "edges",
            data: {
              id: `ce${i}`,
              source: `c${i - 1}`,
              target: `c${i}`,
              kind: "context",
              weight: 0.4,
            },
          });
        }
      });
    }

    const origin = contextTokens.length ? `c${contextTokens.length - 1}` : "root";
    const originX = contextTokens.length ? 70 + (contextTokens.length - 1) * 108 : 80;

    alts.forEach(([token, prob], i) => {
      const chosen = token === step.token;
      const t = alts.length === 1 ? 0.5 : i / (alts.length - 1);
      elements.push({
        group: "nodes",
        data: {
          id: `a${i}`,
          label: token,
          kind: chosen ? "chosen" : "alt",
          prob,
        },
        position: { x: originX + 180, y: 36 + t * 260 },
      });
      elements.push({
        group: "edges",
        data: {
          id: `e${i}`,
          source: origin,
          target: `a${i}`,
          kind: chosen ? "chosen" : "alt",
          weight: prob,
        },
      });
    });

    return elements;
  }

  updateGraph(step) {
    if (!this._cy || !step) {
      return;
    }
    const reduceMotion = this.prefersReducedMotion();
    this._cy.elements().remove();
    this._cy.add(this.graphElementsForStep(step));

    const contextNodes = this._cy.nodes('[kind = "context"], [kind = "fallback"]');
    contextNodes.lock();
    this._cy.nodes('[kind = "alt"], [kind = "chosen"]').unlock();

    this._cy.layout({
      name: "cose",
      animate: !reduceMotion,
      animationDuration: 700,
      randomize: false,
      padding: 28,
      fit: true,
      nodeRepulsion: 9000,
      idealEdgeLength: 84,
      edgeElasticity: 80,
      gravity: 0.35,
      numIter: 600,
      initialTemp: 200,
      minTemp: 1,
    }).run();
  }

  // Update the temperature value display
  updateTemperatureValue() {
    this.elements.temperatureValue.textContent =
      this.elements.temperatureInput.value;
  }

  // Prevent default behavior for drag and drop events
  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Highlight the file upload area during drag
  highlight() {
    this.elements.fileUploadArea.classList.add("highlight");
  }

  // Remove highlight from the file upload area
  unhighlight() {
    this.elements.fileUploadArea.classList.remove("highlight");
  }

  // Handle file drop event
  handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    this.handleFileUpload(file);
  }
}

// Initialize the application when the DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  new PlainTextAI();
});
