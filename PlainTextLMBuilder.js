// Configuration constants
const CONFIG = {
  MIN_LOADING_TIME: 3000, // Minimum 3 seconds for loading
  MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
  MIN_FILE_LENGTH: 100,
  ERROR_DISPLAY_TIME: 5000,
  TYPEWRITER_SPEED: 100,
  DEFAULT_MAX_LENGTH: 100,
  DEFAULT_TEMPERATURE: 1.0,
  DEFAULT_NGRAM_SIZE: 3,
  DEFAULT_ALPHA: 0.1,
  EXPLANATION_DELAY: 2000,
  ANIMATION_DURATION: 300,
  STORAGE_KEY: 'plainTextAI_model',
};

// Main language model builder class
class PlainTextLMBuilder {
  constructor(config = {}) {
    this.corpus = "";
    this.model = new Map();
    this.config = {
      ngramSize: config.ngramSize || CONFIG.DEFAULT_NGRAM_SIZE,
      alpha: config.alpha || CONFIG.DEFAULT_ALPHA,
    };
    this.stats = {
      uniqueNgrams: 0,
      vocabularySize: 0,
      totalTokens: 0,
    };
    this.vocabulary = new Set();
    this.vocabularyArray = []; // Cached array for performance
    this.tokenizer = new OptimizedTokenizer();
    this.maxNgramLength = Math.max(3, this.config.ngramSize);
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
          j <= this.maxNgramLength && i + j <= this.corpus.length;
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
  generate(prompt, maxLength = 50, temperature = 1.0) {
    if (!this.model.size) {
      throw new Error("Model has not been trained yet. Please train a model first.");
    }
    
    if (!prompt || prompt.trim().length === 0) {
      throw new Error("Please provide a prompt to generate text.");
    }

    // Clamp temperature to valid range
    temperature = Math.max(0.1, Math.min(2.0, temperature));
    
    let tokens = this.tokenizer.tokenize(prompt.toLowerCase());
    let generated = [];
    let explanations = [];
    let options = [];

    while (generated.length < maxLength) {
      const context = [...tokens, ...generated]
        .slice(-this.maxNgramLength)
        .join(" ");
      const nextTokens = this.getNextTokens(context, temperature);

      const [selectedToken, probability] = this.selectToken(nextTokens);
      generated.push(selectedToken);

      explanations.push(
        `Selected "${selectedToken}" (probability: ${(
          probability * 100
        ).toFixed(2)}%)`
      );
      options.push(nextTokens);

      if ([".", "!", "?"].includes(selectedToken)) {
        break;
      }
    }

    const generatedText = this.tokenizer.detokenize([...tokens, ...generated]);

    return {
      text: generatedText,
      explanations: explanations,
      options: options,
    };
  }

  // Get the next possible tokens based on the given context
  getNextTokens(gram, temperature = 1.0, topK = 10) {
    const gramTokens = gram.split(" ");
    let possibilities;
    
    for (
      let i = Math.min(gramTokens.length, this.maxNgramLength);
      i > 0;
      i--
    ) {
      const subGram = gramTokens.slice(-i).join(" ");
      possibilities = this.model.get(subGram);
      if (possibilities && possibilities.size > 0) {
        break;
      }
    }

    // Use cached vocabulary array
    const vocabArray = this.vocabularyArray;

    if (!possibilities || possibilities.size === 0) {
      // Return random sample from vocabulary with uniform probability
      const sampled = [];
      const used = new Set();
      while (sampled.length < Math.min(topK, vocabArray.length)) {
        const idx = Math.floor(Math.random() * vocabArray.length);
        if (!used.has(idx)) {
          used.add(idx);
          sampled.push([vocabArray[idx], 1 / this.vocabulary.size]);
        }
      }
      return sampled;
    }

    // Calculate total count using a simple loop (faster than reduce)
    let total = 0;
    for (const count of possibilities.values()) {
      total += count;
    }
    
    const adjustedProbabilities = new Map();
    const vocabSize = this.vocabulary.size;
    const alpha = this.config.alpha;
    const denominator = total + alpha * vocabSize;

    for (const [token, count] of possibilities.entries()) {
      const prob = (count + alpha) / denominator;
      adjustedProbabilities.set(token, Math.pow(prob, 1 / temperature));
    }

    // Add random tokens from vocabulary for diversity
    const numRandomTokens = Math.max(2, Math.floor(topK / 4));
    const used = new Set(adjustedProbabilities.keys());
    let added = 0;
    
    while (added < numRandomTokens && used.size < vocabArray.length) {
      const idx = Math.floor(Math.random() * vocabArray.length);
      const token = vocabArray[idx];
      if (!used.has(token)) {
        used.add(token);
        adjustedProbabilities.set(
          token,
          Math.pow(alpha / denominator, 1 / temperature)
        );
        added++;
      }
    }

    // Calculate total adjusted probability using a simple loop
    let totalAdjustedProb = 0;
    for (const prob of adjustedProbabilities.values()) {
      totalAdjustedProb += prob;
    }
    
    const normalizedProbs = Array.from(adjustedProbabilities.entries()).map(
      ([token, prob]) => [token, prob / totalAdjustedProb]
    );

    // Sort by probability and take top K
    return normalizedProbs.sort((a, b) => b[1] - a[1]).slice(0, topK);
  }
  
  // Select a token based on probabilities
  selectToken(tokens) {
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
      this.config = data.config || { ngramSize: CONFIG.DEFAULT_NGRAM_SIZE, alpha: CONFIG.DEFAULT_ALPHA };
      this.stats = data.stats || { uniqueNgrams: 0, vocabularySize: 0, totalTokens: 0 };
      this.maxNgramLength = Math.max(3, this.config.ngramSize);
      
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
    this.initializeElements();
    this.addEventListeners();
    this.checkForSavedModel();
    this.showStep(1);
  }

  // Initialize DOM elements
  initializeElements() {
    this.elements = {
      fileInput: document.getElementById("fileInput"),
      uploadBtn: document.getElementById("uploadBtn"),
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
      explainReasoningBtn: document.getElementById("explainReasoningBtn"),
      regenerateBtn: document.getElementById("regenerateBtn"),
      newPromptBtn: document.getElementById("newPromptBtn"),
      animatedExplanation: document.getElementById("animatedExplanation"),
      generatedText: document.getElementById("generatedText"),
      steps: document.querySelectorAll(".step"),
      loadingContainer: document.getElementById("loadingContainer"),
      fileUploadArea: document.querySelector(".file-upload"),
      savedModelBanner: document.getElementById("savedModelBanner"),
      loadSavedModelBtn: document.getElementById("loadSavedModelBtn"),
      discardModelBtn: document.getElementById("discardModelBtn"),
      saveModelBtn: document.getElementById("saveModelBtn"),
    };
  }

  // Add event listeners to DOM elements
  addEventListeners() {
    this.elements.fileInput.addEventListener("change", (e) =>
      this.handleFileUpload(e.target.files[0])
    );
    this.elements.uploadBtn.addEventListener("click", () =>
      this.elements.fileInput.click()
    );
    this.elements.continueBtn.addEventListener("click", () => this.showStep(2));
    this.elements.generateBtn.addEventListener("click", () =>
      this.generateText()
    );
    this.elements.regenerateBtn.addEventListener("click", () =>
      this.generateText()
    );
    this.elements.newPromptBtn.addEventListener("click", () =>
      this.showStep(2)
    );
    this.elements.explainReasoningBtn.addEventListener("click", () =>
      this.explainReasoning()
    );
    this.elements.temperatureInput.addEventListener("input", () =>
      this.updateTemperatureValue()
    );

    // Model persistence buttons
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

    // Keyboard support for prompt
    this.elements.promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.generateText();
      }
    });

    // Drag and drop functionality
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

  // Check for saved model in localStorage
  checkForSavedModel() {
    try {
      const savedModel = localStorage.getItem(CONFIG.STORAGE_KEY);
      if (savedModel && this.elements.savedModelBanner) {
        this.elements.savedModelBanner.classList.remove("hidden");
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
        this.showError("No saved model found.");
        return;
      }

      const modelData = JSON.parse(savedModel);
      this.llm.deserialize(modelData);
      
      // Hide banner and show success
      if (this.elements.savedModelBanner) {
        this.elements.savedModelBanner.classList.add("hidden");
      }
      this.elements.uploadForm.classList.add("hidden");
      this.elements.modelStats.innerHTML = this.renderStats(this.llm.getStats());
      this.elements.modelStatus.classList.remove("hidden");
      this.elements.modelStatus.classList.add("fade-in");
      
      this.showSuccess("Model loaded successfully!");
    } catch (error) {
      console.error("Error loading saved model:", error);
      this.showError(`Failed to load saved model: ${error.message}`);
    }
  }

  // Save current model to localStorage
  saveModel() {
    try {
      if (!this.llm.model.size) {
        this.showError("No model to save. Please train a model first.");
        return;
      }

      const modelData = this.llm.serialize();
      const serialized = JSON.stringify(modelData);
      
      // Check storage size (localStorage typically has 5-10MB limit)
      if (serialized.length > 4 * 1024 * 1024) {
        this.showError("Model is too large to save. Try training with a smaller text file.");
        return;
      }

      localStorage.setItem(CONFIG.STORAGE_KEY, serialized);
      this.showSuccess("Model saved successfully!");
    } catch (error) {
      console.error("Error saving model:", error);
      if (error.name === 'QuotaExceededError') {
        this.showError("Storage quota exceeded. Try clearing browser data or using a smaller model.");
      } else {
        this.showError(`Failed to save model: ${error.message}`);
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
          this.elements.savedModelBanner.classList.add("hidden");
          this.elements.savedModelBanner.classList.remove("fade-out");
        }, 500);
      }
    } catch (error) {
      console.error("Error discarding saved model:", error);
    }
  }

  // Show a specific step in the UI
  showStep(stepNumber) {
    this.elements.steps.forEach((step, index) => {
      const progressStep = document.querySelector(`.progress-step:nth-child(${index + 1})`);
      if (index + 1 === stepNumber) {
        step.classList.remove("hidden");
        step.classList.add("fade-in");
        if (progressStep) {
          progressStep.classList.add("active");
          progressStep.setAttribute("aria-current", "step");
        }
      } else {
        step.classList.add("hidden");
        step.classList.remove("fade-in");
        if (progressStep) {
          progressStep.classList.remove("active");
          progressStep.removeAttribute("aria-current");
        }
      }
    });
  }

  // Handle file upload with validation
  handleFileUpload(file) {
    if (!file) {
      return;
    }

    // Validate file type
    if (!file.name.endsWith('.txt') && file.type !== 'text/plain') {
      this.showError('Please upload a .txt file');
      return;
    }

    // Warn about large files
    if (file.size > CONFIG.MAX_FILE_SIZE) {
      this.showError('File is too large. Please use a file smaller than 5MB.');
      return;
    }

    const reader = new FileReader();
    
    reader.onerror = () => {
      this.showError('Failed to read file. Please try again.');
    };

    reader.onload = async (e) => {
      const text = e.target.result;
      
      // Validate content
      if (!text || text.trim().length === 0) {
        this.showError('The file appears to be empty.');
        return;
      }

      // Warn about very short files
      if (text.trim().length < CONFIG.MIN_FILE_LENGTH) {
        this.showError('The file is very short. For better results, use a file with more text.');
        return;
      }

      this.showLoadingUI();
      try {
        await this.trainModel(text);
        this.hideLoadingUI();
      } catch (error) {
        this.hideLoadingUI();
        this.showError(error.message || 'An error occurred during training. Please try again.');
        console.error('Training error:', error);
      }
    };

    reader.readAsText(file);
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
    
    // Insert at the top of main
    const main = document.querySelector('main');
    main.insertBefore(notification, main.firstChild);
    
    // Auto-hide after timeout
    setTimeout(() => {
      notification.classList.remove('fade-in');
      notification.classList.add('fade-out');
      setTimeout(() => {
        if (notification.parentNode) {
          notification.remove();
        }
      }, 500);
    }, CONFIG.ERROR_DISPLAY_TIME);
  }

  // Show loading UI during model training
  showLoadingUI() {
    this.trainingStartTime = Date.now();
    
    this.elements.uploadForm.classList.add("fade-out");
    setTimeout(() => {
      this.elements.uploadForm.classList.add("hidden");
      this.elements.progressBarContainer.classList.remove("hidden");
      this.elements.loadingContainer.classList.remove("hidden");
      this.elements.progressBarContainer.classList.add("fade-in");
      this.elements.loadingContainer.classList.add("fade-in");
      this.updateProgressBar(0, "Training model...");
    }, 500);
  }

  // Hide loading UI after model training
  hideLoadingUI() {
    this.elements.progressBarContainer.classList.add("fade-out");
    this.elements.loadingContainer.classList.add("fade-out");
    setTimeout(() => {
      this.elements.progressBarContainer.classList.add("hidden");
      this.elements.loadingContainer.classList.add("hidden");
      this.elements.progressBarContainer.classList.remove("fade-out", "fade-in");
      this.elements.loadingContainer.classList.remove("fade-out", "fade-in");
      this.elements.modelStats.innerHTML = this.renderStats(this.llm.getStats());
      this.elements.modelStatus.classList.remove("hidden");
      this.elements.modelStatus.classList.add("fade-in");
    }, 500);
  }

  // Update progress bar with actual progress
  updateProgressBar(percent, text = "") {
    this.elements.progressBar.style.width = `${percent}%`;
    this.elements.progressBar.setAttribute('aria-valuenow', Math.round(percent));
    if (this.elements.progressText && text) {
      this.elements.progressText.textContent = text;
    }
  }

  // Render model statistics as HTML (UI layer responsibility)
  renderStats(stats) {
    const statItems = [
      {
        name: 'N-gram Size',
        value: stats.ngramSize,
        explanation: 'N-gram size determines the context length used for predictions. Larger sizes capture more context but require more data.'
      },
      {
        name: 'Unique N-grams',
        value: stats.uniqueNgrams.toLocaleString(),
        explanation: 'The number of distinct n-grams in the model. More unique n-grams can lead to more diverse text generation.'
      },
      {
        name: 'Vocabulary Size',
        value: stats.vocabularySize.toLocaleString(),
        explanation: 'The number of unique tokens (words) in the model. A larger vocabulary allows for more expressive text generation.'
      },
      {
        name: 'Total Tokens',
        value: stats.totalTokens.toLocaleString(),
        explanation: 'The total number of tokens (words) processed during training. More tokens generally lead to better model performance.'
      }
    ];

    return statItems.map(item => `
      <div class="stat-item">
        <span class="stat-name">${this.escapeHtml(item.name)}:</span>
        <span class="stat-value">${this.escapeHtml(String(item.value))}</span>
      </div>
      <div class="stat-explanation">${this.escapeHtml(item.explanation)}</div>
    `).join('');
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
      this.updateProgressBar(progress * 0.5, `Training model... ${Math.round(progress)}%`);
    });

    const trainingTime = Date.now() - startTime;
    const remainingTime = CONFIG.MIN_LOADING_TIME - trainingTime;

    // If training was fast, show a "finalizing" phase
    if (remainingTime > 0) {
      this.updateProgressBar(50, "Finalizing model...");
      
      const finalizingSteps = 50;
      const stepDuration = remainingTime / finalizingSteps;
      
      for (let i = 0; i <= finalizingSteps; i++) {
        await new Promise(resolve => setTimeout(resolve, stepDuration));
        const progress = 50 + (i / finalizingSteps) * 50;
        this.updateProgressBar(progress, `Finalizing model... ${Math.round(progress)}%`);
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
      this.showError("Please enter a prompt to generate text.");
      return;
    }

    try {
      this.generatedResult = this.llm.generate(prompt, CONFIG.DEFAULT_MAX_LENGTH, temperature);
      this.showStep(3);
      this.typewriterEffect(this.generatedResult.text);
    } catch (error) {
      this.showError(error.message || "Failed to generate text. Please try again.");
      console.error("Generation error:", error);
    }
  }

  // Display generated text with a typewriter effect
  typewriterEffect(text) {
    const words = text.split(" ");
    let i = 0;

    this.elements.generatedText.textContent = "";
    this.elements.generatedText.style.width = "100%";
    this.elements.generatedText.style.height = "auto";

    // Hide action buttons during typing
    this.elements.explainReasoningBtn.classList.add("hidden");
    this.elements.regenerateBtn.classList.add("hidden");
    this.elements.newPromptBtn.classList.add("hidden");

    const typeWord = () => {
      if (i < words.length) {
        this.elements.generatedText.textContent += words[i] + " ";
        i++;
        this.elements.generatedText.scrollTop =
          this.elements.generatedText.scrollHeight;
        setTimeout(typeWord, CONFIG.TYPEWRITER_SPEED);
      } else {
        this.elements.generatedText.style.height = "auto";
        this.elements.explainReasoningBtn.classList.remove("hidden");
        this.elements.regenerateBtn.classList.remove("hidden");
        this.elements.newPromptBtn.classList.remove("hidden");
      }
    };

    // Set initial height to prevent layout shifts
    this.elements.generatedText.style.height = "250px";
    typeWord();
  }

  // Explain the reasoning behind the generated text
  explainReasoning() {
    if (!this.generatedResult) {
      this.showError("No generated text to explain.");
      return;
    }

    this.elements.animatedExplanation.innerHTML = "";
    this.elements.animatedExplanation.classList.remove("hidden");

    const { text, explanations, options } = this.generatedResult;
    const inputPrompt = this.elements.promptInput.value.trim().toLowerCase();
    const promptEndIndex = this.inferPromptEndIndex(text, inputPrompt);

    const promptText = text.slice(0, promptEndIndex);
    const generatedText = text.slice(promptEndIndex);

    this.displayExplanationText(promptText, generatedText);
    this.animateExplanation(explanations, options);
  }

  // Infer the end index of the prompt in the generated text
  inferPromptEndIndex(text, inputPrompt) {
    const lowerText = text.toLowerCase();
    let promptEndIndex = lowerText.indexOf(inputPrompt) + inputPrompt.length;

    // If the exact prompt is not found, use the first word as the prompt
    if (promptEndIndex <= inputPrompt.length) {
      promptEndIndex = text.indexOf(" ") + 1;
    }

    return promptEndIndex;
  }

  // Display the explanation text with proper formatting
  displayExplanationText(promptText, generatedText) {
    const promptWords = promptText.trim().split(/\s+/);
    const generatedWords = generatedText.trim().split(/\s+/);

    // Add prompt words (not highlighted)
    promptWords.forEach((word) => {
      const span = document.createElement("span");
      span.textContent = word + " ";
      span.classList.add("word", "prompt-word");
      this.elements.animatedExplanation.appendChild(span);
    });

    // Add generated words (to be highlighted)
    generatedWords.forEach((word) => {
      const span = document.createElement("span");
      span.textContent = word + " ";
      span.classList.add("word", "generated-word");
      this.elements.animatedExplanation.appendChild(span);
    });

    const explanationDiv = document.createElement("div");
    explanationDiv.classList.add("explanation");
    explanationDiv.setAttribute("aria-live", "polite");
    this.elements.animatedExplanation.appendChild(explanationDiv);

    const optionsDiv = document.createElement("div");
    optionsDiv.classList.add("options");
    this.elements.animatedExplanation.appendChild(optionsDiv);
  }

  // Animate the explanation of the generated text
  animateExplanation(explanations, options) {
    let currentIndex = 0;

    const animateNextWord = () => {
      if (currentIndex >= explanations.length) return;

      const explanation = explanations[currentIndex];
      const wordOptions = options[currentIndex] || [];

      const match = explanation.match(/Selected "(.*?)"/);
      const selectedWord = match ? match[1] : null;

      const wordSpan = Array.from(
        this.elements.animatedExplanation.querySelectorAll(".generated-word")
      ).find(
        (span) =>
          span.textContent.trim().toLowerCase() === selectedWord.toLowerCase()
      );

      if (wordSpan) {
        this.highlightWord(wordSpan);
      }

      this.displayExplanationAndOptions(explanation, wordOptions, selectedWord);

      setTimeout(() => {
        if (wordSpan) {
          this.unhighlightWord(wordSpan);
        }
        currentIndex++;
        animateNextWord();
      }, CONFIG.EXPLANATION_DELAY);
    };

    animateNextWord();
  }

  // Highlight a word in the explanation (CSS-based)
  highlightWord(wordSpan) {
    wordSpan.classList.add("word-highlighted");
  }

  // Remove highlighting from a word (CSS-based)
  unhighlightWord(wordSpan) {
    wordSpan.classList.remove("word-highlighted");
  }

  // Display the explanation and options for a word
  displayExplanationAndOptions(explanation, wordOptions, selectedWord) {
    const explanationDiv =
      this.elements.animatedExplanation.querySelector(".explanation");
    const optionsDiv =
      this.elements.animatedExplanation.querySelector(".options");

    explanationDiv.textContent = explanation || "No explanation available.";
    optionsDiv.innerHTML = "";
    wordOptions.forEach(([token, prob]) => {
      const optionSpan = document.createElement("span");
      optionSpan.textContent = `${token} (${(prob * 100).toFixed(2)}%)`;
      optionSpan.classList.add("option");
      if (token === selectedWord) {
        optionSpan.classList.add("selected");
      }
      optionsDiv.appendChild(optionSpan);
    });

    // Trigger CSS animations by removing and re-adding class
    explanationDiv.classList.remove("animate-in");
    optionsDiv.classList.remove("animate-in");
    // Force reflow
    void explanationDiv.offsetWidth;
    explanationDiv.classList.add("animate-in");
    optionsDiv.classList.add("animate-in");
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
