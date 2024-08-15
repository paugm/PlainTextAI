// Main language model builder class
class PlainTextLMBuilder {
  constructor(config = {}) {
    this.corpus = "";
    this.model = new Map();
    this.config = {
      ngramSize: config.ngramSize || 3,
      smoothingMethod: config.smoothingMethod || "laplace",
      alpha: config.alpha || 0.1,
    };
    this.stats = {
      uniqueNgrams: 0,
      vocabularySize: 0,
      totalTokens: 0,
    };
    this.vocabulary = new Set();
    this.cache = new Map();
    this.memoizedCalculations = new Map();
    this.tokenizer = new OptimizedTokenizer();
    this.properNouns = new Set();
    this.getNextTokens = this.memoize(this.getNextTokens.bind(this));
    this.cacheMaxSize = 1000;
    this.maxNgramLength = Math.max(3, this.config.ngramSize);
  }

  // Train the model with the given text
  async train(text) {
    try {
      this.corpus = this.tokenizer.tokenize(text.toLowerCase());
      this.model.clear();
      this.vocabulary.clear();
      this.stats.totalTokens = this.corpus.length;

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
      }

      this.stats.uniqueNgrams = this.model.size;
      this.stats.vocabularySize = this.vocabulary.size;

    } catch (error) {
      console.error("Error during training:", error);
    }
  }

  // Generate text based on the trained model
  generate(prompt, maxLength = 50, temperature = 1.0) {
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
    let possibilities;
    for (
      let i = Math.min(gram.split(" ").length, this.maxNgramLength);
      i > 0;
      i--
    ) {
      const subGram = gram.split(" ").slice(-i).join(" ");
      possibilities = this.model.get(subGram);
      if (possibilities && possibilities.size > 0) {
        break;
      }
    }

    if (!possibilities || possibilities.size === 0) {
      return Array.from(this.vocabulary)
        .sort(() => Math.random() - 0.5)
        .slice(0, topK)
        .map((token) => [token, 1 / this.vocabulary.size]);
    }

    const total = Array.from(possibilities.values()).reduce(
      (sum, count) => sum + count,
      0
    );
    const adjustedProbabilities = new Map();

    for (const [token, count] of possibilities.entries()) {
      let prob =
        (count + this.config.alpha) /
        (total + this.config.alpha * this.vocabulary.size);
      adjustedProbabilities.set(token, Math.pow(prob, 1 / temperature));
    }

    // Add some randomness to prevent always choosing the same top options
    const randomTokens = Array.from(this.vocabulary)
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.max(2, Math.floor(topK / 4)));

    for (const token of randomTokens) {
      if (!adjustedProbabilities.has(token)) {
        adjustedProbabilities.set(
          token,
          Math.pow(
            this.config.alpha /
              (total + this.config.alpha * this.vocabulary.size),
            1 / temperature
          )
        );
      }
    }

    const totalAdjustedProb = Array.from(adjustedProbabilities.values()).reduce(
      (sum, prob) => sum + prob,
      0
    );
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

  // Memoize function results
  memoize(func) {
    return (...args) => {
      const key = JSON.stringify(args);
      if (this.memoizedCalculations.has(key)) {
        return this.memoizedCalculations.get(key);
      }
      const result = func.apply(this, args);
      this.memoizedCalculations.set(key, result);
      return result;
    };
  }

  // Get model statistics as a formatted string
  getStatsString() {
    return `
      <div class="stat-item">
        <span class="stat-name">N-gram Size:</span>
        <span class="stat-value">${this.config.ngramSize}</span>
      </div>
      <div class="stat-explanation">
        N-gram size determines the context length used for predictions. Larger sizes capture more context but require more data.
      </div>
      <div class="stat-item">
        <span class="stat-name">Unique N-grams:</span>
        <span class="stat-value">${this.stats.uniqueNgrams.toLocaleString()}</span>
      </div>
      <div class="stat-explanation">
        The number of distinct n-grams in the model. More unique n-grams can lead to more diverse text generation.
      </div>
      <div class="stat-item">
        <span class="stat-name">Vocabulary Size:</span>
        <span class="stat-value">${this.stats.vocabularySize.toLocaleString()}</span>
      </div>
      <div class="stat-explanation">
        The number of unique tokens (characters) in the model. A larger vocabulary allows for more expressive text generation.
      </div>
      <div class="stat-item">
        <span class="stat-name">Total Tokens:</span>
        <span class="stat-value">${this.stats.totalTokens.toLocaleString()}</span>
      </div>
      <div class="stat-explanation">
        The total number of tokens (characters) processed during training. More tokens generally lead to better model performance.
      </div>
    `;
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
    this.initializeElements();
    this.addEventListeners();
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

    // Navigation
    document.querySelectorAll("header nav ul li a").forEach((item, index) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        this.showStep(index + 1);
      });
    });
  }

  // Show a specific step in the UI
  showStep(stepNumber) {
    this.elements.steps.forEach((step, index) => {
      if (index + 1 === stepNumber) {
        step.classList.remove("hidden");
        document
          .querySelector(`.progress-step:nth-child(${index + 1})`)
          .classList.add("active");
      } else {
        step.classList.add("hidden");
        document
          .querySelector(`.progress-step:nth-child(${index + 1})`)
          .classList.remove("active");
      }
    });
  }

  // Handle file upload
  handleFileUpload(file) {
    if (file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target.result;
        this.showLoadingUI();
        await this.trainModel(text);
        this.hideLoadingUI();
      };
      reader.readAsText(file);
    }
  }

  // Show loading UI during model training
  showLoadingUI() {
    this.elements.uploadForm.classList.add("fade-out");
    setTimeout(() => {
      this.elements.uploadForm.classList.add("hidden");
      this.elements.progressBarContainer.classList.remove("hidden");
      this.elements.loadingContainer.classList.remove("hidden");
      this.elements.progressBarContainer.classList.add("fade-in");
      this.elements.loadingContainer.classList.add("fade-in");
      this.elements.progressBar.style.width = "0%";
    }, 500);
  }

  // Hide loading UI after model training
  hideLoadingUI() {
    this.elements.progressBarContainer.classList.add("fade-out");
    this.elements.loadingContainer.classList.add("fade-out");
    setTimeout(() => {
      this.elements.progressBarContainer.classList.add("hidden");
      this.elements.loadingContainer.classList.add("hidden");
      this.elements.modelStats.innerHTML = this.llm.getStatsString();
      this.elements.modelStatus.classList.remove("hidden");
      this.elements.modelStatus.classList.add("fade-in");
    }, 500);
  }

  // Train the model with the provided text
  async trainModel(text) {
    const startTime = Date.now();

    // Start progress bar animation
    this.animateProgressBar(1500);

    // Train the model
    await this.llm.train(text);

    const elapsedTime = Date.now() - startTime;
    const remainingTime = Math.max(0, 1500 - elapsedTime);

    // Ensure the progress bar completes its animation
    if (remainingTime > 0) {
      await new Promise((resolve) => setTimeout(resolve, remainingTime));
    }
  }

  // Animate the progress bar
  animateProgressBar(duration) {
    anime({
      targets: this.elements.progressBar,
      width: "100%",
      duration: duration,
      easing: "linear",
    });
  }

  // Generate text based on the user's prompt
  generateText() {
    const prompt = this.elements.promptInput.value;
    const temperature = parseFloat(this.elements.temperatureInput.value) || 1.0;
    if (prompt) {
      this.generatedResult = this.llm.generate(prompt, 100, temperature);
      this.showStep(3);
      this.typewriterEffect(this.generatedResult.text);
    }
  }

  // Display generated text with a typewriter effect
  typewriterEffect(text) {
    const words = text.split(" ");
    let i = 0;
    const speed = 100; // milliseconds per word

    this.elements.generatedText.textContent = "";
    this.elements.generatedText.style.width = "100%";
    this.elements.generatedText.style.height = "auto";

    const typeWord = () => {
      if (i < words.length) {
        this.elements.generatedText.textContent += words[i] + " ";
        i++;
        this.elements.generatedText.scrollTop =
          this.elements.generatedText.scrollHeight;
        setTimeout(typeWord, speed);
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
    if (!this.generatedResult) return;

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
      }, 2000);
    };

    animateNextWord();
  }

  // Highlight a word in the explanation
  highlightWord(wordSpan) {
    anime({
      targets: wordSpan,
      backgroundColor: "#ffeaa7",
      duration: 300,
      easing: "easeInOutQuad",
    });
  }

  // Remove highlighting from a word
  unhighlightWord(wordSpan) {
    anime({
      targets: wordSpan,
      backgroundColor: "rgba(255, 234, 167, 0)",
      duration: 300,
      easing: "easeInOutQuad",
    });
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

    anime({
      targets: [explanationDiv, optionsDiv],
      opacity: [0, 1],
      translateY: [20, 0],
      duration: 300,
      easing: "easeInOutQuad",
    });
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
