<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/logo.png" alt="Plain Text AI Builder" width="400" height="auto">
  <br><br>
</div>

Plain Text AI Builder is a web-based application designed for training super-small language models using plain text data. This tool enables users to generate new text based on input prompts with adjustable creativity(temperature) levels. This project is designed as a hobby tool for experimenting with natural language processing (NLP) concepts in an accessible and user-friendly way, and it does not require a web server to run, as it is intended to be usable by anyone regardless of their tech skills.

## How it works

Plain Text AI Builder operates through three main steps: training a model, generating text, and exploring results. The application is implemented entirely in JavaScript and runs within the user's browser. No web server is required.

### Step 1: Training a Model

Users upload a `.txt` file, which serves as the corpus for training the language model. The application tokenizes the uploaded text and creates n-grams (sequences of words used to predict the next word in a sequence), and builds a simple probabilistic model by analyzing the frequency of n-grams in the text.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-1.gif" width="400" height="auto" alt="Step 1">
</div>

### Step 2: Writing a Prompt

After training the model completes, users input a prompt as a starting text that the model will expand upon. The "temperature" setting influences the model's output, where a lower temperature produces more predictable text and a higher temperature generates more varied and creative text.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-2.gif" width="400" height="auto" alt="Step 2">
</div>

### Step 3: Generating Text

The model predicts the next words based on the input prompt and the learned n-gram patterns. 

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-3.gif" width="400" height="auto" alt="Step 3">
</div>

### Step 4: Interactive Review

Users can request explanations for each word choice, including the probabilities and alternative options the model considered during generation, presented through an animated explanation.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-4.gif" width="400" height="auto" alt="Step 4">
</div>

## How to use - Installation and Usage instructions

### Prerequisites

- A modern web browser (e.g., Chrome, Firefox, Edge).
- A `.txt` file containing text data for model training.

### Demo Training Data: Classic Literature

If you don’t have training data on hand, you can download a book in `.txt` format from [Project Gutenberg](https://www.gutenberg.org/), which offers a library of over 70,000 free eBooks. For example:
- [Romeo and Juliet](https://www.gutenberg.org/cache/epub/1513/pg1513.txt)
- [Dracula](https://www.gutenberg.org/cache/epub/345/pg345.txt)

### Installation

No installation or server setup is required. Simply clone or download this repository, and open the `index.html` file in a web browser.

## How the Core / Model Generation Works

### Core Functionality

- **Language Model Builder (`PlainTextLMBuilder`)**: This class handles the core logic for training the model, generating text, and smoothing probabilities. It processes the uploaded text, creates n-grams, and builds a probabilistic model. Text generation uses this model to predict the most likely next word based on the context provided by the prompt.
  
- **Tokenizer (`OptimizedTokenizer`)**: The tokenizer splits the text into words and punctuation, which are then used to create n-grams. It also handles detokenization, converting the sequence of generated words back into readable text with proper punctuation and capitalization.

- **Main Application (`PlainTextAI`)**: This class manages the user interface, handles file uploads, initiates model training, and processes user inputs for text generation. It also controls the display of results and explanations, providing an interactive experience for the user.

### Personalizable Parameters

The application allows for several parameters to be customized in the main JS file, catering to users who wish to experiment with different settings:

1. **N-gram Size (`ngramSize`)**
   - **Description**: Determines the length of n-grams used in the model. Larger n-grams capture more context but require more data.
   - **Default Value**: 3
   - **How to Customize**: Adjusting this value changes the context window size, affecting the model's predictions.

2. **Smoothing Method (`smoothingMethod`)**
   - **Description**: Specifies the technique used to handle unseen n-grams. The default is Laplace smoothing, but developers can implement custom smoothing techniques.
   - **Default Value**: "laplace"
   - **How to Customize**: This can be modified by implementing different smoothing methods within the `PlainTextLMBuilder` class.

3. **Alpha (`alpha`)**
   - **Description**: A smoothing parameter used in Laplace and other smoothing methods. It adjusts the weight given to unseen n-grams.
   - **Default Value**: 0.1
   - **How to Customize**: This parameter can be increased or decreased depending on the desired model behavior. A higher alpha value reduces the impact of unseen n-grams, making the model more conservative.

4. **Temperature**
   - **Description**: Controls the randomness of the model's output. A lower temperature results in more deterministic text, while a higher temperature increases variability and creativity.
   - **Default Value**: 1.0
   - **How to Customize**: Users can adjust the temperature through a slider in the interface. This directly influences the diversity of the generated text.

5. **Maximum Text Length (`maxLength`)**
   - **Description**: Limits the length of the generated text.
   - **Default Value**: 50 tokens
   - **How to Customize**: This can be adjusted when calling the `generate` method, allowing users to generate shorter or longer pieces of text as needed.


## License

This project is licensed under the MIT License. For more details, see the [LICENSE](https://opensource.org/licenses/MIT) file.

## Contributors

Originally created by [Pau Garcia-Mila](https://github.com/paugm).

## Acknowledgments

- **[anime.js](https://animejs.com/)**: Utilized for animations within the user interface.
- **[MIT License](https://opensource.org/licenses/MIT)**: This project is open-source and available under the MIT License.
