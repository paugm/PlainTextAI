<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/logo.png" alt="Plain Text AI Builder" width="400" height="auto">
  <br><br>
</div>

Plain Text AI Builder is a web-based application designed for training super-small language models using plain text data. This tool enables users to generate new text based on input prompts with adjustable creativity(temperature) levels. This project is designed as a hobby tool for experimenting with natural language processing (NLP) concepts in an accessible and user-friendly way, and it does not require a web server to run, as it is intended to be usable by anyone regardless of their tech skills.

## How it works

Plain Text AI Builder operates through three main steps: training a model, generating text, and exploring results. The application is implemented entirely in JavaScript and runs within the user's browser. No web server is required.

### Step 1: Training a Model

Users upload a `.txt` file, or pick a public-domain sample from the `samples/` folder, which serves as the corpus for training the language model. The application tokenizes the uploaded text and creates n-grams (sequences of words used to predict the next word in a sequence), and builds a simple probabilistic model by analyzing the frequency of n-grams in the text.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-1.gif" width="400" height="auto" alt="Step 1">
</div>

### Step 2: Writing a Prompt

After training the model completes, users input a prompt as a starting text that the model will expand upon. The "temperature" setting influences the model's output, where a lower temperature produces more predictable text and a higher temperature generates more varied and creative text.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-2.gif" width="400" height="auto" alt="Step 2">
</div>

### Step 3: Generating and Exploring Text

The model predicts the next words based on the input prompt and the learned n-gram patterns. After it writes, a walkthrough starts on its own: each chosen word, the alternatives, and a small graph of the n-gram that fired. You can pause it, or click a word to jump there.

<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/demo-images/Step-3.gif" width="400" height="auto" alt="Step 3">
</div>

## How to use - Installation and Usage instructions

### Prerequisites

- A modern web browser (e.g., Chrome, Firefox, Edge).
- Optional: a `.txt` file to train on. If you don’t have one, use a public-domain sample from the `samples/` folder.

### Demo Training Data: Classic Literature

If you don’t have a file on hand, use the **public-domain samples** in the `samples/` folder (also listed on Step 1):

- **Romeo and Juliet** by William Shakespeare (complete play). Public domain. Obtained from [Project Gutenberg eBook #1513](https://www.gutenberg.org/ebooks/1513) ([plain text](https://www.gutenberg.org/files/1513/1513-0.txt)). File: `samples/romeo-and-juliet.txt`.
- **Dracula** by Bram Stoker, Chapters I–V. Public domain in the United States (first published 1897). Obtained from [Project Gutenberg eBook #345](https://www.gutenberg.org/ebooks/345) ([plain text](https://www.gutenberg.org/files/345/345-0.txt)). File: `samples/dracula-chapters-i-v.txt`.

The copies omit Project Gutenberg headers and licenses, so they are **not** redistributed as Project Gutenberg eBooks and do not use that trademark. You can still upload any `.txt` file of your own, or download other books from [Project Gutenberg](https://www.gutenberg.org/).

If you open `index.html` as a file (double-click), most browsers will not fetch those texts automatically. Click the sample you want, then choose the matching `.txt` from the `samples/` folder. Served over http, a click loads the file on its own.

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

2. **Alpha (`alpha`)**
   - **Description**: A smoothing parameter used in Laplace and other smoothing methods. It adjusts the weight given to unseen n-grams.
   - **Default Value**: 0.1
   - **How to Customize**: This parameter can be increased or decreased depending on the desired model behavior. A higher alpha value reduces the impact of unseen n-grams, making the model more conservative.

3. **Temperature**
   - **Description**: Controls the randomness of the model's output. A lower temperature results in more deterministic text, while a higher temperature increases variability and creativity.
   - **Default Value**: 1.0
   - **How to Customize**: Users can adjust the temperature through a slider in the interface. This directly influences the diversity of the generated text.

4. **Maximum Text Length (`maxLength`)**
   - **Description**: Caps how many tokens the model will add. Generation also stops at the first `.` `!` or `?` once a short minimum is reached, so output tends to end on a sentence.
   - **Default Value**: 40 tokens (with a little extra room to reach a period if needed)
   - **How to Customize**: This can be adjusted when calling the `generate` method, allowing users to generate shorter or longer pieces of text as needed.


## License

This project is licensed under the MIT License. For more details, see the [LICENSE](https://opensource.org/licenses/MIT) file.

## Contributors

Originally created by [Pau Garcia-Mila](https://github.com/paugm).

## Acknowledgments

- **[MIT License](https://opensource.org/licenses/MIT)**: This project is open-source and available under the MIT License.
- **Sample texts**: William Shakespeare, *Romeo and Juliet* (public domain); Bram Stoker, *Dracula* (public domain in the United States). Both obtained from [Project Gutenberg](https://www.gutenberg.org/) (eBooks [#1513](https://www.gutenberg.org/ebooks/1513) and [#345](https://www.gutenberg.org/ebooks/345)). See `samples.js` and the files in `samples/` for source notes.
- **VT323 font**: Peter Hull / The VT323 Project Authors, licensed under the [SIL Open Font License 1.1](fonts/OFL.txt), self-hosted from [Google Fonts](https://fonts.google.com/specimen/VT323).
- **[Cytoscape.js](https://js.cytoscape.org/)**: graph view of n-gram choices during the explanation walkthrough.
