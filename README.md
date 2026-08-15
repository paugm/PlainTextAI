<div align="center">
  <img src="https://raw.githubusercontent.com/paugm/PlainTextAI/main/logo.png" alt="Plain Text AI" width="400" height="auto">
  <br><br>
</div>

Plain Text AI is a hobby page. You give it a text file, start a sentence, and a small model in the browser keeps writing in that style.

SmolLM2 360M downloads into the tab. It already speaks English, so you aren't training new weights. Your file is the style guide: about 20,000 characters from different parts of it. After you pick *Romeo and Juliet*, try opening with "The night had only just begun," and watch it keep going.

When it finishes, the walkthrough is the writing itself. Click a piece to see a graph of the next pieces that were possible from there.

Nothing is sent off your computer. You don't need an account or an API key.

## How it works

The first visit downloads a small language model into the browser (about 300&nbsp;MB; then cached). A progress bar shows how far along it is.

Upload a `.txt` file, or pick a public-domain sample. The app keeps about 20,000 characters from different parts of the file. The 8,192-token context window cannot hold a whole play.

Start a sentence and set temperature (how safe or wild the next token pick is). The model continues the line instead of answering a question.

Click a piece in the continuation to see a graph of the next pieces that were possible from there.

The model runs locally with [Transformers.js](https://huggingface.co/docs/transformers.js). WebGPU if the browser has it, otherwise WASM/CPU. That first download is the quantized model plus the ONNX runtime. After that the cache has it.

## Run it

You need [Node.js](https://nodejs.org/) 20 or newer.

```bash
npm install
npm test
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Chrome or Edge with a GPU is easier. Firefox and Safari often work; without WebGPU, writing is slower.

```bash
npm run build
npm run preview
```

That writes a static site to `dist/`. Host that folder on any static server.

`npm run smoke` is optional: it loads SmolLM2 on the CPU and continues an opening in the Romeo style. The first run may download the model.

If you want to send a patch, `CONTRIBUTING.md` has setup and the style rules.

## Demo texts

Public-domain samples live in `public/samples/`:

- *Romeo and Juliet* by William Shakespeare (complete play). Public domain. From [Project Gutenberg eBook #1513](https://www.gutenberg.org/ebooks/1513) ([plain text](https://www.gutenberg.org/files/1513/1513-0.txt)).
- *Dracula* by Bram Stoker, Chapters I–V. Public domain in the United States (first published 1897). From [Project Gutenberg eBook #345](https://www.gutenberg.org/ebooks/345) ([plain text](https://www.gutenberg.org/files/345/345-0.txt)).

These copies leave out Project Gutenberg headers and licenses, so they are not redistributed as Project Gutenberg eBooks and do not use that trademark. You can still upload any `.txt` of your own.

## What this is not

You start a sentence; the model keeps writing. It isn't ChatGPT and it isn't for work docs. The model is small, so the sentences wander, which is what you want if you're here to watch next-token sampling with a real transformer.

## License

MIT. See `LICENSE`. Credit for the rest is in `ATTRIBUTIONS.md`.

## Contributors

Originally created by [Pau Garcia-Mila](https://github.com/paugm).
