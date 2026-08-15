# Attributions

This repo is MIT (`LICENSE`). The rest of this file is other people's work: some of it ships with the app, some of it downloads when you first open the page.

## Model

SmolLM2 360M Instruct (`HuggingFaceTB/SmolLM2-360M-Instruct`), Apache 2.0.
The browser fetches it from [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) on first use. The weights are not in this git repo.

## npm packages

[`@huggingface/transformers`](https://github.com/huggingface/transformers.js) (Transformers.js), Apache 2.0.
That's what runs the model in the tab. WebGPU if the browser has it, otherwise WASM/CPU.

[`cytoscape`](https://js.cytoscape.org/), MIT.
The next-token graph on Explore.

[`Vite`](https://vite.dev/) and [`Tailwind CSS`](https://tailwindcss.com/) ([`@tailwindcss/vite`](https://tailwindcss.com/docs/installation/using-vite/)), all MIT.
Vite is the dev server and the production build. Tailwind is the CSS. You only need them while you develop.

Transformers.js also pulls Microsoft's ONNX Runtime Web assembly the first time (MIT). The browser caches it.

## Fonts

[VT323](https://fonts.google.com/specimen/VT323), SIL Open Font License 1.1.
We ship it at `src/fonts/VT323-Regular.woff2`. License text: `src/fonts/OFL.txt`.

[Inter](https://rsms.me/inter/), SIL Open Font License 1.1.
Loaded from `https://rsms.me/inter/inter.css`.

[Instrument Serif](https://fonts.google.com/specimen/Instrument+Serif), SIL Open Font License 1.1.
Loaded from Google Fonts.

## Marks

The little header/footer marks come from [ui.sh](https://ui.sh/) (`https://assets.ui.sh/marks/1.svg`).

## Demo texts

The books in `public/samples/` are public domain. Gutenberg's own headers and licenses are stripped, so these are not Gutenberg eBooks and we don't use that name as a trademark.

*Romeo and Juliet*, William Shakespeare (the whole play). Public domain.
[Project Gutenberg eBook #1513](https://www.gutenberg.org/ebooks/1513) ([plain text](https://www.gutenberg.org/files/1513/1513-0.txt)).

*Dracula*, Bram Stoker, chapters I–V. Public domain in the United States (1897).
[Project Gutenberg eBook #345](https://www.gutenberg.org/ebooks/345) ([plain text](https://www.gutenberg.org/files/345/345-0.txt)).
