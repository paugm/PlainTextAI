# Changelog

## 2.0.0 - 2026-08-15

The app is a Vite + npm project now. `PlainTextLMBuilder.js` is gone.

- SmolLM2 360M runs in a Web Worker through Transformers.js. WebGPU when the browser has it, otherwise WASM.
- Explore draws next-token options as a Cytoscape graph. Click another option and it keeps writing from there.
- Sample books live in `public/samples/`.
- Indent is 2 spaces in HTML, CSS, and JS. Ids are kebab-case. Private fields use `#`.
- `checkJs` is on. Worker messages are typed in `src/lib/worker-protocol.js`. DOM helpers and the sample-book buttons live in their own modules, so `src/app.js` is smaller.
