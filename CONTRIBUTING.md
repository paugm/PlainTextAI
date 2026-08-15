# Contributing

This is a hobby page. You run SmolLM2 in the browser and watch it pick the next token.

## Setup

Node 20 or newer.

```bash
npm install
npm test
npm run typecheck
npm run dev
```

Vite prints a URL, usually `http://localhost:5173`. Chrome or Edge with a GPU is easier. Firefox and Safari often work, just slower without WebGPU.

```bash
npm run build
npm run preview
```

Run `npm test` and `npm run typecheck` before you send a PR. `npm run smoke` is optional and heavy: it loads the model on CPU and continues an opening. First time it may pull about 300 MB.

Don't commit model weights, `node_modules`, or `dist/`.

## Adding a sample text

Public domain only, or something you can clearly reuse. Cut Project Gutenberg headers and licenses. We don't ship those files as Gutenberg eBooks.

Put the `.txt` in `public/samples/`. Add a row in `src/samples.js` (title, author, source URL, ebook id). Note the source in `ATTRIBUTIONS.md` and in the README sample list.

## Code style

There's no Prettier yet, so copy the files next to yours. Two spaces, no tabs, same width in HTML, CSS, and JavaScript.

JavaScript names follow the files around them: camelCase functions and properties, PascalCase classes, SCREAMING_SNAKE constants, kebab-case file names.

`jsconfig.json` turns on `checkJs`. Shared shapes live in `src/lib/types.js` (`TokenStep`, `Voice`, the worker message unions). Worker `type` strings live in `src/lib/worker-protocol.js`. Use those helpers instead of writing `"load"` or `"token"` by hand.

`src/app.js` is the page. DOM lookup and toasts live in `src/lib/dom.js`. Sample-book buttons are in `src/lib/samples-ui.js`.

HTML ids are kebab-case, and they have to contain a hyphen. `generate-btn` is fine; `generateBtn` and `app` are not.

Private members are `#fields` and `#methods`. Transformers.js still wants `_call`, so that name stays.

Loops use `i++` / `i--`, not `i += 1`.

Quotes: double quotes and semicolons, like the rest of the modules.

### Comments

Exported functions, classes, and the constants module get one JSDoc block. First sentence is the summary. Add a second sentence only if the domain is weird (sampling, excerpts, the worker).

Implementation comments are `//`, on the line above. They explain why, not what the next line does. A browser quirk. A heuristic you would otherwise "fix."

Skip JSDoc on private helpers, tests, and names that are already clear (`escapeHtml`, `formatToken`). Parameter types on those are still required for `checkJs`.

Write those summaries in third person, present tense. "Picks passages from different parts of the file."

In tests, the `it("…")` string is enough.

Don't add `@param` or `@returns` that just repeat the argument name. Do add types when they name a typedef (`TokenStep`, `Voice`, worker messages).

## Pull requests

Run `npm test` and `npm run typecheck`. Don't reformat files you didn't touch. If you add a runtime dependency, put its license in `ATTRIBUTIONS.md`. Keep user-facing copy as short as the rest of the page.

If you find a security problem, don't open a public issue. Use a GitHub Security Advisory, or email me.
