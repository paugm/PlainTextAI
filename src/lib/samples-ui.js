import { escapeHtml } from "./html.js";
import { SAMPLE_CORPORA } from "../samples.js";

/** @typedef {import("./types.js").AppElements} AppElements */

/**
 * Fills the sample-book buttons and their Gutenberg attributions.
 *
 * @param {AppElements} elements
 */
export function renderSamples(elements) {
  elements.sampleList.replaceChildren(
    ...Object.values(SAMPLE_CORPORA).map((sample) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-card";
      button.dataset.sample = sample.id;
      button.setAttribute(
        "aria-label",
        `Use ${sample.title} as the text. File ${sample.fileName}`
      );
      button.innerHTML = `
          <span class="font-medium">${escapeHtml(sample.title)}</span>
          <span class="mt-1 block text-zinc-600 dark:text-zinc-400">${escapeHtml(sample.author)} · ${escapeHtml(sample.subtitle)}</span>
          <span class="mt-1 block text-zinc-500 dark:text-zinc-500">${escapeHtml(sample.fileName)}</span>`;
      return button;
    })
  );

  elements.sampleAttribution.replaceChildren(
    ...Object.values(SAMPLE_CORPORA).map((sample) => {
      const p = document.createElement("p");
      p.innerHTML = `<cite>${escapeHtml(sample.title)}</cite> by ${escapeHtml(sample.author)} (${escapeHtml(sample.attributionNote)}). ${escapeHtml(sample.license)}.
          Source: <a href="${escapeHtml(sample.sourceUrl)}" class="underline decoration-zinc-950/20 underline-offset-4 dark:decoration-white/20" target="_blank" rel="noopener noreferrer">${escapeHtml(sample.source)} eBook #${escapeHtml(sample.ebookId)}</a>
          (<a href="${escapeHtml(sample.sourceFile)}" class="underline decoration-zinc-950/20 underline-offset-4 dark:decoration-white/20" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceFileName(sample.sourceFile))}</a>).`;
      return p;
    })
  );
}

/** @param {string} url */
function sourceFileName(url) {
  try {
    return new URL(url).pathname.split("/").pop() || url;
  } catch {
    return url;
  }
}
