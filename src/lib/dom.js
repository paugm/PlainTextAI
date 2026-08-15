import { TOAST_MS } from "./config.js";

/** @typedef {import("./types.js").AppElements} AppElements */

/**
 * Looks up a required page element by kebab-case id.
 *
 * @template {HTMLElement} [T=HTMLElement]
 * @param {string} id
 * @param {new () => T} [ctor]
 * @returns {T}
 */
export function requireElement(id, ctor) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing #${id}`);
  }
  if (ctor && !(el instanceof ctor)) {
    throw new Error(`#${id} is not a ${ctor.name}`);
  }
  return /** @type {T} */ (el);
}

/**
 * Collects the page's interactive elements.
 *
 * @returns {AppElements}
 */
export function bindAppElements() {
  return {
    app: requireElement("app-root"),
    engineStatus: requireElement("engine-status"),
    fileInput: requireElement("file-input", HTMLInputElement),
    progressBarContainer: requireElement("progress-bar-container"),
    progressBar: requireElement("progress-bar"),
    progressText: requireElement("progress-text"),
    promptInput: requireElement("prompt-input", HTMLTextAreaElement),
    generateBtn: requireElement("generate-btn", HTMLButtonElement),
    generateHint: requireElement("generate-hint"),
    temperatureInput: requireElement("temperature-input", HTMLInputElement),
    temperatureValue: requireElement("temperature-value"),
    stopBtn: requireElement("stop-btn", HTMLButtonElement),
    regenerateBtn: requireElement("regenerate-btn", HTMLButtonElement),
    newPromptBtn: requireElement("new-prompt-btn", HTMLButtonElement),
    outputCard: requireElement("output-card"),
    explainCaption: requireElement("explain-caption"),
    explainOptions: requireElement("explain-options"),
    explainProgress: requireElement("explain-progress"),
    generatedText: requireElement("generated-text"),
    progressPercent: requireElement("progress-percent"),
    voicePanel: requireElement("step-1"),
    writePanel: requireElement("step-2"),
    explorePanel: requireElement("step-3"),
    introPanel: requireElement("intro-panel"),
    startBtn: requireElement("start-btn", HTMLButtonElement),
    fileUploadArea: requireElement("file-upload-area"),
    studioModelSummary: requireElement("studio-model-summary"),
    toastRegion: requireElement("toast-region"),
    mobileMenu: requireElement("mobile-menu", HTMLDialogElement),
    mobileMenuBtn: requireElement("mobile-menu-btn", HTMLButtonElement),
    mobileMenuClose: requireElement("mobile-menu-close", HTMLButtonElement),
    sampleList: requireElement("sample-list"),
    sampleAttribution: requireElement("sample-attribution"),
  };
}

/** @param {HTMLElement} el */
export function reveal(el) {
  el.classList.remove("is-hidden");
  el.classList.add("fade-in");
}

/** @param {HTMLElement} el */
export function conceal(el) {
  el.classList.add("is-hidden");
  el.classList.remove("fade-in");
}

/**
 * @param {HTMLElement} host
 * @param {string} message
 */
export function showToast(host, message) {
  document.getElementById("error-toast")?.remove();
  const notification = document.createElement("div");
  notification.id = "error-toast";
  notification.className = "notification notification-error fade-in";
  notification.setAttribute("role", "alert");
  notification.textContent = message;
  host.appendChild(notification);
  setTimeout(() => {
    notification.classList.remove("fade-in");
    notification.classList.add("fade-out");
    setTimeout(() => notification.remove(), 400);
  }, TOAST_MS);
}

/**
 * @param {HTMLElement} area
 * @param {(file: File) => void} onFile
 */
export function wireFileDrop(area, onFile) {
  for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
    area.addEventListener(name, preventDefaults);
  }
  for (const name of ["dragenter", "dragover"]) {
    area.addEventListener(name, () => area.classList.add("highlight"));
  }
  for (const name of ["dragleave", "drop"]) {
    area.addEventListener(name, () => area.classList.remove("highlight"));
  }
  area.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files[0];
    if (file) {
      onFile(file);
    }
  });
}

/** @param {Event} event */
function preventDefaults(event) {
  event.preventDefault();
  event.stopPropagation();
}
