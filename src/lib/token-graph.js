import cytoscape from "cytoscape";
import { formatToken } from "./tokens.js";

/** @typedef {import("./types.js").ContinuationOption} ContinuationOption */

/**
 * Cytoscape instance for the walkthrough graph.
 *
 * @param {HTMLElement} container
 * @returns {import("cytoscape").Core}
 */
export function createTokenGraph(container) {
  const cy = cytoscape({
    container,
    autoungrabify: true,
    autounselectify: true,
    boxSelectionEnabled: false,
    userZoomingEnabled: false,
    userPanningEnabled: true,
    wheelSensitivity: 0.2,
    style: graphStyle(isDark()),
  });
  // Stylesheet `cursor` loses to the canvas pan cursor; set it on hover instead.
  cy.on("mouseover", 'node[kind = "alt"], node[kind = "chosen"]', () => {
    setGraphCursor(container, "pointer");
  });
  cy.on("mouseout", 'node[kind = "alt"], node[kind = "chosen"]', () => {
    setGraphCursor(container, "");
  });
  return cy;
}

/**
 * @param {HTMLElement} container
 * @param {string} cursor
 */
function setGraphCursor(container, cursor) {
  container.style.cursor = cursor;
  for (const canvas of container.querySelectorAll("canvas")) {
    canvas.style.cursor = cursor;
  }
}

/**
 * Draws the current token and the pieces that could have come next.
 *
 * @param {import("cytoscape").Core | null} cy
 * @param {{ fromToken: string, options: ContinuationOption[] }} graph
 */
export function drawTokenGraph(cy, { fromToken, options }) {
  if (!cy) {
    return;
  }

  const dark = isDark();
  /** @type {import("cytoscape").ElementDefinition[]} */
  const elements = [
    {
      data: {
        id: "from",
        label: formatToken(fromToken),
        kind: "from",
      },
    },
  ];

  for (const option of options) {
    const id = `opt-${option.tokenId ?? option.token}`;
    elements.push({
      data: {
        id,
        label: `${formatToken(option.token)}\n${percent(option.probability)}`,
        kind: option.chosen ? "chosen" : "alt",
        weight: Math.max(0.15, option.probability),
      },
    });
    elements.push({
      data: {
        id: `edge-${id}`,
        source: "from",
        target: id,
        kind: option.chosen ? "chosen" : "alt",
        weight: Math.max(0.15, option.probability),
      },
    });
  }

  cy.json({ style: graphStyle(dark), elements });
  cy.layout({
    name: "breadthfirst",
    directed: true,
    roots: ["#from"],
    padding: 20,
    spacingFactor: options.length > 6 ? 1.15 : 1.35,
    animate: false,
  }).run();
  cy.resize();
  cy.fit(undefined, 28);
}

/** @param {number} value */
function percent(value) {
  return `${((Number(value) || 0) * 100).toFixed(1)}%`;
}

function isDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/** @param {boolean} dark @returns {import("cytoscape").StylesheetJson} */
function graphStyle(dark) {
  const fromBg = dark ? "#ff8904" : "#c2410c";
  const chosenBg = dark ? "#ff8904" : "#c2410c";
  const altBg = dark ? "#27272a" : "#f4f4f5";
  const altText = dark ? "#e4e4e7" : "#3f3f46";
  const altBorder = dark ? "rgba(255,255,255,0.14)" : "rgba(9,9,11,0.12)";
  const edge = dark ? "rgba(255,255,255,0.28)" : "rgba(9,9,11,0.22)";
  const chosenEdge = fromBg;

  return /** @type {import("cytoscape").StylesheetJson} */ ([
    {
      selector: "node",
      style: {
        label: "data(label)",
        "text-wrap": "wrap",
        "text-valign": "center",
        "text-halign": "center",
        "font-family": "VT323, ui-monospace, monospace",
        "font-size": 16,
        "text-max-width": 90,
        width: 72,
        height: 48,
        shape: "round-rectangle",
        "border-width": 1,
      },
    },
    {
      selector: 'node[kind = "from"]',
      style: {
        "background-color": fromBg,
        color: "#fffaf7",
        "border-color": fromBg,
        width: 88,
        height: 52,
      },
    },
    {
      selector: 'node[kind = "chosen"]',
      style: {
        "background-color": chosenBg,
        color: "#fffaf7",
        "border-color": chosenBg,
        cursor: "pointer",
        width: "mapData(weight, 0.15, 1, 64, 92)",
        height: "mapData(weight, 0.15, 1, 44, 56)",
      },
    },
    {
      selector: 'node[kind = "alt"]',
      style: {
        "background-color": altBg,
        color: altText,
        "border-color": altBorder,
        cursor: "pointer",
        width: "mapData(weight, 0.15, 1, 56, 84)",
        height: "mapData(weight, 0.15, 1, 40, 52)",
      },
    },
    {
      selector: "edge",
      style: {
        width: "mapData(weight, 0.15, 1, 1.5, 6)",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": edge,
        "line-color": edge,
        "arrow-scale": 0.8,
      },
    },
    {
      selector: 'edge[kind = "chosen"]',
      style: {
        "line-color": chosenEdge,
        "target-arrow-color": chosenEdge,
      },
    },
  ]);
}
