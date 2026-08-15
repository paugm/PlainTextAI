/**
 * Public-domain demo texts in /samples.
 *
 * Project Gutenberg start/end markers and license headers are omitted, so
 * these files are not redistributed as Project Gutenberg eBooks.
 */

/** @typedef {import("./lib/types.js").SampleCorpus} SampleCorpus */

/** @type {Record<string, SampleCorpus>} */
export const SAMPLE_CORPORA = {
  romeo: {
    id: "romeo",
    title: "Romeo and Juliet",
    author: "William Shakespeare",
    subtitle: "complete play",
    license: "Public domain",
    fileName: "samples/romeo-and-juliet.txt",
    file: `${import.meta.env.BASE_URL}samples/romeo-and-juliet.txt`,
    source: "Project Gutenberg",
    ebookId: "1513",
    sourceUrl: "https://www.gutenberg.org/ebooks/1513",
    sourceFile: "https://www.gutenberg.org/files/1513/1513-0.txt",
    attributionNote: "complete play",
  },
  dracula: {
    id: "dracula",
    title: "Dracula",
    author: "Bram Stoker",
    subtitle: "chapters I–V",
    license: "Public domain in the United States (first published 1897)",
    fileName: "samples/dracula-chapters-i-v.txt",
    file: `${import.meta.env.BASE_URL}samples/dracula-chapters-i-v.txt`,
    source: "Project Gutenberg",
    ebookId: "345",
    sourceUrl: "https://www.gutenberg.org/ebooks/345",
    sourceFile: "https://www.gutenberg.org/files/345/345-0.txt",
    attributionNote: "Chapters I–V",
  },
};
