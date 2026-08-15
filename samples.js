/**
 * Catalog of demo training files in the samples/ folder.
 *
 * The books themselves are plain .txt files, loaded only when the user
 * picks one. They are public-domain literary works obtained from
 * Project Gutenberg (https://www.gutenberg.org/) as a convenient source.
 * Project Gutenberg start/end markers and license headers are omitted, so
 * these files are NOT redistributed as Project Gutenberg eBooks and do not
 * use the Project Gutenberg trademark.
 *
 * Romeo and Juliet — William Shakespeare (complete play)
 *   Public domain
 *   Obtained from: https://www.gutenberg.org/ebooks/1513
 *   Plain-text file: https://www.gutenberg.org/files/1513/1513-0.txt
 *
 * Dracula — Bram Stoker, Chapters I–V
 *   Public domain in the United States (first published 1897)
 *   Obtained from: https://www.gutenberg.org/ebooks/345
 *   Plain-text file: https://www.gutenberg.org/files/345/345-0.txt
 *
 * Last retrieved: 2026-08-15
 */
var SAMPLE_CORPORA = {
  romeo: {
    id: "romeo",
    title: "Romeo and Juliet",
    author: "William Shakespeare",
    license: "Public domain",
    source: "Project Gutenberg",
    ebookId: "1513",
    sourceUrl: "https://www.gutenberg.org/ebooks/1513",
    sourceFile: "https://www.gutenberg.org/files/1513/1513-0.txt",
    scope: "Complete play",
    file: "samples/romeo-and-juliet.txt",
  },
  dracula: {
    id: "dracula",
    title: "Dracula",
    author: "Bram Stoker",
    license: "Public domain in the United States",
    source: "Project Gutenberg",
    ebookId: "345",
    sourceUrl: "https://www.gutenberg.org/ebooks/345",
    sourceFile: "https://www.gutenberg.org/files/345/345-0.txt",
    scope: "Chapters I–V",
    file: "samples/dracula-chapters-i-v.txt",
  },
};
