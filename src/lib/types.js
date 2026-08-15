/** Shared JSDoc types: TokenStep, Voice, and the worker message unions. */

/**
 * One alternative next piece, as shown in the walkthrough graph.
 *
 * @typedef {object} TokenAlternative
 * @property {string} token
 * @property {number} tokenId
 * @property {number} probability
 */

/**
 * One generated piece plus the alternatives that were possible there.
 *
 * @typedef {object} TokenStep
 * @property {string} token
 * @property {number} tokenId
 * @property {number} probability
 * @property {TokenAlternative[]} alternatives
 */

/**
 * Style excerpts taken from the user's file (or a sample book).
 *
 * @typedef {object} Voice
 * @property {string} title
 * @property {number} wordCount
 * @property {number} excerptCount
 * @property {number} excerptChars
 * @property {string[]} excerpts
 */

/**
 * @typedef {object} ChatMessage
 * @property {string} role
 * @property {string} content
 */

/**
 * @typedef {object} GenerationResult
 * @property {string} text
 * @property {TokenStep[]} steps
 */

/**
 * A next-piece row in the walkthrough, including whether it was chosen.
 *
 * @typedef {object} ContinuationOption
 * @property {string} token
 * @property {number} tokenId
 * @property {number} probability
 * @property {boolean} chosen
 */

/**
 * @typedef {object} BranchRequest
 * @property {number} fromIndex
 * @property {ContinuationOption} option
 */

/**
 * @typedef {"idle" | "loading" | "ready" | "error"} EngineStatus
 */

/**
 * @template T
 * @typedef {object} Deferred
 * @property {Promise<T>} promise
 * @property {(value: T) => void} resolve
 * @property {(reason?: unknown) => void} reject
 */

/**
 * @typedef {object} WorkerLoadIn
 * @property {"load"} type
 */

/**
 * @typedef {object} WorkerGenerateIn
 * @property {"generate"} type
 * @property {number} id
 * @property {ChatMessage[]} messages
 * @property {string} prefix
 * @property {number} temperature
 * @property {number} maxNewTokens
 * @property {boolean} continueFrom
 */

/**
 * @typedef {object} WorkerAbortIn
 * @property {"abort"} type
 */

/**
 * Messages the page posts to the inference worker.
 *
 * @typedef {WorkerLoadIn | WorkerGenerateIn | WorkerAbortIn} WorkerInMessage
 */

/**
 * @typedef {object} WorkerProgressOut
 * @property {"progress"} type
 * @property {number} [percent]
 * @property {number} [loaded]
 * @property {number} [total]
 * @property {string} [message]
 */

/**
 * @typedef {object} WorkerReadyOut
 * @property {"ready"} type
 * @property {string} device
 * @property {string} dtype
 * @property {string} [message]
 */

/**
 * @typedef {object} WorkerTokenOut
 * @property {"token"} type
 * @property {number} id
 * @property {TokenStep} step
 */

/**
 * @typedef {object} WorkerCompleteOut
 * @property {"complete"} type
 * @property {number} id
 * @property {string} text
 * @property {TokenStep[]} steps
 */

/**
 * @typedef {object} WorkerErrorOut
 * @property {"error"} type
 * @property {string} message
 */

/**
 * Messages the inference worker posts back to the page.
 *
 * @typedef {WorkerProgressOut | WorkerReadyOut | WorkerTokenOut | WorkerCompleteOut | WorkerErrorOut} WorkerOutMessage
 */

/**
 * @typedef {object} SampleCorpus
 * @property {string} id
 * @property {string} title
 * @property {string} author
 * @property {string} subtitle
 * @property {string} license
 * @property {string} fileName
 * @property {string} file
 * @property {string} source
 * @property {string} ebookId
 * @property {string} sourceUrl
 * @property {string} sourceFile
 * @property {string} attributionNote
 */

/**
 * @typedef {object} GenerateRequest
 * @property {ChatMessage[]} messages
 * @property {string} prefix
 * @property {number} temperature
 * @property {number} maxNewTokens
 * @property {(step: TokenStep) => void} [onToken]
 * @property {boolean} [continueFrom]
 */

/**
 * @typedef {object} AppElements
 * @property {HTMLElement} app
 * @property {HTMLElement} engineStatus
 * @property {HTMLInputElement} fileInput
 * @property {HTMLElement} progressBarContainer
 * @property {HTMLElement} progressBar
 * @property {HTMLElement} progressText
 * @property {HTMLTextAreaElement} promptInput
 * @property {HTMLButtonElement} generateBtn
 * @property {HTMLElement} generateHint
 * @property {HTMLInputElement} temperatureInput
 * @property {HTMLElement} temperatureValue
 * @property {HTMLButtonElement} stopBtn
 * @property {HTMLButtonElement} regenerateBtn
 * @property {HTMLButtonElement} newPromptBtn
 * @property {HTMLElement} outputCard
 * @property {HTMLElement} explainCaption
 * @property {HTMLElement} explainOptions
 * @property {HTMLElement} explainProgress
 * @property {HTMLElement} generatedText
 * @property {HTMLElement} progressPercent
 * @property {HTMLElement} voicePanel
 * @property {HTMLElement} writePanel
 * @property {HTMLElement} explorePanel
 * @property {HTMLElement} introPanel
 * @property {HTMLButtonElement} startBtn
 * @property {HTMLElement} fileUploadArea
 * @property {HTMLElement} studioModelSummary
 * @property {HTMLElement} toastRegion
 * @property {HTMLDialogElement} mobileMenu
 * @property {HTMLButtonElement} mobileMenuBtn
 * @property {HTMLButtonElement} mobileMenuClose
 * @property {HTMLElement} sampleList
 * @property {HTMLElement} sampleAttribution
 */

export {};
