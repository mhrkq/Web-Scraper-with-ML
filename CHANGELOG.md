# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- BART model (facebook/bart-large-cnn) for text summarization from Hugging Face's transformers library.
- `scrape_and_summarize` function to scrape content from a URL, then summarize it using the BART model.
- `safe_summarize_tokens` function to break large texts into manageable chunks for summarization based on token size.

### Changed

### Fixed

### Removed

## 24/09/2025

### Added

- `torch.no_grad()` context to avoid unnecessary gradient calculations during the summarization process.
- `model.eval()` to disable dropout and other training-specific behaviors during inference.

### Changed

- `model.generate()` call moved inside the `torch.no_grad()` context to avoid redundancy and ensure proper inference behaviour when generating summaries for each chunk and for combined chunks.

### Fixed

### Removed

## 25/09/2025

### Added

- `nltk` library to tokenize text into sentences for chunking.
- `chunk_text_by_sentence function` splits input text into smaller chunks based on sentence boundaries, ensuring each chunk stays within token limit for summarization. Text is divided into logical parts, improving summarization performance.

### Changed

- summarization process in `safe_summarize_tokens` function to use sentence-based chunks instead of arbitrary token-based splits, improving flow and coherence of summarized text.
- the way `inputs` were passed to model for summarization. Each chunk now has its own set of inputs, optimizing performance.

### Fixed

- proper use of `**inputs` within `generate()` call, preventing incorrect input handling during inference.

### Removed

## 28/09/2025

### Added

- `trafilatura` library to fetch and clean article text from URLs, a more reliable and efficient method for extracting main content from web pages.
- `fetch_clean_text` function to download and extract clean text from a URL using trafilatura, falling back to `requests` if the text extraction fails.
- time tracking in `scrape_and_summarize` function to log time taken for entire process.

### Changed

- `torch.no_grad()` changed to `torch.inference_mode()` for more efficient inference by reducing memory usage and computational overhead.

### Fixed

### Removed

## 29/09/2025

### Added

### Changed

- `logging` for structured, timestamped info, warning, and error messages throughout the script
- model loads with appropriate torch_dtype (float16 for GPU, float32 for CPU) to optimize memory usage and performance.
- batch summarization in `safe_summarize_tokens` encodes and generates summaries for all chunks at once, instead of iterating chunk-by-chunk.

### Fixed

### Removed

## 30/09/2025