MODEL_NAME = "facebook/bart-large-cnn"
CHUNK_SIZE_TOKENS = 800
SWITCH_THRESHOLD = 5        # chunk-by-chunk or batch summarization; in summarize_text in summarizer.py
SUMMARY_MAX_LENGTH = 120
SUMMARY_MIN_LENGTH = 40
REQUEST_TIMEOUT = 15