import time
import logging
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import MODEL_NAME, CHUNK_SIZE_TOKENS, SWITCH_THRESHOLD, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH

log = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
    log.info("NLTK punkt tokenizer found.")
except LookupError:
    log.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download("punkt")
    log.info("NLTK punkt tokenizer downloaded successfully.")

device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if device >= 0 else torch.float32

log.info(f"Loading model {MODEL_NAME} on {'GPU' if device >= 0 else 'CPU'}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(
    "cuda" if device >= 0 else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer_pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
log.info("Model and tokenizer loaded successfully.")

def chunk_text_by_sentence(text, tokenizer, chunk_size_tokens=800):
    sentences = sent_tokenize(text)
    token_counts = [len(tokenizer.encode(s)) for s in sentences]
    chunks = []
    current_chunk, current_len = [], 0

    for s, t_len in zip(sentences, token_counts):
        if current_len + t_len > chunk_size_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [s], t_len
        else:
            current_chunk.append(s)
            current_len += t_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    log.info(f"Text split into {len(chunks)} chunks.")
    return chunks

# determine dynamic batch size based on GPU memory
def get_dynamic_batch_size():
    if device == -1:
        return 2  # CPU safe batch size
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
        if total_mem < 8:
            return 2
        elif total_mem < 16:
            return 4
        else:
            return 8
    except Exception:
        return 2

def summarize_text(text):
    start_time = time.time()
    chunks = chunk_text_by_sentence(text, tokenizer, CHUNK_SIZE_TOKENS)
    n_chunks = len(chunks)
    log.info(f"Detected {n_chunks} chunks.")
    batch_size = get_dynamic_batch_size()
    log.info(f"Using batch size: {batch_size}")

    def run_summarization(input_texts):
        try:
            with torch.inference_mode():
                return summarizer_pipe(
                    input_texts,
                    max_length=SUMMARY_MAX_LENGTH,
                    min_length=SUMMARY_MIN_LENGTH,
                    truncation=True,
                    batch_size=batch_size
                )
        except RuntimeError as e:
            if "CUDA" in str(e):
                log.warning("CUDA error detected. Retrying on CPU...")
                cpu_model = model.to("cpu")
                cpu_pipe = pipeline(
                    "summarization",
                    model=cpu_model,
                    tokenizer=tokenizer,
                    device=-1
                )
                return cpu_pipe(
                    input_texts,
                    max_length=SUMMARY_MAX_LENGTH,
                    min_length=SUMMARY_MIN_LENGTH,
                    truncation=True,
                    batch_size=2
                )
            else:
                raise e

    # main summarization logic
    if n_chunks <= SWITCH_THRESHOLD:
        log.info("Using chunk-by-chunk summarization (slower, more concise).")
        summaries = run_summarization(chunks)
        combined_summary = " ".join(s['summary_text'] for s in summaries)
    else:
        log.info("Using batch summarization (faster, longer).")
        summaries = run_summarization(chunks)
        combined = " ".join(s['summary_text'] for s in summaries)

        # chunk the combined summary again if it is still too long
        token_count = len(tokenizer.encode(combined))
        if token_count > CHUNK_SIZE_TOKENS:
            log.info(f"Combined summary is {token_count} tokens – re-chunking before final summarization.")
            mid_chunks = chunk_text_by_sentence(combined, tokenizer, CHUNK_SIZE_TOKENS)
            mid_summaries = run_summarization(mid_chunks)
            combined = " ".join(s['summary_text'] for s in mid_summaries)

        log.info("Summarizing combined chunks for final output...")
        final_summary = run_summarization([combined])[0]['summary_text']
        combined_summary = final_summary

    elapsed = time.time() - start_time
    log.info(f"Summarization completed in {int(elapsed // 60)} min {int(elapsed % 60)} sec")
    return combined_summary