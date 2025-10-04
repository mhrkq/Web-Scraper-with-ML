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


def summarize_text(text):
    start_time = time.time()
    chunks = chunk_text_by_sentence(text, tokenizer, CHUNK_SIZE_TOKENS)
    n_chunks = len(chunks)
    log.info(f"Detected {n_chunks} chunks.")
    combined_summaries = []
    batch_size = 2 if n_chunks <= SWITCH_THRESHOLD else 4

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=1024, padding=True).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs,
                                 max_length=SUMMARY_MAX_LENGTH,
                                 min_length=SUMMARY_MIN_LENGTH,
                                 do_sample=False)
        summaries = tokenizer.batch_decode(ids, skip_special_tokens=True)
        combined_summaries.extend(summaries)

    combined_text = " ".join(combined_summaries)
    final_chunks = chunk_text_by_sentence(combined_text, tokenizer, CHUNK_SIZE_TOKENS)
    final_summary_list = []
    for chunk in final_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs,
                                 max_length=SUMMARY_MAX_LENGTH,
                                 min_length=SUMMARY_MIN_LENGTH,
                                 do_sample=False)
        final_summary_list.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    final_summary = " ".join(final_summary_list)
    elapsed = time.time() - start_time
    log.info(f"Summarization completed in {int(elapsed // 60)} min {int(elapsed % 60)} sec")
    return final_summary