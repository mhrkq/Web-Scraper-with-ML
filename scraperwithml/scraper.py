import re
import time
import torch
import requests
import logging
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import trafilatura
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
    log.info("NLTK punkt tokenizer found.")
except LookupError:
    log.info("NLTK punkt tokenizer not found. Downloading...")
    nltk.download("punkt")
    log.info("NLTK punkt tokenizer downloaded successfully.")

# MODEL 
model_name = "facebook/bart-large-cnn"
device = 0 if torch.cuda.is_available() else -1
log.info(f"Loading model {model_name} on {'GPU' if device >= 0 else 'CPU'} ...")

dtype = torch.float16 if device >= 0 else torch.float32
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(
    "cuda" if device >= 0 else "cpu"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
log.info("Model and tokenizer loaded successfully.")

# CHUNKING
def chunk_text_by_sentence(text, tokenizer, chunk_size_tokens=800):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # try to add sentence to curr chunk
        candidate = (current_chunk + " " + sentence).strip()
        token_count = len(tokenizer.encode(candidate))

        if token_count > chunk_size_tokens:
            # save curr chunk and start new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    log.info(f"Text split into {len(chunks)} chunks.")
    return chunks

SWITCH_THRESHOLD = 5

# SUMMARISATION
def safe_summarize_tokens(text, max_length=120, min_length=40, chunk_size_tokens=800):
    
    chunks = chunk_text_by_sentence(text, tokenizer, chunk_size_tokens)
    n_chunks = len(chunks)
    log.info(f"Detected {n_chunks} chunks.")
    
    if n_chunks <= SWITCH_THRESHOLD:
        log.info("Using chunk-by-chunk summarization.") # longer time taken, but shorter summary
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            log.info(f"Summarizing chunk {i}/{n_chunks}...")
            with torch.inference_mode():
                ids = model.generate(
                    **tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(model.device),
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
            summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        return " ".join(summaries)

    log.info("Using batch summarization.") # shorter time taken, but longer summary
    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.inference_mode():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

    summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]
    log.info("All chunks summarized. Combining summaries...")

    combined = " ".join(summaries)
    combined_inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024
    ).to(model.device)

    with torch.inference_mode():
        final_ids = model.generate(
            **combined_inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

    final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    log.info("Final combined summary generated.")
    return final_summary

def fetch_html(url, timeout=15):
    """Fetch HTML content once."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        log.info("HTML fetched successfully.")
        return resp.text
    except Exception as e:
        log.error(f"Failed to fetch URL: {e}")
        raise RuntimeError(f"Failed to fetch URL: {e}")

def extract_main_text(html):
    """Extract article text using trafilatura or fallback to BeautifulSoup."""
    text = trafilatura.extract(html, include_comments=False)
    if text and len(text.split()) > 100:
        log.info("Text successfully extracted with trafilatura.")
        return re.sub(r"\s+", " ", text).strip()

    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find("main")

    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    else:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")][:10]

    text = " ".join(paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    log.info("Text successfully extracted with fallback method.")
    return text if text else None

def extract_metadata(html):
    """Extract title and meta description from the same HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc.get("content") if meta_desc else None
    return title, description

# MAIN FUNCTION
def scrape_and_summarize(url):
    start_time = time.time()
    
    try:
        html = fetch_html(url)
    except Exception as e:
        return {"error": str(e)}

    text_to_summarize = extract_main_text(html)
    if not text_to_summarize:
        return {"error": "No suitable text extracted from page."}

    title, description = extract_metadata(html)
    summary = safe_summarize_tokens(text_to_summarize)
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log.info(f"Total time taken: {minutes} min {seconds} sec")

    return {
        "title": title,
        "description": description,
        "summary": summary
    }

if __name__ == "__main__":
    # url = "https://en.wikipedia.org/wiki/Web_scraping"
    # url = "https://energyeducation.ca/encyclopedia/Thermohaline_circulation"
    # url = "https://www.amazon.com/Amazon-Basics-Color-Coded-Dishwasher-Multicolor/dp/B01B3GARVG/ref=s9_acsd_al_ot_c2_x_4_t?_encoding=UTF8&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-20&pf_rd_r=9XG4Q4Z294ZPGT2F0AG3&pf_rd_p=e4cf989f-2cd6-409b-8da8-0428b845e12a&pf_rd_t=&pf_rd_i=20853249011"
    # url = "https://www.lazada.sg/products/pdp-i301078910-s527100805.html?scm=1007.17760.398138.0&pvid=81c9b8f6-b7a0-428f-ac9d-04fca13e4cc9&search=flashsale&spm=a2o42.homepage.FlashSale.d_301078910"
    url = "https://warhammerfantasy.fandom.com/wiki/Skaven"
    result = scrape_and_summarize(url)
    print(result)