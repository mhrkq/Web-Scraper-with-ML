import re
import time
import torch
import requests
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import trafilatura
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "facebook/bart-large-cnn"
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if device >= 0 else "cpu")
model.eval()

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

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

    return chunks

def safe_summarize_tokens(text, max_length=120, min_length=40, chunk_size_tokens=800):
    
    chunks = chunk_text_by_sentence(text, tokenizer, chunk_size_tokens)
    summaries = []
    
    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i} / {len(chunks)}")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True).to(model.device)

        with torch.inference_mode():
            summary_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    if len(summaries) > 1:
        combined = " ".join(summaries)
        combined_inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.inference_mode():
            final_ids = model.generate(
                **combined_inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
        return tokenizer.decode(final_ids[0], skip_special_tokens=True)

    return summaries[0]

def fetch_clean_text(url, timeout=15):
    """Download and clean main article text from a URL."""

    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded, include_comments=False)
        if text and len(text.split()) > 100:
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article") or soup.find("main")
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    else:
        paragraphs = soup.find_all("p")
        paragraphs = sorted(paragraphs, key=lambda p: len(p.get_text()), reverse=True)
        paragraphs = [p.get_text(" ", strip=True) for p in paragraphs[:10]]

    text = " ".join(paragraphs)
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else None

def scrape_and_summarize(url):
    start_time = time.time()
    try:
        text_to_summarize = fetch_clean_text(url)
    except Exception as e:
        return {"error": str(e)}

    if not text_to_summarize:
        return {"error": "No suitable text extracted from page."}

    title, description = None, None
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc.get("content") if meta_desc else None
    except Exception:
        pass
    
    summary = safe_summarize_tokens(text_to_summarize)
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTotal time taken: {minutes} min {seconds} sec")

    return {
        "title": title,
        "description": description,
        "summary": summary
    }

if __name__ == "__main__":
    result = scrape_and_summarize(
        # "https://en.wikipedia.org/wiki/Web_scraping"
        # "https://en.wikipedia.org/wiki/The_Three_Little_Pigs"
        # "https://energyeducation.ca/encyclopedia/Thermohaline_circulation"
        "https://warhammerfantasy.fandom.com/wiki/Skaven"
        # "https://www.amazon.com/Amazon-Basics-Color-Coded-Dishwasher-Multicolor/dp/B01B3GARVG/ref=s9_acsd_al_ot_c2_x_4_t?_encoding=UTF8&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-20&pf_rd_r=9XG4Q4Z294ZPGT2F0AG3&pf_rd_p=e4cf989f-2cd6-409b-8da8-0428b845e12a&pf_rd_t=&pf_rd_i=20853249011"
        # "https://www.lazada.sg/products/pdp-i301078910-s527100805.html?scm=1007.17760.398138.0&pvid=81c9b8f6-b7a0-428f-ac9d-04fca13e4cc9&search=flashsale&spm=a2o42.homepage.FlashSale.d_301078910"
        )
    print(result)