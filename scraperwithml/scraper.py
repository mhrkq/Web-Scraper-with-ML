import torch
import requests
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
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

        with torch.no_grad():
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
        
        with torch.no_grad():
            final_ids = model.generate(
                **combined_inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
        return tokenizer.decode(final_ids[0], skip_special_tokens=True)

    return summaries[0]

def scrape_and_summarize(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else None

    description = None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"]

    article = soup.find("article") or soup.find("main")
    if article:
        content = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    else:
        paragraphs = soup.find_all("p")
        sorted_paragraphs = sorted(paragraphs, key=lambda p: len(p.get_text()), reverse=True)
        content = [p.get_text(" ", strip=True) for p in sorted_paragraphs[:5]]

    text_to_summarize = " ".join(content)
    summary = None
    if text_to_summarize:
        summary = safe_summarize_tokens(text_to_summarize)

    return {
        "title": title,
        "description": description,
        # "content": content,
        "summary": summary
    }

print(scrape_and_summarize(
    # "https://en.wikipedia.org/wiki/Web_scraping"
    # "https://en.wikipedia.org/wiki/The_Three_Little_Pigs"
    # "https://energyeducation.ca/encyclopedia/Thermohaline_circulation"
    "https://warhammerfantasy.fandom.com/wiki/Skaven"
    # "https://www.amazon.com/Amazon-Basics-Color-Coded-Dishwasher-Multicolor/dp/B01B3GARVG/ref=s9_acsd_al_ot_c2_x_4_t?_encoding=UTF8&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-20&pf_rd_r=9XG4Q4Z294ZPGT2F0AG3&pf_rd_p=e4cf989f-2cd6-409b-8da8-0428b845e12a&pf_rd_t=&pf_rd_i=20853249011"
    # "https://www.lazada.sg/products/pdp-i301078910-s527100805.html?scm=1007.17760.398138.0&pvid=81c9b8f6-b7a0-428f-ac9d-04fca13e4cc9&search=flashsale&spm=a2o42.homepage.FlashSale.d_301078910"
    ))