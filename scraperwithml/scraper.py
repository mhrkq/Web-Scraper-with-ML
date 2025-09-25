import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "facebook/bart-large-cnn"
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if device >= 0 else "cpu")
model.eval()

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

def safe_summarize_tokens(text, max_length=120, min_length=40, chunk_size_tokens=800):
    
    input_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    total_tokens = len(input_ids)
    summaries = []
    start = 0
    chunk_num = 1

    while start < total_tokens:
        end = min(start + chunk_size_tokens, total_tokens)
        chunk_ids = input_ids[start:end].unsqueeze(0).to(model.device)
        print(f"Summarizing chunk {chunk_num} ({start}-{end}/{total_tokens} tokens)")
        
        with torch.no_grad():
            summary_ids = model.generate(
                chunk_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)
        start = end
        chunk_num += 1

    if len(summaries) > 1:
        combined = " ".join(summaries)
        combined_ids = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
        with torch.no_grad():
            summary_ids = model.generate(combined_ids, max_length=max_length, min_length=min_length, do_sample=False)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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
    # "https://warhammerfantasy.fandom.com/wiki/Skaven"
    # "https://www.amazon.com/Amazon-Basics-Color-Coded-Dishwasher-Multicolor/dp/B01B3GARVG/ref=s9_acsd_al_ot_c2_x_4_t?_encoding=UTF8&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-20&pf_rd_r=9XG4Q4Z294ZPGT2F0AG3&pf_rd_p=e4cf989f-2cd6-409b-8da8-0428b845e12a&pf_rd_t=&pf_rd_i=20853249011"
    "https://www.lazada.sg/products/pdp-i301078910-s527100805.html?scm=1007.17760.398138.0&pvid=81c9b8f6-b7a0-428f-ac9d-04fca13e4cc9&search=flashsale&spm=a2o42.homepage.FlashSale.d_301078910"
    ))