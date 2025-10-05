import time
import logging
from extractor import fetch_html, extract_main_text, extract_metadata
from summarizer import summarize_text
from config import REQUEST_TIMEOUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def scrape_and_summarize(url):
    start_time = time.time()
    
    try:
        html = fetch_html(url)
    except Exception as e:
        return {"error": str(e)}

    text = extract_main_text(html=html)
    if not text:
        return {"error": "No suitable text extracted from page."}

    title, description = extract_metadata(html=html)
    summary = summarize_text(text)

    elapsed = time.time() - start_time
    log.info(f"Total pipeline time: {int(elapsed // 60)} min {int(elapsed % 60)} sec")

    return {
        "title": title,
        "description": description,
        "summary": summary
    }

if __name__ == "__main__":
    # url = "https://en.wikipedia.org/wiki/Web_scraping"
    # url = "https://energyeducation.ca/encyclopedia/Thermohaline_circulation"
    # url = "https://www.amazon.com/Amazon-Basics-Color-Coded-Dishwasher-Multicolor/dp/B01B3GARVG/ref=s9_acsd_al_ot_c2_x_4_t"
    # url = "https://www.lazada.sg/products/pdp-i301078910-s527100805.html"
    # url = "https://warhammerfantasy.fandom.com/wiki/Skaven"
    # url = "https://wiki.warframe.com/w/Orokin"
    url = "https://warframe.fandom.com/wiki/Orokin"
    result = scrape_and_summarize(url)
    print(result)