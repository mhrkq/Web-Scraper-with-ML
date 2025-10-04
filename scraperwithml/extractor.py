import re
import logging
from bs4 import BeautifulSoup
import trafilatura
import requests

log = logging.getLogger(__name__)


def fetch_html(url, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        log.info("HTML fetched successfully.")
        return resp.text
    except Exception as e:
        log.error(f"Failed to fetch URL: {e}")
        raise RuntimeError(f"Failed to fetch URL: {e}")


def extract_main_text(url=None, html=None):
    """
    extract main text from url or pre-fetched html
    if html: use it directly
    else: function will fetch the page using trafilatura
    """
    if html is None:
        if url is None:
            log.error("No URL or HTML provided to extract_main_text.")
            return None
        log.info("Fetching and extracting text with trafilatura...")
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.split()) > 100:
                log.info("Text successfully extracted with trafilatura.")
                return re.sub(r"\s+", " ", text).strip()
    else:
        text = trafilatura.extract(html, include_comments=False)
        if text and len(text.split()) > 100:
            log.info("Text successfully extracted with trafilatura from provided HTML.")
            return re.sub(r"\s+", " ", text).strip()

    log.warning("trafilatura failed or text too short, falling back to BeautifulSoup...")

    if html is None:
        try:
            html = fetch_html(url)
        except Exception as e:
            log.error(f"Failed to fetch URL for fallback: {e}")
            return None

    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find("main")
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    else:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")][:10]

    text = " ".join(paragraphs)
    text = re.sub(r"\s+", " ", text).strip()
    log.info("Text extracted with fallback parser.")
    return text if text else None


def extract_metadata(url=None, html=None):
    if html is None:
        if url is None:
            log.error("No URL or HTML provided to extract_metadata.")
            return None, None
        try:
            html = fetch_html(url)
        except Exception as e:
            log.error(f"Failed to fetch URL for metadata: {e}")
            return None, None

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc.get("content") if meta_desc else None
    return title, description