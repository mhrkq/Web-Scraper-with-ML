# Web-Scraper-with-ML
## Set Up Instructions

Clone this repository.

Create virtual environment.
```
python -m venv aiscrap-env
```

Activate virtual environment.
```
aiscrap-env\Scripts\activate
```

Install required libraries.
```
pip install requests beautifulsoup4 transformers torch nltk trafilatura
```

Install sentence tokenizer data:
```
python -m nltk.downloader punkt
```
and
```
python -m nltk.downloader punkt_tab
```

Run the script.
```
python scraper.py
```



