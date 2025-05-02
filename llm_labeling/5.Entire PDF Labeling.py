# pip install pdfplumber

import pandas as pd
import requests
import time
import os
import pdfplumber
import google.generativeai as genai
from bs4 import BeautifulSoup
import re
from urllib.parse import quote

# --- Gemini Classifier ---
class FullTextClassifier:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.min_interval = 4.0
        self.last_time = 0

    def _wait(self):
        elapsed = time.time() - self.last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_time = time.time()

    def classify(self, full_text: str, study_types: list, poverty_contexts: list, mechanisms: list):
        self._wait()
        prompt = f"""
Classify the following research paper into:
1. Study Type
2. Poverty Context
3. Psychological Mechanism

Choose only from:
Study Types: {', '.join(study_types)}
Poverty Contexts: {', '.join(poverty_contexts)}
Mechanisms: {', '.join(mechanisms)}

Be conservative. If unclear, answer 'Insufficient info'.

Text:
{full_text[:9000]}  # limit to ~9000 tokens

Return format:
study_type, poverty_context, mechanism
"""
        response = self.model.generate_content(prompt)
        return response.text.strip()

# --- DOI → PDF Retriever ---
class PDFDownloader:
    def __init__(self):
        self.scihub_urls = [
            "https://sci-hub.se/", "https://sci-hub.ru/",
            "https://sci-hub.st/", "https://sci-hub.ee/"
        ]
        self.index = 0
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def get_pdf_url(self, doi: str):
        for _ in range(len(self.scihub_urls)):
            base = self.scihub_urls[self.index]
            self.index = (self.index + 1) % len(self.scihub_urls)
            try:
                url = f"{base}{quote(doi)}"
                res = requests.get(url, headers=self.headers, timeout=15)
                soup = BeautifulSoup(res.text, 'html.parser')
                iframe = soup.find("iframe")
                if iframe and iframe.get("src"):
                    link = iframe["src"]
                    return f"https:{link}" if link.startswith("//") else link
            except Exception:
                continue
        return None

    def download(self, pdf_url: str, filename: str) -> str:
        r = requests.get(pdf_url, headers=self.headers, timeout=30)
        with open(filename, "wb") as f:
            f.write(r.content)
        return filename

# --- PDF Text Extraction ---
def extract_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()

# --- Constants ---
study_types = [
    "Randomized controlled trial", "Quasi-experimental", "Cross-sectional",
    "Longitudinal", "Review", "Regression", "Qualitative", "Experimental"
]
poverty_contexts = [
    "Low resource level", "Resource volatility", "Physical environment", "Social environment"
]
mechanisms = [
    "Anxiety", "Depression", "Happiness", "Stress", "Aspirations", "Internalized stigma",
    "Mindset", "Self-efficacy", "Optimism", "Attention", "Cognitive flexibility",
    "Executive control", "Memory", "Fluid intelligence", "Time preference", "Risk preference"
]

# --- Run Pipeline ---
def classify_papers_from_dois(doi_list: list, api_key: str, output_csv: str):
    retriever = PDFDownloader()
    classifier = FullTextClassifier(api_key)
    results = []

    for doi in doi_list:
        print(f"\nProcessing DOI: {doi}")
        try:
            pdf_url = retriever.get_pdf_url(doi)
            if not pdf_url:
                raise Exception("PDF not found")
            print(f"PDF URL: {pdf_url}")
            filename = f"temp_{doi.replace('/', '_')}.pdf"
            retriever.download(pdf_url, filename)

            full_text = extract_text(filename)
            os.remove(filename)

            result = classifier.classify(full_text, study_types, poverty_contexts, mechanisms)
            results.append((doi, result))
        except Exception as e:
            print(f"Error for {doi}: {e}")
            results.append((doi, "Failed"))

    # Save results
    df = pd.DataFrame(results, columns=["DOI", "Classification"])
    df.to_csv(output_csv, index=False)
    print(f"\n Saved to {output_csv}")

# Example use case
doi_list = [
    "10.1126/science.1238041"
]
classify_papers_from_dois(doi_list, api_key="API Key here", output_csv="gemini_fulltext_results.csv")
