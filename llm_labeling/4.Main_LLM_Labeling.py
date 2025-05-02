import pandas as pd
import time
import random
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
import pdfplumber
import os

class StudyTypeClassifier:
    def __init__(self, api_key: str):
        """Initialize the classifier with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        self.min_request_interval = 4.0  # 4 seconds between requests for 15 RPM limit

    def classify_abstracts(self, abstracts: list, study_types: list, poverty_contexts: list, mechanisms: list, max_retries: int = 3) -> list:
        """
        Classify multiple abstracts in a single request
        Args:
            abstracts: List of abstract texts
            study_types: List of valid study types
            poverty_contexts: List of valid poverty contexts
            mechanisms: List of valid mechanisms
            max_retries: Maximum number of retry attempts
        Returns:
            List of tuples (study_type, poverty_context, mechanism) for each abstract
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()

                # Create a single prompt for all abstracts
                prompt = self._create_prompt(abstracts, study_types, poverty_contexts, mechanisms)

                response = self.model.generate_content(prompt)
                result = response.text

                # Parse the response
                results = self._parse_response(result, study_types, poverty_contexts, mechanisms)

                if len(results) == len(abstracts):
                    return results

                print(f"Warning: Received {len(results)} results for {len(abstracts)} abstracts. Retrying...")

            except Exception as e:
                print(f"Error processing batch on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue

        return [("Insufficient info", "Insufficient info", "Insufficient info")] * len(abstracts)

    def _create_prompt(self, abstracts: list, study_types: list, poverty_contexts: list, mechanisms: list) -> str:
        """Create a prompt for processing multiple abstracts"""
        prompt = f"""
        Analyze the following {len(abstracts)} abstracts and classify each one according to:
        1. Study Type
        2. Poverty Context
        3. Mechanism

        For each abstract, provide the classification in the format:
        study_type, poverty_context, mechanism

        Guidelines:
        1. Focus ONLY on identifying the study type used in each specific abstract
        2. Do not consider other studies mentioned in the abstracts
        3. If an abstract doesn't clearly describe its own study design, poverty context, or mechanism, respond with "Insufficient info"
        4. Be conservative - only classify if you find clear evidence in the abstract
        5. Look for specific methodological details that indicate the study type

        Study Type Definitions:
        {self._format_definitions(study_types)}

        Poverty Context Definitions:
        {self._format_definitions(poverty_contexts)}

        Mechanism Definitions:
        {self._format_definitions(mechanisms)}

        Abstracts:
        {self._format_abstracts(abstracts)}

        Provide your response as a numbered list, one classification per line, in the format:
        1. study_type, poverty_context, mechanism
        2. study_type, poverty_context, mechanism
        ...
        """
        return prompt

    def _format_definitions(self, items: list) -> str:
        """Format definitions for the prompt"""
        return "\n".join([f"- {item}" for item in items])

    def _format_abstracts(self, abstracts: list) -> str:
        """Format abstracts for the prompt"""
        return "\n\n".join([f"Abstract {i+1}:\n{abstract}" for i, abstract in enumerate(abstracts)])

    def _parse_response(self, response: str, study_types: list, poverty_contexts: list, mechanisms: list) -> list:
        """Parse the response into individual classifications"""
        results = []
        lines = response.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Extract the classification part (after the number and dot)
            parts = line.split('.', 1)
            if len(parts) != 2:
                continue

            classification = parts[1].strip()
            classifications = [c.strip() for c in classification.split(',')]

            if len(classifications) != 3:
                results.append(("Insufficient info", "Insufficient info", "Insufficient info"))
                continue

            study_type, poverty_context, mechanism = classifications

            # Validate each classification
            study_type = self._validate_classification(study_type, study_types)
            poverty_context = self._validate_classification(poverty_context, poverty_contexts)
            mechanism = self._validate_classification(mechanism, mechanisms)

            results.append((study_type, poverty_context, mechanism))

        return results

    def _validate_classification(self, classification: str, valid_options: list) -> str:
        """Validate a single classification with more flexible matching"""
        if classification.lower() == "insufficient info":
            return "Insufficient info"

        # Try exact match first
        if classification in valid_options:
            return classification

        # Try case-insensitive match
        classification_lower = classification.lower()
        for option in valid_options:
            if option.lower() == classification_lower:
                return option

        # Try partial match
        for option in valid_options:
            if option.lower() in classification_lower or classification_lower in option.lower():
                return option

        print(f"Warning: Invalid classification: {classification}")
        return "Insufficient info"

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by waiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

class PaperRetriever:
    def __init__(self):
        self.scihub_urls = [
            "https://sci-hub.se/",
            "https://sci-hub.st/",
            "https://sci-hub.ru/",
            "https://sci-hub.ee/",
            "https://sci-hub.shop/"
        ]
        self.current_url_index = 0
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def _get_next_scihub_url(self):
        url = self.scihub_urls[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.scihub_urls)
        return url

    def get_paper_text(self, doi: str) -> str:
        """Retrieve paper text from Sci-Hub using DOI."""
        if pd.isna(doi) or not isinstance(doi, str) or len(doi.strip()) == 0:
            return None

        for _ in range(len(self.scihub_urls)):
            try:
                self._wait_for_rate_limit()
                base_url = self._get_next_scihub_url()
                url = f"{base_url}{quote(doi)}"

                print(f"Trying to access: {url}")
                response = requests.get(url, headers=self.headers, timeout=15)

                if response.status_code != 200:
                    print(f"Failed to access Sci-Hub. Status code: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                # Try multiple methods to find the PDF
                pdf_url = None

                # Method 1: Look for iframe
                iframe = soup.find('iframe')
                if iframe and iframe.get('src'):
                    pdf_url = iframe['src']

                # Method 2: Look for PDF link
                if not pdf_url:
                    pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$'))
                    if pdf_links:
                        pdf_url = pdf_links[0]['href']

                # Method 3: Look for embedded PDF
                if not pdf_url:
                    embed = soup.find('embed')
                    if embed and embed.get('src'):
                        pdf_url = embed['src']

                if pdf_url:
                    # Fix relative URLs
                    if not pdf_url.startswith('http'):
                        if pdf_url.startswith('//'):
                            pdf_url = f"https:{pdf_url}"
                        else:
                            pdf_url = f"{base_url}{pdf_url}"

                    print(f"Found PDF URL: {pdf_url}")
                    return pdf_url

                print("Could not find PDF URL in the response")

            except requests.exceptions.Timeout:
                print("Request timed out")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request error: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                continue

        return None

def extract_text_from_pdf(pdf_url: str, max_retries: int = 3) -> str:
    """Extract text from a PDF URL with retry logic."""
    for attempt in range(max_retries):
        try:
            # Increase timeout and add headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            print(f"Attempt {attempt + 1} to download PDF from: {pdf_url}")
            response = requests.get(pdf_url, headers=headers, timeout=30)  # Increased timeout to 30 seconds

            if response.status_code != 200:
                print(f"Failed to download PDF. Status code: {response.status_code}")
                continue

            # Save the PDF temporarily
            temp_file = f'temp_pdf_{attempt}.pdf'
            with open(temp_file, 'wb') as f:
                f.write(response.content)

            try:
                # Extract text from the saved PDF
                with pdfplumber.open(temp_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                # Clean up the temporary file
                os.remove(temp_file)

                if text.strip():  # Check if we actually got any text
                    return text
                else:
                    print(f"No text extracted from PDF on attempt {attempt + 1}")
                    continue

            except Exception as e:
                print(f"Error processing PDF file on attempt {attempt + 1}: {str(e)}")
                # Clean up the temporary file even if processing fails
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                continue

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
        except Exception as e:
            print(f"Error extracting text from PDF on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue

    return None

def process_insufficient_cases(df: pd.DataFrame, classifier: StudyTypeClassifier,
                            study_types: list, poverty_contexts: list, mechanisms: list):
    """Process papers with insufficient information by retrieving full text."""
    paper_retriever = PaperRetriever()

    insufficient_mask = (
        (df['study_type_label'] == 'Insufficient info') |
        (df['poverty_context'] == 'Insufficient info') |
        (df['mechanism'] == 'Insufficient info')
    )

    insufficient_df = df[insufficient_mask].copy()

    for idx, row in insufficient_df.iterrows():
        print(f"\nProcessing paper with insufficient info at index {idx}")
        print(f"DOI: {row['DOI']}")

        paper_url = paper_retriever.get_paper_text(row['DOI'])
        if paper_url:
            print(f"Retrieved paper from: {paper_url}")
            full_text = extract_text_from_pdf(paper_url)
            if full_text:
                study_type, poverty_context, mechanism = classifier.classify_abstracts(
                    [full_text], study_types, poverty_contexts, mechanisms
                )[0]
                df.at[idx, 'study_type_label'] = study_type
                df.at[idx, 'poverty_context'] = poverty_context
                df.at[idx, 'mechanism'] = mechanism
                print(f"Updated classifications for paper {idx}")
            else:
                print(f"Could not extract text from PDF for DOI: {row['DOI']}")
                # Save the URL for manual processing
                df.at[idx, 'pdf_url'] = paper_url
        else:
            print(f"Could not retrieve paper for DOI: {row['DOI']}")

    # Save the updated dataframe
    df.to_csv('results_with_full_text.csv', index=False)

# Read the CSV file
df = df2  # Must have a column named "abstract"

# Define study types, poverty contexts, and mechanisms
study_types = [
    "Randomized controlled trial",
    "Quasi-experimental",
    "Cross-sectional",
    "Longitudinal",
    "Review",
    "Regression",
    "Qualitative",
    "Experimental",
    "Experimental study",
    "Experimental design"
]

poverty_contexts = [
    "Low resource level",
    "Resource volatility",
    "Physical environment",
    "Social environment"
]

mechanisms = [
    # Affective
    "Anxiety", "Depression", "Happiness", "Stress",
    # Beliefs
    "Aspirations", "Internalized stigma", "Mindset", "Self-efficacy", "Optimism",
    # Cognitive Function
    "Attention", "Cognitive flexibility", "Executive control", "Memory", "Fluid intelligence",
    # Preferences
    "Time preference", "Risk preference"
]

# Initialize classifier
classifier = StudyTypeClassifier("AIzaSyBTnTeS5XjN4eauA5YqU4BcuA6zX_HuGsE")

# Process all abstracts in batches of 10:
abstracts = df['abstract'].tolist()
results = classifier.classify_abstracts(abstracts, study_types, poverty_contexts, mechanisms)

# Update the DataFrame with the results
for idx, (study_type, poverty_context, mechanism) in enumerate(results):
    df.at[idx, 'study_type_label'] = study_type
    df.at[idx, 'poverty_context'] = poverty_context
    df.at[idx, 'mechanism'] = mechanism

# Save the updated dataframe
df.to_csv('results_with_full_texts.csv', index=False)

# Usage example:
# process_insufficient_cases(df, classifier, study_types, poverty_contexts, mechanisms)