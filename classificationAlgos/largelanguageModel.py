import pandas as pd
import time
import google.generativeai as genai
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiModel:
    def __init__(self, api_key, model_name="gemini-2.0-flash", batch_size=500):
        """
        Initialize the classifier with API key, model name, and batch size.
        
        Args:
            api_key: API key for Gemini model
            model_name: Name of the model to use
            batch_size: Number of abstracts to process in each batch
        """
        self.api_key = api_key
        self.batch_size = batch_size
        self.model_name = model_name
        self.cache = {}
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _generate(self, prompt, delay=6):
        if prompt in self.cache:
            return self.cache[prompt]
        time.sleep(delay)  # Rate limit management
        output = self.model.generate_content(prompt).text
        self.cache[prompt] = output
        return output
        
    def _load_labels(self):
        """Load risk types from the JSON file."""
        try:
            with open('data/trainingData/labels.json', 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return {}
                
    def _create_prompt(self, abstracts_batch, study_types, poverty_contexts, mechanisms) -> str:
        """Create a prompt for processing a batch of abstracts"""
        # Format abstracts for the prompt
        formatted_abstracts = ""
        for doi, abstract in abstracts_batch.items():
            formatted_abstracts += f"DOI: {doi}\nAbstract: {abstract}\n\n"
            
        prompt = f"""
        Analyze the following {len(abstracts_batch)} abstracts and classify each one according to:
        1. Study Type
        2. Poverty Context
        3. Mechanism
        
        For each abstract, provide the classification in the format:
        Study Type, Poverty Context, Mechanism
        
        Guidelines:
        1. Focus ONLY on identifying the study type used in each specific abstract
        2. Do not consider other studies mentioned in the abstracts
        3. If an abstract doesn't clearly describe its own study design, poverty context, or mechanism, respond with "Insufficient info"
        4. Look for specific methodological details that indicate the study type
        
        Study Type Definitions:
        {study_types}
        
        Poverty Context Definitions:
        {poverty_contexts}
        
        Mechanism Definitions:
        {mechanisms}
        
        Abstracts:
        {formatted_abstracts}
        
        Please return your response in the following JSON format:
        {{
            "doi1": {{"study_type": "<Study Type>", "poverty_context": "<Poverty Context>", "mechanism": "<Mechanism>"}},
            "doi2": {{"study_type": "<Study Type>", "poverty_context": "<Poverty Context>", "mechanism": "<Mechanism>"}}
        }}
        """
        return prompt
        
    def classify_batch(self, abstracts_batch, study_types, poverty_contexts, mechanisms):
        """
        Classify a batch of abstracts using the Gemini model.
        
        """
        prompt = self._create_prompt(abstracts_batch, study_types, poverty_contexts, mechanisms)
        
        try:
            response = self._generate(prompt)
            logger.info(f"Successfully classified batch of {len(abstracts_batch)} abstracts")

            return response
                
        except Exception as e:
            logger.error(f"Error generating classification: {e}")
            # Add exponential backoff retry logic here if needed
            return {}
            
    def classify(self, papers, study_types, poverty_contexts, mechanisms):
        """
        Classify abstracts into study types, poverty_contexts, and mechanisms.
        
        Args:
            papers: List of paper dictionaries containing 'doi' and 'abstract'
            study_types: Study type definitions
            poverty_contexts: Poverty context definitions
            mechanisms: Mechanism definitions
            
        Returns:
            Dictionary of classification results for all papers
        """
        # Extract abstracts dictionary from papers
        all_abstracts = {paper['doi']: paper['abstract'] for paper in papers if 'abstract' in paper and isinstance(paper['abstract'], str) and paper['abstract'].strip()}
        logger.info(f"Processing {len(all_abstracts)} abstracts in batches of {self.batch_size}")
        
        # Process in batches
        labels = {}
        batch = {}
        batch_count = 0
        
        for i, (doi, abstract) in enumerate(all_abstracts.items()):
            batch[doi] = abstract
            
            # When batch is full or we reach the end of abstracts, process the batch
            if len(batch) >= self.batch_size or i == len(all_abstracts) - 1:
                batch_count += 1
                logger.info(f"Processing batch {batch_count} with {len(batch)} abstracts")
                
                # Add exponential backoff for API rate limiting
                max_retries = 3
                retry_count = 0
                success = False
                
                while not success and retry_count < max_retries:
                    try:
                        batch_results = self.classify_batch(batch, study_types, poverty_contexts, mechanisms)
                        
                        if batch_results:
                            # Parse the JSON response
                            try:
                                classification_results = json.loads(batch_results)
                                labels.update(classification_results)
                                success = True
                            except json.JSONDecodeError:
                                start_idx = batch_results.find('{')
                                end_idx = batch_results.rfind('}') + 1
                                if start_idx >= 0 and end_idx > start_idx:
                                    json_str = batch_results[start_idx:end_idx]
                                    classification_results = json.loads(json_str)
                                    labels.update(classification_results)
                                    success = True
                                else:
                                    raise ValueError("Could not extract valid JSON from response")
                        else:
                            retry_count += 1
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Retry {retry_count}/{max_retries}. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                    except Exception as e:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.error(f"Error processing batch: {e}. Retry {retry_count}/{max_retries}. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                
                if not success:
                    logger.error(f"Failed to process batch after {max_retries} retries")
                
                # Clear batch for next iteration
                batch = {}
                
                # Add a small delay between batches to avoid API rate limits
                time.sleep(1)

        # make a copy of papers to avoid modifying the original list
        results = papers.copy()

        for paper in results:
            if paper['doi'] in labels:
                # Update all fields at once
                paper.update({
                    'study_type': labels[paper['doi']]['study_type'],
                    'poverty_context': labels[paper['doi']]['poverty_context'],
                    'mechanism': labels[paper['doi']]['mechanism']
                })

        logger.info(f"Classification complete. Processed {len(results)} abstracts successfully")
    
        return results