import pandas as pd
import json
import logging
from data import openAlex, googleSheets, bigQuery
from featureEngineering import encoder, labels
from classificationAlgos import nearestNeighbor, largelanguageModel
from data.ETL.utils import clustering, density
import streamlit as st
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLPipeline():
    def __init__(self, api_key, credentials_json, spreadsheet_id_json, limit=None, batch_size=100):
        self.scraper = openAlex.Scraper()
        self.api_key = api_key
        self.batch_size = batch_size
        self.classification_model = largelanguageModel.GeminiModel(
                api_key=self.api_key,
                batch_size=self.batch_size,
            )
        self.label_model = labels.GeminiModel(
             api_key=self.api_key
        )
        self.limit = limit
        self.categories = self._load_categories()
        self.training_data = self._load_examples()
        self.labels = self._load_labels()
        self.study_types = self.labels['study_types']
        self.poverty_contexts = self.labels['poverty_contexts']
        self.mechanisms = self.labels['mechanisms']
        self.credentials_json = credentials_json
        self.google_sheets = bigQuery.Client(credentials_json=self.credentials_json)
        self.spreadsheet_id_json = spreadsheet_id_json

    def _load_examples(self):
        """Load example papers from a JSON file."""
        try:
            with open('data/trainingData/relevantExamples.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading examples: {e}")
            return []
        
    def _load_labels(self):
        """Load risk types from the JSON file."""
        try:
            with open('data/trainingData/labels.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return {}
        
    def _load_categories(self):
        """Load categories from a JSON file."""
        try:
            with open('data/trainingData/categories.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return {}
        
    def _scraper(self, start_date, end_date): 
        """Scrape data from OpenAlex."""
        try:
            extracted_papers = []
            for key, value in self.categories.items():
                for keyword in value: 
                    extracted_papers.extend(self.scraper.run(
                        start_date=start_date,
                        end_date=end_date,
                        broad_field=key, 
                        keyword=keyword, 
                        limit=self.limit))
            return extracted_papers
        except Exception as e:
            logger.error(f"Error scraping data: {e}")
            return []
        
    def _encoder(self, papers):
        """Encode the scraped papers."""
        try:
            encoded_papers = encoder.run(papers)
            return encoded_papers
        except Exception as e:
            logger.error(f"Error encoding data: {e}")
            return []
        
    def _relevance_classifier(self, papers):
        """Classify the relevance of the papers."""
        try:  
            relevant_papers = nearestNeighbor.classify(papers, self.training_data)
            return relevant_papers
        except Exception as e:
            logger.error(f"Error classifying relevance: {e}")
            return []
        
    def _label_classifier(self, papers):
        """Classify the papers using the LLM."""
        try:
            papers_labeled = self.classification_model.classify(papers, 
                                                             self.study_types, 
                                                             self.poverty_contexts, 
                                                             self.mechanisms)
            return papers_labeled
        except Exception as e:
            logger.error(f"Error classifying papers: {e}")
            return []
        
    def _format_dataframe(self, df):
        """ Clean the DataFrame for Google Sheets compatibility."""

        clean_df = df.copy()
        
        # Function to clean individual values
        def clean_value(val):
            # Convert NaN to empty string
            if pd.isna(val):
                return ""
            
            # Handle numeric values with special formatting
            if isinstance(val, (int, float)):
                # Remove problematic characters in numeric values
                return val
                
            # Convert to string and clean
            try:
                str_val = str(val)
                # Remove problematic characters and control characters
                str_val = str_val.replace("^", "").strip()
                return str_val
            except:
                return ""
        
        # Apply cleaning to each cell in the dataframe
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(clean_value)
        
        # Handle lists and dictionaries that need JSON serialization
        for col in clean_df.columns:
            if clean_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                # Convert lists/dicts to strings
                clean_df[col] = clean_df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        
        return clean_df
            
    def run(self, start_date, end_date):
        """
        Run the entire pipeline to scrape data and perform feature engineering.
        Returns:
        tuple: (density_dict, clusters_with_text, papers_with_clusters)
        """
        try:
            # Step 1: Scrape data
            with st.spinner("Extracting Research Papers..."):
                logger.info("Scraping data...")
                scraped_papers = self._scraper(
                    start_date=start_date,
                    end_date=end_date
                )
                if not scraped_papers:
                    logger.warning("No papers were scraped")
                    return {}, {}, []
            
            # Step 2: Encode data
            with st.spinner("Filtering Irrelevant Papers..."):
                logger.info("Encoding data...")
                encoded_papers = self._encoder(scraped_papers)
                if not encoded_papers:
                    logger.warning("No papers were encoded, using original papers")
                    encoded_papers = scraped_papers
            
            # Step 3: Classify relevance
            with st.spinner("Filtering Irrelevant Papers..."):
                logger.info("Classifying relevance...")
                relevant_papers = self._relevance_classifier(encoded_papers)
                if not relevant_papers:
                    logger.warning("No relevant papers found, using all papers")
                    relevant_papers = encoded_papers
                # only keep relevant papers
                relevant_papers = [paper for paper in relevant_papers if paper['predicted_label'] == 'Related']
            
            # Step 4: Classify using LLM
            with st.spinner(f"{len(relevant_papers)} Papers Extracted, Performing Feature Engineering..."):
                logger.info("Classifying papers using LLM...")
                papers_labeled = self._label_classifier(relevant_papers)
                if not papers_labeled:
                    logger.warning("No papers were labeled, using previous step results")
                    papers_labeled = relevant_papers
                
                # Check for empty papers
                if not papers_labeled:
                    logger.warning("No papers were labeled, returning empty results")
                    return {}, {}, []
                    
                ## Save papers to Google Sheets
                papers_df = pd.DataFrame(papers_labeled)
                papers_df = papers_df[['doi', 'title', 'authors', 'abstract', 'keyword', 'publication', 'country', 'date', 'institution', 'cited_by_count', 'citing_works', 'referenced_works', 'study_type', 'poverty_context', 'mechanism', 'embedding', 'UMAP1', 'UMAP2']]
                # Replace NaN values
                papers_df = papers_df.fillna("")
                # Check for problematic data in text columns
                for col in ['title', 'authors', 'abstract', 'keyword', 'publication', 'country', 'institution', 'study_type', 'poverty_context', 'mechanism' ]:
                    papers_df[col] = papers_df[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
                papers_df['cited_by_count'] = papers_df['cited_by_count'].astype(str)
                papers_df['citing_works'] = papers_df['citing_works'].astype(str)
                papers_df['embedding'] = papers_df['embedding'].astype(str)
                papers_df['UMAP1'] = papers_df['UMAP1'].astype(str)
                papers_df['UMAP2'] = papers_df['UMAP2'].astype(str)
                
                self.google_sheets.append(
                    df=papers_df,
                    spreadsheet_id=self.spreadsheet_id_json['papers']
                )
            
            # Step 5: Perform topic clustering
            with st.spinner("Labeling Poverty Contexts, Mechanisms, and Study Types..."):
                logger.info("Performing topic clustering...")
                clustering.run(
                    google_sheets=self.google_sheets,
                    label_model=self.label_model,
                    spreadsheet_id_json=self.spreadsheet_id_json
                )

            time.sleep(2)
                
            return papers_df
        except Exception as e:
            logger.error(f"Error in the pipeline: {e}")
            st.error(f"Error in the pipeline: {e}")
            return {}, {}, []