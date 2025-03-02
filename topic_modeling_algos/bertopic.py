import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('topic_modeling')

def model(df: pd.DataFrame, text_column: str, 
                  n_components: int = 2, 
                  n_neighbors: int = 3, 
                  min_cluster_size: int = 15,
                  min_samples: int = 15,
                  random_state: int = 42) -> Tuple[pd.DataFrame, Optional[BERTopic]]:
    """
    Perform topic modeling on text data using BERTopic.
    
    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text
        n_components: Number of components for UMAP dimensionality reduction
        n_neighbors: Number of neighbors for UMAP
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - DataFrame with original data plus topic modeling results
            - Fitted BERTopic model
    """    
    # Check if the text column exists in the DataFrame
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in DataFrame")
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Check if DataFrame is empty
    if df.empty:
        logger.error("DataFrame is empty")
        raise ValueError("Cannot perform topic modeling on empty DataFrame")
    
    # Check for missing values in the text column
    missing_values = df[text_column].isna().sum()
    if missing_values > 0:
        logger.warning(f"Found {missing_values} missing values in '{text_column}'. These will be converted to empty strings.")
        df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        df[text_column] = df[text_column].fillna("")
    
    try:
        with tqdm(total=3, leave=False, desc="BERTopic model") as pbar:
            representation_model = KeyBERTInspired()
            
            umap_model = UMAP(
                n_components=n_components, 
                n_neighbors=n_neighbors, 
                random_state=random_state
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,  
                min_samples= min_samples,       
                metric='euclidean',
                cluster_selection_method='eom', 
                prediction_data=True
            )
            
            # Initialize BERTopic 
            topic_model = BERTopic(
                language="english",
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                calculate_probabilities=True,
                representation_model=representation_model,
                verbose=True
            )
            
            # Convert text to string and fit the model
            texts = df[text_column].astype(str).tolist()
            
            fitted_model = topic_model.fit(texts)
            pbar.update(1)
            
            # Transform the text column using the fitted model
            topics, probs = fitted_model.transform(texts)
            pbar.update(1)
            
            # Get the topic distribution for each document
            try:
                topic_distr, _ = fitted_model.approximate_distribution(texts)
            except Exception as e:
                logger.error(f"Error in approximate_distribution: {str(e)}")
                logger.warning("Using empty topic distributions as fallback")
                # Create empty distribution as fallback
                num_topics = len(fitted_model.get_topic_freq())
                topic_distr = np.zeros((len(texts), num_topics))
            pbar.update(1)
            
            # Extract UMAP embeddings from the fitted model
            if hasattr(fitted_model.umap_model, 'embedding_'):
                umap_embeddings = fitted_model.umap_model.embedding_
                
                # Convert UMAP embeddings to a DataFrame
                umap_df = pd.DataFrame(umap_embeddings)
                umap_df.columns = umap_df.columns.map(lambda x: f'umap_{x}')
            else:
                logger.warning("UMAP embeddings not found in the model")
                umap_df = pd.DataFrame(index=df.index)
            
            # Convert topic distribution to a DataFrame
            probs_df = pd.DataFrame(topic_distr)
            probs_df.columns = probs_df.columns.map(lambda x: f'topic_{x}')
            
            # Convert topics to a DataFrame with a single 'topic' column
            topics_df = pd.DataFrame(topics, columns=['topic'])
            
            # Ensure all DataFrames have the same length
            if len(df) != len(topics_df) or len(df) != len(umap_df) or len(df) != len(probs_df):
                logger.error("Length mismatch in resulting DataFrames")
                raise ValueError("Length mismatch in resulting DataFrames")
            
            # Concatenate the original DataFrame with topics, UMAP embeddings, and topic probabilities
            df_topics = pd.concat([df.reset_index(drop=True), 
                                  topics_df.reset_index(drop=True), 
                                  umap_df.reset_index(drop=True), 
                                  probs_df.reset_index(drop=True)], axis=1)
            
            return df_topics, fitted_model
            
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}", exc_info=True)
        raise RuntimeError(f"Topic modeling failed: {str(e)}") from e