from typing import Union, Optional
import numpy as np
import pandas as pd
from umap import UMAP
from sentence_transformers import SentenceTransformer

def model(
    df: pd.DataFrame, 
    label_column: str, 
    umap_components: int = 2, 
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    random_state: Optional[int] = 42,
    min_dist: float = 0.1,
    n_neighbors: int = 15
) -> pd.DataFrame:
    """
    Create UMAP embeddings from text data using a sentence transformer.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing text data
    label_column : str
        Name of the column containing text to be embedded
    umap_components : int, optional (default=2)
        Number of dimensions for UMAP reduction
    model_name : str, optional (default='sentence-transformers/all-MiniLM-L6-v2')
        Name of the sentence transformer model to use
    random_state : int, optional (default=42)
        Seed for reproducibility
    min_dist : float, optional (default=0.1)
        Minimum distance between points in UMAP
    n_neighbors : int, optional (default=15)
        Number of neighboring points to consider in UMAP
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with UMAP coordinates
    
    Raises:
    -------
    ValueError
        If the specified label column does not exist in the dataframe
        If the dataframe is empty
    """
    # Validate input
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in the dataframe")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Create a copy to avoid modifying the original dataframe
    dataframe = df.copy()
    
    # Extract text and handle potential NaN values
    text = dataframe[label_column].fillna('')
    
    # Validate text data
    if text.empty:
        raise ValueError("No text data found in the specified column")
    
    # Initialize models with consistent random state
    umap_instance = UMAP(
        n_components=umap_components, 
        random_state=random_state,
        min_dist=min_dist,
        n_neighbors=n_neighbors
    )
    
    model = SentenceTransformer(model_name)
    
    # Encode text to embeddings
    try:
        encoded_text = model.encode(text.tolist(), show_progress_bar=False)
    except Exception as e:
        raise RuntimeError(f"Error during text encoding: {str(e)}")
    
    # Reduce dimensionality
    try:
        reduced_text = umap_instance.fit_transform(encoded_text)
    except Exception as e:
        raise RuntimeError(f"Error during UMAP dimensionality reduction: {str(e)}")
    
    # Create output dataframe
    umap_coords = pd.DataFrame(
        reduced_text, 
        columns=[f'UMAP{i+1}' for i in range(umap_components)]
    )
    
    return umap_coords