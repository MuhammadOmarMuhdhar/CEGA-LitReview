import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run(papers,                
        n_clusters=7, 
        random_state=42):
    """
    Perform K-means clustering on a list of papers based on their UMAP embeddings.

    Args:
        papers (list): A list of dictionaries, where each dictionary represents a paper 
                       and contains a 'umap_embedding' key with its embedding.
        n_clusters (int): The number of clusters to form. Default is 7.
        random_state (int): Random seed for reproducibility. Default is 42.
        scale_features (bool): Placeholder argument (not used in the current implementation). Default is True.

    Returns:
        tuple: A set of unique cluster labels and the updated list of papers with cluster assignments.
    """
    # Make a copy of the papers list to avoid modifying the original data
    papers_copy = papers.copy()

    # Extract UMAP embeddings from the papers
    umap_embeddings = [paper['umap_embedding'] for paper in papers_copy]   
    
    # Apply K-means clustering to the UMAP embeddings
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=10  # Recommended default to avoid warnings
    )
    
    # Predict cluster labels for each paper
    clusters = kmeans.fit_predict(umap_embeddings)

    # Add the cluster assignments back to the papers dictionary
    for i, paper in enumerate(papers_copy):
        paper['cluster'] = clusters[i]
    
    # Return the set of unique cluster labels and the updated papers list
    return set(clusters), papers_copy