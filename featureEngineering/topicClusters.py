import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run(papers,                
        n_clusters=10, 
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

    # Create a list of valid papers with same filtering criteria
    def is_valid_number(val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
    
    valid_papers = [paper for paper in papers if is_valid_number(paper.get('UMAP1')) and is_valid_number(paper.get('UMAP2'))]

    # Extract embeddings only from valid papers
    x = [float(paper['UMAP1']) for paper in valid_papers]
    if len(x) < 2:
        raise ValueError("Not enough valid data points for clustering")
    y = [float(paper['UMAP2']) for paper in valid_papers]
    umap_embeddings = np.array(list(zip(x, y)))

  
    
    # Apply K-means clustering to the UMAP embeddings
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=10  # Recommended default to avoid warnings
    )

      # Now assign clusters only to valid papers
    clusters = kmeans.fit_predict(umap_embeddings)
    for i, paper in enumerate(valid_papers):
        paper['cluster'] = clusters[i]

    
    # Return the set of unique cluster labels and the updated papers list
    return set(clusters), papers