from sentence_transformers import SentenceTransformer
from umap import UMAP
import torch
import numpy as np

def run(papers,
        model_name='all-MiniLM-L6-v2',
        batch_size=32,
        umap_components=2,
        random_state=42,
        min_dist=0.1,
        n_neighbors=15,
        device=None):
    """
    Efficiently encode abstracts in batches using sentence transformers,
    and add UMAP dimensionality reduction.
    
    Parameters:
    -----------
    papers : list of dict
        List of paper dictionaries, each containing at least an 'abstract' key
    model_name : str, optional
        Name of the SentenceTransformer model to use (default: 'all-MiniLM-L6-v2')
    batch_size : int, optional
        Number of abstracts to process in each batch (default: 32)
    umap_components : int, optional
        Number of dimensions for UMAP reduction (default: 5)
    random_state : int, optional
        Random seed for UMAP for reproducibility (default: 42)
    min_dist : float, optional
        UMAP min_dist parameter controlling how tightly points are packed (default: 0.1)
    n_neighbors : int, optional
        UMAP n_neighbors parameter controlling local versus global structure (default: 15)
    device : str, optional
        Device to run the model on ('cpu', 'cuda', 'mps', etc.)
        If None, will use CUDA if available, otherwise CPU
        
    Returns:
    --------
    list of dict
        The input papers with 'embedding' and 'umap_embedding' fields added to each paper that has an abstract
    """
    # Load model
    model = SentenceTransformer(model_name)
    
    # Set device if specified
    if device:
        model = model.to(device)
    
    # Extract abstracts (skipping None or empty abstracts)
    valid_indices = []
    abstracts_to_encode = []
    
    for i, paper in enumerate(papers):
        abstract = paper.get('abstract')
        if abstract and isinstance(abstract, str) and abstract.strip():
            valid_indices.append(i)
            abstracts_to_encode.append(abstract)
    
    # Process abstracts in batches to get original embeddings
    original_embeddings = []
    
    for i in range(0, len(abstracts_to_encode), batch_size):
        batch = abstracts_to_encode[i:i+batch_size]
        batch_embeddings = model.encode(
            batch, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Convert to numpy for storage
        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings_np = batch_embeddings.cpu().numpy()
        else:
            batch_embeddings_np = np.array(batch_embeddings)
            
        original_embeddings.append(batch_embeddings_np)
    
    # Combine all batches
    if original_embeddings:
        all_embeddings = np.vstack(original_embeddings)
        
        # Apply UMAP transformation to the combined embeddings
        umap_instance = UMAP(
            n_components=umap_components,
            random_state=random_state,
            min_dist=min_dist,
            n_neighbors=n_neighbors
        )
        umap_embeddings = umap_instance.fit_transform(all_embeddings)
        
        # Convert both embeddings to list format
        original_embeddings_list = all_embeddings.tolist()
        umap_embeddings_list = umap_embeddings.tolist()
        
        # Assign both embeddings back to papers
        for idx, paper_idx in enumerate(valid_indices):
            papers[paper_idx]['embedding'] = original_embeddings_list[idx]
            papers[paper_idx]['UMAP1'] = umap_embeddings_list[idx][0]
            papers[paper_idx]['UMAP2'] = umap_embeddings_list[idx][1]
    
    return papers

# Example usage:
# paper_samples = [{'title': 'Paper 1', 'abstract': 'This is the first abstract.'},
#                  {'title': 'Paper 2', 'abstract': None},
#                  {'title': 'Paper 3', 'abstract': 'This is the third abstract.'}]
# processed_papers = run(paper_samples)