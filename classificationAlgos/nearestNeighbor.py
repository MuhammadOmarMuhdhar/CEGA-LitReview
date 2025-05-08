import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import logging
from typing import List, Dict, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def classify(
    texts,
    examples,
    n_neighbors: int = 2,
    confidence_threshold: float = 0.2,
) -> List[Dict]:
    try:
        # Prepare examples
        example_embeddings = [paper['embedding'] for key, value in examples.items() for paper in value]
        example_labels = [paper['label'] for key, value in examples.items() for paper in value]
        
        # Initialize the model
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='cosine',
            weights='distance'
        )
        
        # Fit the model
        knn.fit(example_embeddings, example_labels)
        
        # final papers
        papers = []
        
        # check if each paper matches a label
        for item in texts:
            # Extract the text and embedding
            embedding = item['embedding'] if 'embedding' in item else None
            
            # turn the embedding into a list
            embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # classify the paper
            if embedding is not None:
                # Reshape embedding for sklearn if needed
                embedding_array = np.array([embedding])
                
                # Get predictions and confidence scores
                predicted_label = knn.predict(embedding_array)[0]
                probabilities = knn.predict_proba(embedding_array)[0]
                confidence = np.max(probabilities)
                
                # Only keep papers above the confidence threshold
                if confidence >= confidence_threshold:
                    # Add relevant info to the paper
                    item['predicted_label'] = predicted_label
                    item['confidence'] = float(confidence)  # Convert numpy float to Python float
                    
                    # Add to final papers list
                    papers.append(item)
        
        logger.info(f"Classified {len(texts)} papers, kept {len(papers)} above threshold {confidence_threshold}")
        
        return papers
        
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error
        return []