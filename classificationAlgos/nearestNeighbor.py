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
    batch_size: int = 1000,
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
        
        # Final papers
        papers = []
        
        # Process texts in batches
        total_texts = len(texts)
        logger.info(f"Processing {total_texts} texts in batches of {batch_size}")
        
        for batch_start in range(0, total_texts, batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch_texts = texts[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size} "
                       f"(items {batch_start + 1}-{batch_end})")
            
            # Extract embeddings for the batch
            batch_embeddings = []
            batch_items = []
            
            for item in batch_texts:
                embedding = item['embedding'] if 'embedding' in item else None
                if embedding is not None:
                    # Convert embedding to list if it's a numpy array
                    embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    batch_embeddings.append(embedding)
                    batch_items.append(item)
            
            if not batch_embeddings:
                logger.warning(f"No valid embeddings found in batch {batch_start//batch_size + 1}")
                continue
            
            # Convert to numpy array for batch prediction
            embeddings_array = np.array(batch_embeddings)
            
            # Get predictions and confidence scores for the entire batch
            predicted_labels = knn.predict(embeddings_array)
            probabilities = knn.predict_proba(embeddings_array)
            confidences = np.max(probabilities, axis=1)
            
            # Process results for this batch
            batch_papers = []
            for i, (item, predicted_label, confidence) in enumerate(zip(batch_items, predicted_labels, confidences)):
                # Only keep papers above the confidence threshold
                if confidence >= confidence_threshold:
                    # Add relevant info to the paper
                    item['predicted_label'] = predicted_label
                    item['confidence'] = float(confidence)  # Convert numpy float to Python float
                    batch_papers.append(item)
            
            papers.extend(batch_papers)
            logger.info(f"Batch {batch_start//batch_size + 1}: classified {len(batch_items)} items, "
                       f"kept {len(batch_papers)} above threshold {confidence_threshold}")
        
        logger.info(f"Classification complete: processed {total_texts} texts, "
                   f"kept {len(papers)} above threshold {confidence_threshold}")
        return papers
        
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error
        return []