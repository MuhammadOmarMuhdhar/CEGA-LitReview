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
    texts: pd.Series,
    examples: List[Dict[str, str]],
    n_neighbors: int = 2,
    confidence_threshold: float = 0.2,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> pd.DataFrame:
    """
    Perform few-shot classification on texts using provided examples
    
    Args:
        texts: Series of texts to classify
        examples: List of dicts with 'text' and 'label' keys
        n_neighbors: Number of neighbors for KNN
        confidence_threshold: Minimum confidence threshold
        model_name: Name of sentence transformer model
    
    Returns:
        DataFrame with predictions and confidence scores
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If model loading or prediction fails
    """
    try:
        # Input validation
        if not isinstance(texts, pd.Series):
            raise ValueError("'texts' must be a pandas Series")
        
        if not examples or not all('text' in ex and 'label' in ex for ex in examples):
            raise ValueError("'examples' must be a non-empty list of dicts with 'text' and 'label' keys")
            
        if n_neighbors < 1 or n_neighbors > len(examples):
            raise ValueError(f"'n_neighbors' must be between 1 and {len(examples)}")
            
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("'confidence_threshold' must be between 0 and 1")

        # Load model
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Prepare examples
        example_texts = [ex["text"] for ex in examples]
        example_labels = [ex["label"] for ex in examples]

        # Create and normalize embeddings
        try:
            example_embeddings = model.encode(example_texts, convert_to_numpy=True)
            example_embeddings_normalized = normalize(example_embeddings)
            
            # Process texts in batches to prevent memory issues
            batch_size = 32
            all_results = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts.tolist(), convert_to_numpy=True)
                batch_embeddings_normalized = normalize(batch_embeddings)
                
                # Initialize and fit KNN for each batch
                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    metric='cosine',
                    weights='distance'
                )
                knn.fit(example_embeddings_normalized, example_labels)
                
                # Get predictions and confidences
                predictions = knn.predict(batch_embeddings_normalized)
                probabilities = knn.predict_proba(batch_embeddings_normalized)
                confidences = np.max(probabilities, axis=1)
                
                # Apply confidence threshold
                predictions[confidences < confidence_threshold] = None
                confidences[confidences < confidence_threshold] = None
                
                batch_results = pd.DataFrame({
                    'label': predictions,
                    'confidence': confidences
                })
                all_results.append(batch_results)
            
            results = pd.concat(all_results, ignore_index=True)
            return results

        except Exception as e:
            logger.error(f"Failed to process embeddings: {str(e)}")
            raise RuntimeError(f"Failed to process embeddings: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in classify: {str(e)}\n{traceback.format_exc()}")
        raise