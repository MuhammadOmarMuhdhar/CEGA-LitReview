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
    try:
        # Input validation remains the same...
        # Load model
        model = SentenceTransformer(model_name)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        # Prepare examples
        example_texts = [ex["text"] for ex in examples]
        example_labels = [ex["label"] for ex in examples]
        
        # Create and normalize embeddings - disable progress bar directly
        example_embeddings = model.encode(
            example_texts,
            convert_to_numpy=True,
            show_progress_bar=False  # Disable progress bar here
        )
        example_embeddings_normalized = normalize(example_embeddings)
        
        # Process texts in batches
        batch_size = 32
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Disable progress bar for batch encoding
            batch_embeddings = model.encode(
                batch_texts.tolist(),
                convert_to_numpy=True,
                show_progress_bar=False  # Disable progress bar here
            )
            batch_embeddings_normalized = normalize(batch_embeddings)
            
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric='cosine',
                weights='distance'
            )
            knn.fit(example_embeddings_normalized, example_labels)
            predictions = knn.predict(batch_embeddings_normalized)
            probabilities = knn.predict_proba(batch_embeddings_normalized)
            confidences = np.max(probabilities, axis=1)
            
            # Apply confidence threshold
            predictions[confidences < confidence_threshold] = None
            confidences[confidences < confidence_threshold] = None
            
            # Unpack probabilities into separate columns
            probability_columns = pd.DataFrame(
                probabilities,
                columns=[f'prob_{label}' for label in knn.classes_]  # Create one column per class
            )
            
            # Create the batch results DataFrame first
            batch_results = pd.DataFrame({
                'label': predictions,
            }).join(probability_columns)
            
            # Apply a post-processing rule
            for idx, row in batch_results.iterrows():
                if row['label'] == 'Not Related' and row['prob_Related'] > 0.3:
                    batch_results.at[idx, 'label'] = 'Related'
            
            all_results.append(batch_results)
        
        results = pd.concat(all_results, ignore_index=True)
        return results
    
    except Exception as e:
        logger.error(f"Unexpected error in classify: {str(e)}\n{traceback.format_exc()}")
        raise