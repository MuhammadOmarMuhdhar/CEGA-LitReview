import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Callable
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dynamic_lda')

def model(
    df: pd.DataFrame,
    text_column: str,
    topic_column: str,
    min_topics: int = 1,
    max_topics: int = 2,
    min_docs_per_topic: int = 10,
    topic_scaling_method: str = 'log',  # 'log', 'sqrt', 'linear'
    scaling_factor: float = 0.1,  # Used for linear scaling
    random_state: int = 42,
    max_features: int = 5000,
    max_df: float = 0.95,
    min_df: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Perform LDA topic modeling on text data within BERTopic clusters,
    dynamically adjusting the number of LDA topics based on cluster size.
    
    Args:
        df: DataFrame containing the data
        text_column: Name of the column containing text
        topic_column: Name of the column containing BERTopic topics
        min_topics: Minimum number of LDA topics per cluster
        max_topics: Maximum number of LDA topics per cluster
        min_docs_per_topic: Minimum documents required per LDA topic
        topic_scaling_method: Method to scale topics ('log', 'sqrt', 'linear')
        scaling_factor: Factor used for linear scaling (topics = cluster_size * scaling_factor)
        random_state: Random seed for reproducibility
        max_features: Maximum number of features for CountVectorizer
        max_df: Maximum document frequency for CountVectorizer
        min_df: Minimum document frequency for CountVectorizer
        
    Returns:
        Tuple containing:
            - DataFrame with original data plus LDA results
            - DataFrame with topic means and labels
            - Dictionary with topic information and keywords
    """
    
    # Input validation
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found in DataFrame")
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
    if topic_column not in df.columns:
        logger.error(f"Topic column '{topic_column}' not found in DataFrame")
        raise ValueError(f"Topic column '{topic_column}' not found in DataFrame")
    
    # Check if UMAP columns exist
    umap_columns = ['umap_0', 'umap_1']
    for col in umap_columns:
        if col not in df.columns:
            logger.warning(f"UMAP column '{col}' not found in DataFrame. Using zeros instead.")
            df[col] = 0.0
    
    # Initialize the output DataFrame (copy to avoid modifying the original)
    result_df = df.copy()
    result_df['lda_topic'] = -1
    result_df['lda_topic_prob'] = 0.0
    result_df['num_lda_topics'] = 0
    result_df['hierarchical_topic'] = ""
    
    # Initialize topic information dictionary
    topic_info = {}
    topic_words = {}
    
    # Get unique topics
    unique_topics = df[topic_column].unique()
    
    # Create custom stop words list based on the note
    # custom_stop_words = set(['poverty'])  # Add 'poverty' as a stop word
    
    # Create vectorizer for LDA
    vectorizer = CountVectorizer(
        max_features=max_features, 
        max_df=max_df, 
        min_df=min_df,
        stop_words='english'  # We'll add custom stop words later
    )
    
    # Process each topic cluster
    for topic_id in tqdm(unique_topics, desc="Processing topics"):
        # Filter documents for this topic
        topic_docs = df[df[topic_column] == topic_id]
        cluster_size = len(topic_docs)
        
        # Skip small clusters
        if cluster_size < min_topics * min_docs_per_topic:
            logger.warning(f"Topic {topic_id}: Only {cluster_size} documents. Minimum required: {min_topics * min_docs_per_topic}. Skipping.")
            continue
            
        # Calculate number of topics based on cluster size
        if topic_scaling_method == 'log':
            # Logarithmic scaling (e.g., log10(100) ≈ 2, log10(1000) ≈ 3)
            num_topics = max(min_topics, min(max_topics, int(math.log10(max(10, cluster_size)) * 3)))
        elif topic_scaling_method == 'sqrt':
            # Square root scaling
            num_topics = max(min_topics, min(max_topics, int(math.sqrt(cluster_size) / 2)))
        else:  # 'linear'
            # Linear scaling
            num_topics = max(min_topics, min(max_topics, int(cluster_size * scaling_factor)))
            
        # Ensure we have enough documents per topic
        max_possible_topics = max(1, cluster_size // min_docs_per_topic)
        num_topics = min(num_topics, max_possible_topics)
                
        try:
            # Prepare text data
            texts = topic_docs[text_column].astype(str).tolist()
            
            # Vectorize the text (add custom stop words)
            vectorizer = CountVectorizer(
                max_features=max_features, 
                max_df=max_df, 
                min_df=min_df,
                stop_words='english'
            )
            X = vectorizer.fit_transform(texts)
            
            # Fit LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=random_state,
                max_iter=15,
                learning_method='online',
                n_jobs=-1  # Use all available cores
            )
            
            # Get document-topic distributions
            topic_distributions = lda.fit_transform(X)
            
            # Get most probable topic and probability for each document
            dominant_topics = np.argmax(topic_distributions, axis=1)
            topic_probs = np.max(topic_distributions, axis=1)
            
            # Get topic keywords
            feature_names = vectorizer.get_feature_names_out()
            topic_keywords = {}
            topic_word_distributions = {}
            
            for lda_idx, topic_dist in enumerate(lda.components_):
                # Get top keywords
                top_indices = topic_dist.argsort()[:-11:-1]
                keywords = [feature_names[i] for i in top_indices]
                keyword_weights = [topic_dist[i] for i in top_indices]
                
                # Normalize weights
                total_weight = sum(keyword_weights)
                normalized_weights = [w/total_weight for w in keyword_weights]
                
                # Store keyword data
                topic_keywords[lda_idx] = keywords
                topic_word_distributions[lda_idx] = dict(zip(keywords, normalized_weights))
            
            # Store topic info
            topic_info[topic_id] = {
                'num_documents': cluster_size,
                'num_lda_topics': num_topics,
                'keywords': topic_keywords,
                'keyword_weights': topic_word_distributions
            }
            
            # Store top words for each topic
            for idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-4:-1]]
                topic_words[(topic_id, idx)] = ', '.join(top_words)
            
            # Prepare updates for the DataFrame
            lda_topics = []
            lda_probs = []
            lda_nums = []
            hierarchical_topics = []
            indices = topic_docs.index.tolist()
            
            for i in range(len(indices)):
                lda_topic_id = dominant_topics[i]
                lda_topics.append(int(lda_topic_id))
                lda_probs.append(float(topic_probs[i]))
                lda_nums.append(num_topics)
                hierarchical_topics.append(f"{topic_id}.{lda_topic_id}")
            
            # Update DataFrame efficiently
            result_df.loc[indices, 'lda_topic'] = lda_topics
            result_df.loc[indices, 'lda_topic_prob'] = lda_probs
            result_df.loc[indices, 'num_lda_topics'] = lda_nums
            result_df.loc[indices, 'hierarchical_topic'] = hierarchical_topics
                
        except Exception as e:
            logger.error(f"Error processing LDA for topic {topic_id}: {str(e)}", exc_info=True)
    
    # Create a summary of topic distribution
    topic_summary = []
    for topic_id, info in topic_info.items():
        for lda_id, keywords in info['keywords'].items():
            topic_summary.append({
                'bertopic_id': topic_id,
                'lda_id': lda_id,
                'hierarchical_id': f"{topic_id}.{lda_id}",
                'num_documents': info['num_documents'],
                'num_lda_topics': info['num_lda_topics'],
                'top_keywords': ', '.join(keywords[:5]),
                'all_keywords': keywords
            })
    
    # Create topic means DataFrame
    topic_means_df = result_df.groupby([topic_column, 'lda_topic'])[['umap_0', 'umap_1']].mean().reset_index()
    
    # Add labels to topic means
    topic_means_df['label'] = topic_means_df.apply(
        lambda row: topic_words.get((row[topic_column], row['lda_topic']), ''), 
        axis=1
    )
    
    # Create summary DataFrame
    topic_summary_df = pd.DataFrame(topic_summary) if topic_summary else pd.DataFrame()

    return result_df, topic_means_df, topic_info