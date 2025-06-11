import pandas as pd
import logging
from collections import defaultdict
from statistics import mean
from featureEngineering import topicClusters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run(google_sheets, label_model, spreadsheet_id_json):
    """
    Perform topic clustering on the papers and save results to Google Sheets.
    
    Args:
        google_sheets: GoogleSheets API instance
        label_model: Gemini model for labeling
        spreadsheet_id_json: Dictionary containing spreadsheet IDs
        
    Returns:
        tuple: (clusters_with_text, papers_with_clusters)
    """
    try:
        # Load papers from Google Sheets
        papers = google_sheets.read(spreadsheet_id_json['papers'])
        papers = papers.to_dict(orient='records')

        # # Run topic clustering
        clusters, papers_with_clusters = topicClusters.run(papers)

        # Add UMAP coordinates
        clusters_with_text = defaultdict(dict)
        for cluster in clusters:
            cluster_papers = [paper for paper in papers_with_clusters if paper.get('cluster') == cluster]
            
            if cluster_papers:
                clusters_with_text[cluster]['doi'] = [paper.get('doi', '') for paper in cluster_papers]
                clusters_with_text[cluster]['title'] = [paper.get('title', '') for paper in cluster_papers]
                
                try:
                    def is_valid_number(val):
                        try:
                            float(val)
                            return True
                        except (ValueError, TypeError):
                            return False
                    
                    # Here's the fix: use cluster_papers instead of papers
                    x = [float(paper['UMAP1']) for paper in cluster_papers if is_valid_number(paper.get('UMAP1'))]
                    y = [float(paper['UMAP2']) for paper in cluster_papers if is_valid_number(paper.get('UMAP2'))]
                    
                    # Make sure there are valid values to compute the mean
                    if x and y:
                        clusters_with_text[cluster]['umap_1_mean'] = mean(x)
                        clusters_with_text[cluster]['umap_2_mean'] = mean(y)
                    else:
                        # No valid values found, set to 0
                        clusters_with_text[cluster]['umap_1_mean'] = 0
                        clusters_with_text[cluster]['umap_2_mean'] = 0
                        
                except Exception as e:
                    clusters_with_text[cluster]['umap_1_mean'] = 0
                    clusters_with_text[cluster]['umap_2_mean'] = 0

        # Generate labels for each cluster
        _generate_labels(clusters_with_text, label_model)
        
        # Save clusters to Google Sheets
        clusters_df = pd.DataFrame(clusters_with_text)
        clusters_df = clusters_df.T
        
        google_sheets.replace(
            df=clusters_df,
            spreadsheet_id=spreadsheet_id_json['topics']
        )
        return clusters_df
        
    except Exception as e:
        logger.error(f"Error performing topic clustering: {e}")

def _generate_labels(clusters_with_text, label_model):
    """
    Generate labels for each cluster.
    
    Args:
        clusters_with_text: Dictionary containing cluster information
        label_model: GeminiModel for labeling
    """
    labels = []
    for key, value in clusters_with_text.items():
        try:
            text = value.get('title', [])
            if text:
                label = label_model.run(text, "cluster")
                labels.append(label)
                value['label'] = label
            else:
                value['label'] = "Unlabeled"
        except Exception as e:
            logger.warning(f"Error labeling cluster {key}: {e}")
            value['label'] = "Error"