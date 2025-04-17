
import google.generativeai as genai
import json 
import time
import logging
from tqdm import tqdm
tqdm = lambda x: x

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model(topic_clusters, api_key, model_name="gemini-1.5-flash"):
    """
    Generate unique labels for topic clusters based on their research paper titles.
    
    Parameters:
    -----------
    topic_clusters : pandas.DataFrame
        DataFrame containing topic clusters with a 'title' column of lists of paper titles
    api_key : str
        Google Generative AI API key
    model_name : str, optional
        Name of the Gemini model to use (default is "gemini-1.5-flash")
    
    Returns:
    --------
    list
        A list of generated unique labels for each topic cluster
    """
    def generate_content_cached(prompt, model_name=model_name):
        """
        Generates content using a generative model with caching to avoid redundant calls.
        
        Parameters:
        -----------
        prompt : str
            The prompt to generate content for
        model_name : str, optional
            Name of the Gemini model to use
        
        Returns:
        --------
        str
            Generated content
        """
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Use a dictionary to cache results
        if not hasattr(generate_content_cached, 'llm_cache'):
            generate_content_cached.llm_cache = {}
        
        # Check if the prompt is already cached
        if prompt in generate_content_cached.llm_cache:
            return generate_content_cached.llm_cache[prompt]
        
        # Wait to manage rate limits
        time.sleep(10)
        
        # Initialize the generative model
        gemini_model = genai.GenerativeModel(model_name)
        
        # Generate content
        output = gemini_model.generate_content(prompt).text
        
        # Cache the output
        generate_content_cached.llm_cache[prompt] = output
        
        return output

    def summarize_contour_area(text, other_labels):
        """
        Generate a unique label for a research paper topic area.
        
        Parameters:
        -----------
        text : str
            Concatenated titles of research papers in the topic area
        other_labels : list
            Labels of previously processed topic areas
        
        Returns:
        --------
        str
            Generated label for the topic area
        """
        template = """
You are an AI system designed to analyze and generate labels for topic areas of research papers, 
represented in a 2D embedding landscape. Each group of research papers share similar themes, 
indicating a related research focus.
YOUR MAIN TASK:
Generate a label that distinctly characterizes this topic area of research, ensuring that the label is **unique** compared to the labels of other contour areas. 
Steps:
1. Review the research paper titles within this topic area to identify key recurring themes and perspectives.
2. Compare these themes to those from other topic areas to **ensure that this label is unique** and does not overlap with others.
Research papers in this contour area:
{text}
Labels from other contour areas:
{other_labels}
Your output must be a valid JSON object in this format:
{{"title": "<Your descriptive label here>"}}
Requirements for the label:
- 1 - 3 words long
- Accurately reflects what makes this research area unique
- **Clearly distinguishes this label** from the labels of other topic areas, ensuring it is **uniquely identifiable**
"""
        
        # Format the other labels
        other_labels_text = "\n".join([f"- {label}" for label in (other_labels or [])])
        if not other_labels_text:
            other_labels_text = "No other labels available yet."
        
        # Format the prompt with the input text
        prompt = template.format(
            text=text,
            other_labels=other_labels_text,
        )
        
        # Generate the content using the cached function
        return generate_content_cached(prompt)

    # Initialize labels list
    labels = []
    
    # Process each topic cluster
    for i, row in topic_clusters.iterrows():
        try:
            # Combine titles for the current cluster
            text = "\n".join(row['title'])
            
            # Generate label
            output = summarize_contour_area(text, labels)
            
            # Parse the output
            if isinstance(output, str):
                label_result = output.strip()
                label_result = label_result.lstrip('```json').rstrip('```')
                label_result = label_result.strip()
               
                try:
                    # Try to parse as JSON
                    label_data = json.loads(label_result)
                    label = label_data['title']
                    labels.append(label)
                except Exception as e:
                    logger.error(f"Error parsing label JSON: {str(e)}")
                    labels.append(output)
        
        except Exception as e:
            logger.error(f"Error processing cluster {i}: {str(e)}")
            labels.append(f"Cluster_{i}_Label")
    
    return labels
