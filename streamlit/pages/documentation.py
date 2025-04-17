import streamlit as st 


# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebar"] {
#             width: 170px !important;
#             min-width: 170px !important;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown("""
          
            ## **Documentation**  

            This documentation provides a technical guide to our **Interactive Literature Review Tool**, detailing its architecture, data sources, and visualization methods.

            ### What You’ll Find in This Documentation  

            - **Using the Tool** – Instructions on navigating the interface and interpreting visualizations.  
            - **Data & Sources** – Information on data collection, processing, and updates.  
            - **Technical Details** – Overview of the underlying technology, including APIs, databases, and algorithms.  

            If you have questions or need assistance, please reach out to our team at **[insert contact info]**.

        
            """)


tab1, tab2, tab3, tab4= st.tabs(["Data Sourcing & Processing", "Document Classification", "MISC 1", "MISC 2" ])




with tab1:
    # Page Title & Overview
    st.markdown("### Data Collection from OpenAlex API")
    st.markdown("""
                
    This section explains how our system sources, processes, and stores research papers. The workflow is broken into several steps—from API extraction to data normalization and storage.
                

    
    """)

    # Supported APIs
    st.markdown("#### 1. OpenAlex API")
    st.markdown("""
    Papers are sourced from the **OpenAlex API**, a comprehensive and freely accessible database that provides metadata on academic papers, journals, authors, and institutions.  
                
    - **Authentication:** Not required  
    - **Rate Limits:**  
      - 10 requests per second (burst limit)  
      - 100,000 requests per 24-hour period  
    - **Documentation:** [OpenAlex API Docs](https://docs.openalex.org)
                
    The papers are extracted based on a set of keywords, grouped into thematic categories. The categories cover various aspects of how poverty affects psychological and economic outcomes, including emotional health, beliefs, cognitive functions, and preferences.
                
    """)

    # Search Keywords Overview
    with st.expander("Search Keywords"):
        

        st.code('''{
    "Affective": [
        "Poverty and mental health",
        "Poverty and Depression",
        "Poverty and Anxiety",
        "Poverty and Stress",
        "Poverty and Happiness"
    ],
    "Beliefs": [
        "Poverty and Beliefs",
        "Poverty and Internalized stigma",
        "Poverty and Mindset",
        "Poverty and self-efficacy",
        "Poverty and locus of control",
        "Poverty and self concept",
        "Poverty and self esteem",
        "Poverty and Optimism",
        "Poverty and Aspirations"
    ],
    "Cognitive function": [
        "Poverty and Cognitive function",
        "Poverty and Cognition",
        "Poverty and Cognitive flexibility",
        "Poverty and Executive control",
        "Poverty and Memory",
        "Poverty and working memory",
        "Poverty and Fluid intelligence",
        "Poverty and Attention"
    ],
    "Preferences": [
        "Poverty and Time preference",
        "Poverty and Risk preference"
    ]
}''', language='json')

    # Data Structure
    st.markdown("#### 2. Data Structure")
    st.markdown("Each research paper is extracted in JSON format with the following schema:")
    st.code('''{
    "doi": "string",          // Digital Object Identifier (unique)
    "title": "string",        // Full academic paper title
    "link": "string",         // URL to access the paper
    "authors": ["string"],    // Array of author names
    "keyword": "string",      // Search keyword used
    "publication": "string",  // Journal or conference name
    "country": "string",      // Country of publication (if available)
    "date": "string",         // Publication year (YYYY format)
    "field": "string",        // Primary academic discipline (e.g., psychology, economics)
    "institution": "string",  // Affiliated research institution (if available)
    "abstract": "string",     // Summary/abstract of the paper
    "cited_by_count": "integer", // Total citation count
    "citing_works": ["string"],  // DOIs of papers citing this work
    "referenced_works": ["string"] // DOIs of works cited in this paper
}''', language='json')

    # Search Keywords Details
    

    # Data Storage & Schema
    st.markdown("#### 3. Data Storage & Schema")
    st.markdown("To enable efficient cross-referencing, the dataset is normalized into multiple tables using the **DOI** as the primary key.")

    st.markdown("##### Authors Table")
    st.markdown("""
    | **DOI**    | **Authors**   |
    |------------|---------------|
    | "string"   | ["string"]    |
    """)

    st.markdown("##### Citations Table")
    st.markdown("""
    | **DOI**    | **Cited By Count** | **Referenced Works** | **Citing Works**   |
    |------------|--------------------|----------------------|--------------------|
    | "string"   | "integer"          | ["string"]           | ["string"]         |
    """)

    st.markdown("##### Institutions Table")
    st.markdown("""
    | **DOI**    | **Institution** | **Country**  |
    |------------|-----------------|--------------|
    | "string"   | "string"        | "string"     |
    """)

    st.markdown("##### Publications Table")
    st.markdown("""
    | **DOI**    | **Title**    | **Abstract**  | **Publication** | **Field**  | **Keyword**  |
    |------------|--------------|---------------|-----------------|------------|--------------|
    | "string"   | "string"     | "string"      | "string"        | "string"   | "string"     |
    """)

    # Usage Notes & Data Dictionary
    st.markdown("#### 4. Usage Notes")
    st.markdown("""
    - **DOI** acts as the primary key to join data across tables.  
    - Some fields (e.g., `country`, `institution`) might be missing in certain entries.  
    - The citation network can be constructed using the `citing_works` and `referenced_works` arrays.  
    - Publication dates are stored in the `YYYY` format.
    """)

with tab2:
    
   st.markdown("""
### Classification Algorithms

This document provides an overview of the classification algorithms implemented in this project, focusing on functionality, configuration, and usage.

#### Implemented Algorithm

- **Few-Shot Text Classification**  
  A method leveraging sentence transformers for text embedding and K-Nearest Neighbors (KNN) for classification with minimal labeled examples.

-----
               
#### Few-Shot Text Classification

This module enables few-shot classification of text data by utilizing sentence embeddings from a pretained transformer model and a KNN-based classification approach.

##### Dependencies

- `pandas`
- `numpy`
- `sentence_transformers`
- `scikit-learn`
- `logging`
- `typing`

##### Classification Process

1. **Model Loading**  
   - Loads the specified sentence transformer model for embedding generation.  

2. **Embedding Generation**  
   - Converts both input texts and labeled examples into numerical embeddings.  

3. **Batch Processing**  
   - Handles large datasets efficiently by processing inputs in batches to prevent memory issues.  

4. **Classification via KNN**  
   - Fits a KNN classifier to the example embeddings.  
   - Computes distances and retrieves the closest neighbors.  
   - Assigns labels based on majority voting among neighbors.  

6. **Confidence Filtering**  
   - Filters results based on the confidence threshold.  
   - If confidence falls below the threshold, the label is set to `None`.  

##### Usage Example

```python
import pandas as pd
from your_module import classify

# Input texts
texts = pd.Series([
    "The government increased interest rates again.",
    "Recent studies highlight the impact of meditation.",
    "Quantum computing is advancing at a rapid pace."
])

# Few-shot examples
# In practice, minimum 3 examples per label are recommended for robust classification. 
examples = [
    {"text": "Inflation rates and fiscal policies.", "label": "economics"},
    {"text": "Cognitive behavioral therapy and mindfulness.", "label": "psychology"},
    {"text": "Advancements in artificial intelligence and quantum computing.", "label": "technology"}
]

# Perform classification
results = classify(texts, examples)

# Display results
print(results)
               
|                         text                         predicted_label | confidence |
|-----------------------------------------------------|----------------|------------|
| The government increased interest rates again.      | economics      | 0.87       |
| Recent studies highlight the impact of meditation.  | psychology     | 0.92       |
| Quantum computing is advancing at a rapid pace.     | technology     | 0.89       |


""")
   
   