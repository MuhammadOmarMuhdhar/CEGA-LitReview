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


tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["Sourcing Papers",  "Relevance Classification", "Google Sheets Database", "Data Dictionary", "Sankey", "Heat Map" ])



with tab1:

   st.markdown(f"""
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/openAlex.py)
               
    #### Sourcing Papers

    We source our research papers using the **OpenAlex API**, a large-scale, openly accessible academic database offering metadata on over 200 million scholarly works. OpenAlex provides a powerful interface to access publications, authors, institutions, journals, and related research metadata across all disciplines.

    ##### OpenAlex Integration

    Our system queries OpenAlex based on keyword categories, research fields, and date filters to ensure a focused and contextually relevant literature base. These queries are guided by a structured taxonomy developed specifically for poverty-related research.

    - **Platform:** [https://openalex.org](https://openalex.org)  
    - **API Docs:** [https://docs.openalex.org](https://docs.openalex.org)  
    - **Authentication:** Not required  
    - **Rate Limits:** 10 requests/sec; 100,000 requests/day  
    - **Response Format:** JSON with extensive metadata fields

    📁 **Keyword Taxonomy:**  
    We use a curated set of research themes to guide our search process.  
    👉 [View on GitHub](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/trainingData/categories.json)

    ##### Custom Scraper: Automated Literature Harvesting

    To operationalize large-scale, reproducible data collection from OpenAlex, we have developed a dedicated scraping module. This scraper ensures robust performance and data integrity throughout the ingestion pipeline.

    ##### Key Features

    - **Rate Limiting**: Built-in token bucket logic respects OpenAlex constraints
    - **Fault Tolerance**: Retries on network errors and handles 429/500 responses
    - **Concurrency**: Utilizes multithreading for parallel API calls
    - **Abstract Stitching**: Merges segmented abstracts into coherent text
    - **Full Metadata Extraction**: Collects DOIs, titles, abstracts, authors, and more
    - **Flexible Filtering**: Enables queries by keyword, research field, and date

    ##### Implementation Details
        
    - **Installation Requirements:**
        ```bash
        pip install requests urllib3
        ```
    - **Parameters:**
        - `requests_per_second` *(int)*: API throttle rate (default: 10)
        - `email` *(str)*: Contact email to include in headers per OpenAlex guidelines
        - `broad_field` *(str)*: Discipline or research domain (e.g., "Psychology")
        - `keyword` *(str)*: Search term
        - `start_date`, end_date *(str)*: Time range for publication dates
        - `limit` *(int)*: Max number of papers to retrieve
               
    - **Usage:**
        ```python
        # Initialize scraper with appropriate headers and limits
        extractor = Scraper(requests_per_second=10, email="your_email@example.com")

        # Retrieve papers matching the search criteria
        papers = extractor.run(
            broad_field="Psychology",
            keyword="cognitive",
            start_date="2020-01-01",
            end_date="2023-12-31",
            limit=100
        )
        ```
            

    """, unsafe_allow_html=True)
   
with tab2: 
   
   st.markdown(f"""
               
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/classificationAlgos/nearestNeighbor.py)


    #### Relevance Classification

    After collecting papers through the OpenAlex API, we use an **embedding-based classification pipeline** to determine whether each paper is relevant to our research focus. This ensures we retain only literature aligned with our domain-specific criteria.

    ##### Model Overview

    The classifier uses a **K-Nearest Neighbors (KNN)** model built on **SentenceTransformer embeddings**. It compares new papers to a curated set of labeled examples and assigns a predicted label with a confidence score.

    - **Embedding model**: [SentenceTransformer](https://www.sbert.net/)
    - **Classifier**: `sklearn.neighbors.KNeighborsClassifier`
    - **Distance metric**: Cosine similarity
    - **Filtering**: Only papers above a configurable confidence threshold are retained

    ##### How it Works
    - Embeddings are extracted from papers using a transformer model.
    - These are compared to a **manually labeled training set** of example papers.
    - The classifier predicts a category label for each paper and assigns a confidence score.
    - Papers that do not meet the threshold are discarded to ensure precision.
               
    ##### Design Considerations

    To maximize **coverage** in our literature review process, we prioritize **false positives over false negatives**. This means we are more willing to include papers that might be marginally relevant rather than risk missing potentially important ones.

    - As a result, the **confidence threshold is intentionally lowered** (e.g., 0.2), increasing sensitivity.
    - This decision allows for a broader pool of papers, though some may require manual post-review.


    ##### Manually Curated Training Data

    Our classifier is powered by a **hand-labeled training dataset**, reviewed to ensure conceptual relevance. 

    🔗 **View our labeled training set:** [GitHub - training data](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/trainingData/relevantExamples.json)

    
    ##### Implementation Details:
               
    - **Installation Requirements:**
        ```bash
        pip install pandas numpy scikit-learn sentence-transformers
                
        ```
    - **Parameters:**
        - `texts` (`List[Dict]`): A list of paper entries with precomputed embeddings to be classified.
        - `examples` (`Dict[str, List[Dict]]`): Manually curated and labeled training data used for fitting the KNN classifier.
        - `n_neighbors` (`int`, default=`2`): Number of neighbors to use in KNN; controls how local the classification decision is.
        - `confidence_threshold` (`float`, default=`0.2`): Minimum confidence level required to retain a paper. Lowering this increases sensitivity.

    - **Usage:**    
        ```python
        from classifier import classify
        import json

        # Load embeddings and training examples
        with open("data/trainingData/examples.json", "r") as f:
            examples = json.load(f)

        with open("data/interim/embedded_papers.json", "r") as f:
            papers = json.load(f)

        # Run classification
        relevant_papers = classify(
            texts=papers,
            examples=examples,
            n_neighbors=2,
            confidence_threshold=0.2
        )
        ```

               
               """, unsafe_allow_html=True)
   
with tab3:
   
   st.markdown(f"""
    
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/googleSheets.py)

    #### Google Sheets Database

    We use **Google Sheets as a primary database** for storing and managing the research papers sourced from the OpenAlex API. This approach combines the flexibility of spreadsheets with the power of automation, making the database both human-readable and machine-operable.

    ##### Why Google Sheets?

    - **Accessibility & Collaboration**
        - Real-time updates from automated scrapers
        - Multi-user editing with granular sharing controls
        - Accessible from any device, no installation required
        - Familiar interface for researchers with no technical background

    - **Technical Advantages**
        - Seamless integration via Google Sheets API
        - Flexible structure that supports varying metadata schemas
        - Built-in export to CSV, Excel, and PDF
        - Cloud-native storage with automatic versioning

    ##### Authentication & Setup

    To enable programmatic access, configure a [**Google Cloud Service Account**](https://cloud.google.com/iam/docs/service-account-creds). This grants secure access to your spreadsheets without needing manual sign-in.

    ##### Our Google Sheets API Wrapper

    We developed a robust API wrapper to streamline interaction with Google Sheets, ensuring scalable, fault-tolerant data management.

    **Key Features:**

    - Efficient **batch writing** to handle large document sets
    - Automatic **type conversion** for nested and complex fields
    - **List flattening** for fields like author arrays and keywords
    - Resilient **error handling** with logging for traceability
    - **Rate limit awareness** to avoid exceeding Google API quotas
    - Support for sheet **creation, replacement, and appending**

    ###### Implementation Details

    - **Installation Requirements:**
        ```bash
        pip install gspread google-auth pandas
        ```
    - **Parameters:**
        - Initialization
            - `credentials_json` *(dict or str)*: Path to or dictionary of Google Service Account credentials used for authenticating API access.
        - replace()
            - `df` *(pd.DataFrame)*: The complete dataset to write to the sheet.
            - `spreadsheet_id` *(str)*: The unique ID of the Google Sheets file.
            - `sheet_name` *(str)*: The name of the target sheet within the file.
            - `include_headers` *(bool, optional)*: Whether to include column headers in the output (default: `True`).
        - append()
            - `df` *(pd.DataFrame)*: New rows to be appended to the specified sheet.
            - `spreadsheet_id` *(str)*: The unique ID of the Google Sheets file.
            - `sheet_name` *(str)*: The name of the sheet to which data will be appended.
        - read()
            - `spreadsheet_id` *(str)*: The unique ID of the Google Sheets file.
            - `sheet_range` *(str)*: The A1 notation range (e.g., `"Sheet1!A1:E100"`) or just the sheet name.
            - `header_row` *(int, optional)*: Index of the row to use as column headers (default: `0`).
                
    - **Usage:**
        - Writing to google sheets:
            ```python
                
            from googleSheets import API

            # Initialize with Google Service Account credentials
            sheets_api = API(credentials_json)

            # Replace sheet contents with new data
            results = sheets_api.replace(
                df=papers_dataframe,
                spreadsheet_id="your_sheet_id",
                sheet_name="Literature_Review",
                include_headers=True
            )

            # Append new rows to existing sheet
            results = sheets_api.append(
                df=new_papers,
                spreadsheet_id="your_sheet_id",
                sheet_name="Recent_Papers"
            )
            ```
        - Reading from google sheets:
            ```python
            from googleSheets import API
               
            # Initialize with Google Service Account credentials
            sheets_api = API(credentials_json)

               
            # Read spreadsheet contents into a DataFrame
            papers_df = sheets_api.read(
                spreadsheet_id="your_sheet_id",
                sheet_range="Literature_Review",
                header_row=0
            )
            ```     
   
    """, unsafe_allow_html=True)
   
with tab4:
   
   st.markdown(f"""

    #### Data Dictionary

    This data dictionary describes the structure and content of each paper record stored in our Google Sheets database. Each row represents a single research paper with the following standardized fields:

    ##### Core Paper Information

    | Field | Type | Description | Example |
    |-------|------|-------------|---------|
    | `doi` | String | Digital Object Identifier without the URL prefix | `10.1037/dev0001234` |
    | `title` | String | Full title of the research paper | `"Cognitive Development in Early Childhood"` |
    | `abstract` | String | Paper abstract (cleaned and reconstructed from inverted index) | `"This study examines the relationship between..."` |

    ##### Author & Institutional Data

    | Field | Type | Description | Example |
    |-------|------|-------------|---------|
    | `authors` | List of Strings | All author names associated with the paper | `["John Smith", "Jane Doe", "Alex Johnson"]` |
    | `country` | List of Strings | Country codes for each author's primary institution | `["US", "GB", "CA"]` |
    | `institution` | List of Strings | Institution names for each author's primary affiliation | `["Harvard University", "Oxford University", "University of Toronto"]` |

    ##### Metadata

    | Field | Type | Description | Example |
    |-------|------|-------------|---------|
    | `date` | Integer | Year of publication | `2023` |
    | `field` | String | Broad academic field category | `"Psychology"` |
    | `keyword` | String | Search keyword used to find this paper | `"cognitive development"` |

    ##### Citation Metrics

    | Field | Type | Description | Example |
    |-------|------|-------------|---------|
    | `cited_by_count` | Integer | Number of times this paper has been cited | `45` |
    | `citing_works` | Null | Future field for papers that cite this work | `null` |
    | `referenced_works` | Null | Future field for papers referenced by this work | `null` |
               
    ##### Features

    | Field | Type | Description | Example |
    |-------|------|-------------|---------|
    | `embeddings` | Array of Floats | High-dimensional vector representation of paper abstract | `[0.234, -0.567, 0.891, ...]` |
    | `UMAP1` | Float | X-coordinate from UMAP dimensionality reduction of sentence embeddings | `-2.345` |
    | `UMAP2` | Float | Y-coordinate from UMAP dimensionality reduction of sentence embeddings | `1.678` |


        """, unsafe_allow_html=True)
   

with tab5:
    import json
   
    hierarchy = {
        'Poverty Context': ['Low Resource Level', 'Resource Volatility', '...'],
        'Study TYpe': {
            'Quantitative': ['Experimental', 'Longitudinal Study'],
            'Qualitative': ['Case Study', 'Focus Group'],
        }
    }

    active_filters = {
        'poverty_context': ['Low Resource Level'],
        'study_types': ['Experimental'],
    }
   
   
    st.markdown(f"""
                
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/streamlit/visuals/sankey.py)

    #### Sankey
               
    Sankey diagrams effectively visualize the flow and connections between categories, emphasizing how elements move or relate across hierarchical levels.

    In our analysis, papers are coded within a four-level hierarchical structure:
               
    - **Poverty Context**: The particular dimension or aspect of poverty examined in the study.
    - **Study Type**: The research design or methodological framework employed.
    - **Psychological Mechanism**: The cognitive or emotional process related to poverty that the study investigates.
    - **Behavioral Outcome**: The specific behavior or action influenced by poverty that the paper analyzes.
               
    The Sankey diagram maps the relationships between these categories, illustrating how papers classified under a specific poverty context are distributed across different study types, psychological mechanisms, and behavioral outcomes. 
    The width of the flows represents the number of papers linking one category to another, highlighting the dominant pathways and interactions within the research.
               
    ##### Nested Filtering 
               
    To enhance the utility of our Sankey diagram for research exploration, we have implemented dynamic filtering functionality. This feature enables users to focus on specific relationships and flows most relevant to their research interests.
    
    You can apply filters across any of the four categories, allowing targeted examination of particular dimensions within poverty research and their connections to other categories.
    
    For example, if your interest lies in studies addressing "Mental Health" as a psychological mechanism, you can filter the diagram to display only those papers. The visualization will then reveal how these papers relate to other categories such as study type, poverty context, and behavioral outcomes.
               

    ##### Implementation Details
            

    - **Installation Requirements:**
        ```bash     
        pip install plotly pandas
        ```

    - **Parameters:**
        - `filters_json` *(dict, optional)*: A dictionary defining the hierarchy of categories used to group data in the Sankey diagram. Each top-level key represents a dimension (e.g., `study_types`, `mechanisms`), and each value is a nested dictionary specifying subcategories.

    - **Usage:**

        ```python
        from visuals.sankey import Sankey

        # Load predefined category hierarchy
        filters = {json.dumps(hierarchy)}

        # Initialize the Sankey diagram with category definitions
        sankey_diagram = Sankey(filters_json=filters)

        # Define active filters as selected by the user (e.g., via Streamlit multiselect inputs)
        active_filters = {json.dumps(active_filters)}

        # Draw the Sankey diagram using the full dataset and the active filters
        fig = sankey_diagram.draw(df, active_filters=active_filters)

        # Display the figure in the Streamlit app
        fig.show()
        ```

""", unsafe_allow_html=True)
    

with tab6:
   
   st.markdown(f"""

    ##### Research Density Animation

    This visual is a heat map showing the density of research papers in a two-dimensional space derived from embedding representations.
    Papers are first encoded via embeddings capturing their semantic content, then projected into two dimensions using for spatial visualization. 
    The tool is designed to:

    1. **Provide a bird’s-eye view** of the research field, where users can hover over points to see individual paper titles.
    2. **Show the temporal evolution** of the research space, illustrating how clusters and research activity grow and shift over time.

    This tool is intended to complement prior filtering steps (such as those performed using Sankey diagrams), allowing researchers to further explore the filtered subset in spatial and temporal context.

    ###### Key Features

    - **Embedding-based spatialization** using UMAP projections.
    - **Hover-enabled data points** to reveal detailed paper information.
    - **Animated cumulative frames** representing research development by date.
    - **Density contours** highlighting research intensity in core areas.
    - **Cluster labels** marking topic centroids within the visible area.

    ###### Implementation Details
    
    - **Installation Requirements:**
               
        ```bash
        pip install plotly pandas numpy
        ```

    - **Parameters:**
        - `plot_df` (pd.DataFrame): Contains document embeddings projected with UMAP (UMAP1, UMAP2), dates (date), and titles (title).
        - `topic_clusters` (pd.DataFrame): Contains topic cluster centroid coordinates (umap_1_mean, umap_2_mean) and labels (label).

    - **Usage:**
        ```python
        from visuals import heatMap

        # Generate the animated visualization after filtering research papers via Sankey or other means
        fig = heatMap.draw(plot_df, topic_clusters)

        # Render the visualization, for example in a Streamlit app
        fig.show()

        ```

    """, unsafe_allow_html=True)
    
