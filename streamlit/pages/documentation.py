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


tab1, tab2, tab3, tab4= st.tabs(["Sourcing Papers", "Google Sheets Database", "Data Dictionary", "MISC 2" ])



with tab1:

   st.markdown(f"""
    #### Sourcing Papers

    We source our research papers through the **OpenAlex API**, a comprehensive and freely accessible academic database. This platform provides extensive metadata on over 200 million scholarly works, including research papers, authors, institutions, and journals across all academic disciplines.

    Our approach to literature collection leverages OpenAlex's search capabilities, allowing us to query papers by keywords, research fields, and publication timeframes. To maintain focus and relevance in poverty research, we've developed a carefully curated keyword taxonomy that guides our data collection process.

    **🔗 Explore our keyword categories:** [View on GitHub](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/trainingData/categories.json)

    ##### About OpenAlex
    - **Platform:** [https://openalex.org](https://openalex.org)
    - **Documentation:** [https://docs.openalex.org](https://docs.openalex.org)
    - **No authentication required** – completely open access
    - **Generous rate limits:** 10 requests/second, 100,000 requests/24 hours
    - **Rich data format:** Structured JSON responses with comprehensive metadata

    ##### Our Custom Scraper Solution

    To efficiently harvest literature at scale, we've built a scraper built on OpenAlex that automates the entire data collection process. This tool intelligently manages API interactions while ensuring data quality and completeness.

    **Key capabilities include:**
    - **Smart rate limiting** using token bucket algorithms to respect API constraints
    - **Robust error handling** with automatic retries for network issues and rate limit responses
    - **Multithreaded processing** for faster data collection
    - **Intelligent abstract reconstruction** when papers provide segmented text
    - **Comprehensive metadata extraction** including titles, DOIs, abstracts, authors, and publication dates
    - **Flexible filtering** by date ranges and keyword combinations

    **Quick start example:**
    ```python
    from openAlex import Scraper

    # Initialize the scraper
    extractor = Scraper(requests_per_second=10, email="your_email@example.com")

    # Collect papers matching your criteria
    papers = extractor.run(
        broad_field="Psychology",
        keyword="cognitive",
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=100
    )
    ```
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/openAlex.py)

    """, unsafe_allow_html=True)
   
with tab2:
   
   st.markdown(f"""
    #### Google Sheets Database

    We leverage **Google Sheets as our primary database** for storing and managing research papers collected from the OpenAlex API. This cloud-based approach provides real-time accessibility, collaborative editing capabilities, and seamless integration with our automated data collection pipeline.

    Our system transforms Google Sheets into a powerful research database by automatically populating spreadsheets with paper metadata, abstracts, and bibliographic information. This eliminates manual data entry while maintaining the familiar spreadsheet interface that researchers prefer for data exploration and analysis.

    ##### Why Google Sheets as a Database?

    **Accessibility & Collaboration**
    - **Real-time updates** from our automated scrapers
    - **Multi-user access** for research teams
    - **Cloud-based storage** with automatic backups
    - **Familiar interface** that requires no technical expertise

    **Integration Benefits**
    - **Direct API connectivity** for automated data population
    - **Flexible data structures** that adapt to varying paper formats
    - **Built-in sharing and permissions** for research collaboration
    - **Export capabilities** to multiple formats (CSV, Excel, PDF)
               
    ##### Authentication & Setup

    To use our Google Sheets database system, you'll need to set up [**Google Cloud Service Account credentials**](https://cloud.google.com/iam/docs/service-account-creds). This enables secure, programmatic access to your Google Sheets without requiring manual authentication.


    ##### Our Custom Google Sheets API Wrapper

    To efficiently manage large-scale literature data, we've developed an API wrapper that handles the Google Sheets connection.

    **Features:**
    - **Intelligent batching** to handle large datasets efficiently while respecting Google's API limits
    - **Automatic data type conversion** ensuring clean storage of complex paper metadata
    - **Smart list handling** that converts author arrays and keyword lists to readable strings
    - **Robust error handling** with comprehensive logging for debugging and monitoring
    - **Rate limit management** to prevent API quota exhaustion during bulk operations
    - **Flexible sheet management** including creation, replacement, and append operations

    **Core Operations:**

    **📝 Writing Papers to Sheets**
    ```python
    from googleSheets import API

    # Initialize with your service account credentials
    sheets_api = API(credentials_json)

    # Replace entire sheet with new paper collection
    results = sheets_api.replace(
        df=papers_dataframe,
        spreadsheet_id="your_sheet_id",
        sheet_name="Literature_Review",
        include_headers=True
    )

    # Append new papers to existing collection
    results = sheets_api.append(
        df=new_papers,
        spreadsheet_id="your_sheet_id",
        sheet_name="Recent_Papers"
    )
    ```

    **📊 Reading Data for Analysis**
    ```python
    # Load papers for analysis
    papers_df = sheets_api.read(
        spreadsheet_id="your_sheet_id",
        sheet_range="Literature_Review",
        header_row=0
    )
    ```
               
    [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MuhammadOmarMuhdhar/CEGA-LitReview/blob/main/data/googleSheets.py)
   
    """, unsafe_allow_html=True)
   
with tab3:
   
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
   
   