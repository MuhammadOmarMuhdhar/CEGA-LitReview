import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
from data_fetching import openalex
import pandas as pd

# Dictionary of research categories and their associated keywords
categories = {
    "Economics-related": [
        "Poverty and Aspirations",
        "Poverty and Time preference",
        "Poverty and Risk preference",
        "Poverty and self-efficacy",
        "Poverty and locus of control",
        "Poverty and Optimism",
        "Poverty and Beliefs",
        "Poverty and Mindset",
        "Poverty and Internalized stigma"
    ],
    "Psychology-related": [
        "Poverty and mental health",
        "Poverty and Depression",
        "Poverty and Anxiety",
        "Poverty and Stress",
        "Poverty and Happiness",
        "Poverty and self concept",
        "Poverty and self esteem",
        "Poverty and Cognitive Function",
        "Poverty and Cognition",
        "Poverty and Cognitive flexibility",
        "Poverty and Executive control",
        "Poverty and Memory",
        "Poverty and working memory",
        "Poverty and Fluid intelligence",
        "Poverty and Attention"
    ]
}

# List to store all extracted paper data
data = []

# Iterate through categories and subcategories to fetch papers
for category, subcategories in categories.items():
    for subcategory in subcategories:
        # Extract papers for each subcategory (limited to 40 papers)
        papers = openalex.extract_papers(broad_field=category, keyword=subcategory, limit=40)
        data.extend(papers)

# Convert the list of papers to a DataFrame
df = pd.DataFrame(data)

# Remove duplicate papers based on DOI
df.drop_duplicates(subset="doi", inplace=True)

# Split the data into separate DataFrames for different aspects
publications = df[['doi', 'title', 'link', 'abstract', 'date', 'publication', 'field', 'keyword', 'cited_by_count']]
authors = df[['doi', 'authors']]
institutions = df[['doi', 'institution', 'country']]
citations = df[['doi', 'cited_by_count', 'referenced_works']]

# Save the separate DataFrames as CSV files
publications.to_csv("data/sample_data/publications.csv", index=False)
authors.to_csv("data/sample_data/authors.csv", index=False)
institutions.to_csv("data/sample_data/institutions.csv", index=False)
citations.to_csv("data/sample_data/citations.csv", index=False)

