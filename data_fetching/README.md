# Research Paper API Collection Scripts

## Overview
This folder contains scripts for automated collection of academic papers through public APIs. The primary purpose is to gather scholarly articles that can be processed and tagged for research purposes, particularly focusing on psychology and economics of poverty literature.

## Supported APIs
Currently implemented:
- OpenAlex
  - No authentication required
  - Rate Limits:
    - 10 requests per second (burst limit)
    - 100,000 requests per 24-hour period
  - Documentation: https://docs.openalex.org


## Data Structure
All papers are stored in JSON format with the following schema:

```

{
    "doi": "string",  // Digital Object Identifier - unique paper identifier
    "title": "string",  // Complete academic paper title
    "link": "string",  // URL to access the paper
    "authors": ["string"],  // Array of author names
    "publication": "string",  // Journal or publication venue name
    "country": "string",  // Country of publication origin (when available)
    "date": "string",  // Publication year in YYYY format
    "field": "string",  // Primary academic discipline (e.g., psychology, economics)
    "institution": "string",  // Affiliated research institution (when available)
    "abstract": "string",  // Paper summary/abstract
    "cited_by_count": "integer",  // Total citation count
    "citing_works": ["string"],  // DOIs of papers citing this work
    "referenced_works": ["string"]  // DOIs of works cited in this paper
}

```

## Usage
1. Each API should have its own dedicated script in the repository
2. Scripts can be run independently to collect data from specific sources
3. Output files are stored in JSON format for further processing

## File Organization

```
├── api_scripts/
│ ├── openalex.py
│ ├── [future_api_scripts].py
| ├── README.md
├── Data
```

## Contributing
When adding new API integrations:
1. Create a new script in the `api_requests` directory
2. Follow the established JSON schema
3. Document any API-specific requirements or limitations
4. Update this README with new API details

