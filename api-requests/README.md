# Research Paper API Collection Scripts

## Overview
This folder contains scripts for automated collection of academic papers through various public APIs. The primary purpose is to gather scholarly articles that can be processed and tagged for research purposes, particularly focusing on psychology and poverty literature.

## Supported APIs
Currently implemented:
- AgEcon
  - No authentication required
  - No Rate-Limits
  - Documentation: https://doapr.coar-repositories.org/repositories/agecon-search/

*More APIs to be added as the project expands*

## Data Structure
All papers are stored in JSON format with the following schema:

```
{
"doi": "string" 
"title": "string", // Full title of the paper
'link': "string", // Link where the paper is found
'authors': "list", // A list of all the authors of a paper
'publication': "string", // Where the paper was published
'country': "string", // Country where paper is sourced (if available)
"date": "string", // Publication date (YYYY)
"field": "string", // Publication broad field(psychology, economics, political science e.t.c)
"institution": "string", // Publishing institution (if available)
"abstract": "string" // Paper abstract
}

```

## Usage
1. Each API has its own dedicated script in the repository
2. Scripts can be run independently to collect data from specific sources
3. Output files are stored in JSON format for further processing

## File Organization

```
├── api_scripts/
│ ├── AgEcon.py
│ ├── [future_api_scripts].py
| ├── README.md
| └── Data
|   ├──

```

## Contributing
When adding new API integrations:
1. Create a new script in the `api_requests` directory
2. Follow the established JSON schema
3. Document any API-specific requirements or limitations
4. Update this README with new API details

