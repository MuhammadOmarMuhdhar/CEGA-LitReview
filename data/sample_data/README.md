# Dataset Documentation: Psychology of Poverty Literature Analysis

This dataset contains academic papers collected through the OpenAlex API focusing on the intersection of poverty with psychological and economic factors. The data is structured to support comprehensive analysis of research trends, citation networks, and institutional collaborations in this field.

## Dataset Structure

### Search Keywords
The papers are collected using two categories of search terms:

**Economics Keywords**
- Poverty and Aspirations
- Poverty and Time preference
- Poverty and Risk preference
- Poverty and self-efficacy
- Poverty and locus of control
- Poverty and Optimism
- Poverty and Beliefs
- Poverty and Mindset
- Poverty and Internalized stigma

**Psychology Keywords**
- Poverty and mental health
- Poverty and Depression
- Poverty and Anxiety
- Poverty and Stress
- Poverty and Happiness
- Poverty and self concept
- Poverty and self esteem
- Poverty and Cognitive Function
- Poverty and Cognition
- Poverty and Cognitive flexibility
- Poverty and Executive control
- Poverty and Memory
- Poverty and working memory
- Poverty and Fluid intelligence
- Poverty and Attention

## Data Schema

### Raw JSON Structure

```
{
"doi": "string",
"title": "string",
"link": "string",
"authors": ["string"],
"keyword": "string",
"publication": "string",
"country": "string",
"date": "string",
"field": "string",
"institution": "string",
"abstract": "string",
"cited_by_count": "integer",
"citing_works": ["string"],
"referenced_works": ["string"]
}

```


### Normalized Tables

The dataset is normalized into multiple tables using DOI as the primary key:

#### Authors Table

| doi           | authors       |
|---------------|---------------|
| `"string"`    | `["string"]`  |

#### Citations Table

| doi           | cited_by_count | referenced_works | citing_works   |
|---------------|----------------|------------------|----------------|
| `"string"`    | `"integer"`    | `["string"]`     | `["string"]`   |

#### Institutions Table

| doi           | institution   | country      |
|---------------|---------------|--------------|
| `"string"`    | `"string"`    | `"string"`   |

#### Publications Table

| doi           | title         | abstract     | publication   | field       | keyword      |
|---------------|---------------|--------------|---------------|-------------|--------------|
| `"string"`    | `"string"`    | `"string"`   | `"string"`    | `"string"`  | `"string"`   |

## Usage Notes

- The sampling is distributed evenly across keywords rather than being stratified.
- DOI serves as the primary key for joining across tables.
- Some fields (e.g., `country`, `institution`) may not be available for all entries.
- Citation networks can be constructed using the `citing_works` and `referenced_works` arrays.
- Publication dates are stored in `YYYY` format.

## Data Dictionary

| Field              | Description                                | Type        |
|--------------------|--------------------------------------------|-------------|
| `doi`             | Digital Object Identifier                  | String      |
| `title`           | Academic paper title                       | String      |
| `authors`         | List of author names                       | Array       |
| `keyword`         | Search term used                           | String      |
| `publication`     | Journal/venue name                         | String      |
| `country`         | Publication origin                         | String      |
| `date`            | Publication year                           | String      |
| `field`           | Academic discipline                        | String      |
| `institution`     | Research affiliation                       | String      |
| `abstract`        | Paper summary                              | String      |
| `cited_by_count`  | Citation count                             | Integer     |
| `citing_works`    | Papers citing this work                    | Array       |
| `referenced_works`| Works cited in paper                       | Array       |



