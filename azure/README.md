# Azure Cognitive Search Pipeline

This directory contains an Azure Cognitive Search (or simply Azure AI Search) implementation that leverages Microsoft's managed search service with semantic search capabilities and advanced filtering. The pipeline supports query feature extraction, index management, and configurable search approaches.

## Overview

Azure AI Search is a managed search-as-a-service that lets you index and query your data using full-text, vector, and hybrid search.
All index creation and data ingestion is handled by SDK.

This pipeline uses Azure's managed search as a "black box" solution, performing semantic search and applying reranking for improved result relevance.
Also this implementation provides NLQ parameter extraction for the advanced filtering.

## Search Approaches

The benchmark implements three main approaches:

### 1. Basic Search (`--params none`)

- Uses only the query text for search
- Leverages Azure's built-in semantic understanding
- No additional filtering applied

### 2. NLQ Parameters (`--params nlq`)

- Uses parameters extracted by Azure OpenAI from natural language queries
- Applies dynamic filtering based on extracted attributes
- Combines semantic search with structured filtering

### 3. Ground Truth Parameters (`--params gt`)

- Uses predefined ground truth parameters
- Applies exact filtering based on known query parameters
- Provides baseline performance comparison

## Pipeline Components

### 1. Index Management (`create_index.py`)

Creates the Azure Cognitive Search index with semantic configuration

### 2. Data Upload (`upload_data.py`)

Uploads product data to the Azure search index

### 3. Query Feature Extraction (`query_params_extraction.py`)

Extracts structured parameters from natural language queries using Azure OpenAI

### 4. Search Execution (`ranking.py`)

Performs semantic search with optional filtering and parameter-based constraints

### 5. Index Utilities (`check_index.py`, `delete_index.py`)

Utility scripts for index management and verification

- **Check Index**: Verify index configuration and semantic setup
- **Delete Index**: Remove the search index (useful for cleanup)

## Setup

### Prerequisites

- [Azure Cognitive Search service](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)
- [Azure OpenAI service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- Google Cloud credentials for data access

## Configuration

Key configuration parameters in `config.py`:

- **Azure Search**: Endpoint, index name, API key, semantic configuration
- **Azure OpenAI**: Endpoint, API key, deployment name, API version
- **Data Sources**: Google Cloud Storage bucket paths for products and queries
- **Output Files**: Paths for results and query parameters

## Usage

### 1. Create Search Index

```bash
python azure/create_index.py
```

Creates the Azure Cognitive Search index with semantic configuration.

### 2. Upload Product Data

```bash
python azure/upload_data.py
```

Uploads product data from the dataset to the Azure search index.

### 3. Extract Query Features

```bash
python azure/query_params_extraction.py
```

Extracts structured parameters from queries using Azure OpenAI.

### 4. Run Search

```bash
python azure/ranking.py --params nlq
```

Performs semantic search with the specified parameter source.

### Command Line Options

- `--params`: Parameter source (`nlq`/`gt`/`none`)
  - `nlq`: Use parameters extracted by Azure OpenAI from natural language queries
  - `gt`: Use predefined ground truth parameters
  - `none`: Use only query text for search

### Example Commands

```bash
# Search with NLQ parameters
python azure/ranking.py --params nlq

# Search with ground truth parameters
python azure/ranking.py --params gt

# Search without any parameters (query text only)
python azure/ranking.py --params none
```

## Technical Details

### Index Schema

Created by `azure/create_index.py`. Fields and capabilities:

- `product_id`: Edm.String (key, searchable, filterable, sortable, facetable)
- `product_name`: Edm.String (searchable, filterable, sortable, facetable)
- `product_description`: Edm.String (searchable, filterable)
- `product_class`: Edm.String (searchable, filterable, sortable, facetable)
- `material`: Collection(Edm.String) (searchable, filterable, facetable)
- `style`: Collection(Edm.String) (searchable, filterable, facetable)
- `color`: Collection(Edm.String) (searchable, filterable, facetable)
- `rating_count`: Edm.Int32 (filterable, sortable, facetable)
- `average_rating`: Edm.Double (filterable, sortable, facetable)
- `countryoforigin`: Collection(Edm.String) (searchable, filterable, facetable)
- `price`: Edm.Double (filterable, sortable, facetable)

### Semantic Configuration

- Name: `default`
- Title field: `product_name`
- Content fields: `product_description`, `product_class`
- Keyword fields: `product_class`, `material`, `style`, `color`, `countryoforigin`

## Output Files

- `azure/output/gt_params_results.json`: Search results using predefined ground truth parameters
- `azure/output/full_text_results.json`: Search results without any parameters
- `azure/output/nlq_results.json`: Search results using NLQ parameters
- `azure/output/nlq_params_queries.json`: Extracted query parameters (when using NLQ)
