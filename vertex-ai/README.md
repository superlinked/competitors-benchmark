# Vertex AI Search Pipeline

This directory contains a Vertex AI Search implementation that leverages Google Cloud's fully managed search solution with sophisticated ranking and personalization capabilities. The pipeline supports multiple search approaches with configurable parameter sources and advanced boost specifications.

## Overview

Vertex AI Search (VAI) is a fully managed search solution provided by Google Cloud that integrates with the Ranking API to enable sophisticated ranking and personalization. This implementation also provides NLQ paramter extraction for the advanced filtering by external script (still using Gemini)

## Search Approaches

The implementation supports three main parameter-based approaches:

### 1. NLQ Parameters (`--params nlq`)

- Uses parameters extracted by LLM from natural language queries
- Applies dynamic filtering based on extracted attributes (color, material, style, price, rating)
- Uses boost specifications for personalized ranking
- Supports negation filters for excluded attributes

### 2. Predefined Parameters (`--params gt`)

- Uses manually defined query parameters from Google Cloud Storage
- Applies the same filtering and boosting logic as NLQ parameters
- Useful for testing with known parameter sets

### 3. No Parameters (`--params none`)

- Uses only the query text for search
- Leverages Vertex AI's built-in semantic understanding
- No additional filtering or boosting applied

## Setup

### Prerequisites

- Google Cloud Platform account
- Vertex AI Search engine configured
- Google Cloud credentials

### GCP Configuration

To get started with Vertex AI Search, follow these steps:

1. Review Google's setup guide: [Before You Begin](https://cloud.google.com/generative-ai-app-builder/docs/before-you-begin)
2. In the Google Cloud Console, create a Vertex AI Search app (select "Search app" when prompted).
3. Prepare your dataset (products, queries, etc.) and upload it to your designated Google Cloud Storage bucket.
4. In the Vertex AI Search app configuration, connect your data bucket and follow the ingestion workflow to index your data.
5. Once indexing is complete, the search engine is ready for use with this pipeline and scripts.

## Configuration Settings

Key configuration parameters in `config.py`:

- **Project Settings**: Google Cloud project ID, location, engine ID
- **Data Sources**: Google Cloud Storage bucket paths for products and queries
- **Output Files**: Paths for results and query parameters

## Usage

### Query Parameter Extraction

Before running the ranking script, you need to extract query parameters:

```bash
# Generate parameters from query text using LLM
python vertex-ai/query_params_extraction.py
```

### Basic Usage

```bash
python vertex-ai/ranking.py --params nlq
```

### Command Line Options

- `--params`: Parameter source (`nlq`/`gt`/`none`)
  - `nlq`: Use parameters extracted by LLM from natural language queries (applies filtering and boosting)
  - `gt`: Use manually defined query parameters from GCS (applies filtering and boosting)
  - `none`: Use only query text for search (no filtering or boosting)

### Example Commands

```bash
# Search with NLQ parameters (filtering and boosting enabled)
python vertex-ai/ranking.py --params nlq

# Search with ground truth parameters (filtering and boosting enabled)
python vertex-ai/ranking.py --params gt

# Search without any parameters (query text only, no filtering)
python vertex-ai/ranking.py --params none
```

## Technical Details

### Filtering

The implementation supports dynamic filtering based on query parameters:

- **Attribute Filters**: Color, material, style matching with `ANY()` operator
- **Negation Filters**: Exclude specific colors, materials, or styles using `NOT` operator
- **Numeric Constraints**: Price maximum, minimum rating, minimum review count

### Boost Specifications

Personalized ranking uses statistical percentiles from product data:

- **Price Boosting**: Based on price weight preferences (positive for higher prices, negative for lower)
- **Rating Boosting**: Based on rating weight preferences (positive for higher ratings, negative for lower)
- **Review Count Boosting**: Based on review count weight preferences (positive for more reviews, negative for fewer)

Boost values are clamped between -0.9 and 0.9 and applied using percentile-based conditions (p25, p50, p75, p90).

## Output Files

- `vertex-ai/output/gt_params_results.json`: Search results using predefined ground truth parameters
- `vertex-ai/output/full_text_results.json`: Search results without any parameters
- `vertex-ai/output/nlq_results.json`: Search results using NLQ parameters
- `vertex-ai/output/nlq_params_queries.json`: Extracted query parameters (when using NLQ)
