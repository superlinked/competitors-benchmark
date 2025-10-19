# Wands: Product Generation Search Pipeline and Benchmarking

A comprehensive product search pipeline that processes product data, generates queries based on the data, and performs
candidate selection using dense and sparse embeddings and filters topped with RankingAPI to establish ground
truth rankings. Finally, it evaluates retrieval approaches against the established ground truth rankings.

## Quick Start

### Prerequisites

Create a new venv

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Config

Check the [config.py](dataset-generation/config.py) file and change the settings based on your setup.

### Environment Setup

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
AZURE_SEARCH_API_KEY=...
AZURE_SEARCH_OPEN_AI_KEY=...
AZURE_OPENAI_ENDPOINT=...
```

### Run Pipeline

Execute scripts in order:

```bash
# 1. Process data and create embeddings
python dataset-generation/01_data_processing.py

# 2. Generate queries
python dataset-generation/02_query_generation.py

# 3. Enrich queries with features
python dataset-generation/03_enrich_queries.py

# 4. Rerank results using Google RankingAPI
python dataset-generation/04_reranking.py
```

## Overview

This pipeline has the following capabilities:

- **Data Processing**: Cleans and normalizes product features with vector database integration
- **Query Generation**: Creates diverse queries with real-time validation using vector search
- **Semantic Search**: Uses multiple embedding models (SPLADE, Qwen, Sentence Transformers) and Qdrant vector database
- **AI Ranking**: Employs Google RankingAPI for enterprise-grade reranking
- **Query Enrichment**: Extracts structured parameters from natural language using Gemini API
- **Retrieval Evaluation**: Evaluates different retrieval systems based on generated queries and candidates

## Data Generation Pipeline Components

### 1. [Data Processing](dataset-generation/01_data_processing.py)

- Loads raw product data from CSV
- Extracts and normalizes features (product_class, material, color, style, country) with LLM-predefined mapping
- Predicts missing product classes
- Combines with synthetic products
- Creates embeddings using Splade and Qwen3 for Qdrant vector database
- Outputs: `products.jsonl`, `products.csv`, `product_embs.json`

#### [Price Mapping System](dataset-generation/price_mapping.py)

- **Product Class Pricing**: Maps 400+ product categories to realistic price ranges
- **Feature-Based Pricing**: Adjusts prices based on materials, styles, and country of origin
  - Premium materials (mahogany, marble, brass): 1.4x multiplier
  - Mid-range materials (wood, metal, cotton): 1.1x multiplier
  - Budget materials (particle board, plastic): 0.85x multiplier
- **Rating-Based Pricing**: Incorporates product ratings into pricing
  - Excellent ratings (4.5-5.0): 1.2x multiplier
  - Poor ratings (1.0-2.5): 0.9x multiplier
- **Smart Price Generation**: Creates realistic prices considering all factors

### 2. [Query Generation](dataset-generation/02_query_generation.py)

- Creates diverse query types:
  - **Simple**: Basic attribute-based queries
  - **Negation**: Queries with excluded attributes
  - **Number**: Queries with rating/review constraints
  - **Complex**: Queries that require specific numeric ranking
- Uses Qdrant for hydrid search. DBSF is used
- Outputs: `queries_temp.json`, `search_results_temp.json`, `query_generation.jsonl`

### 3. [Query Enrichment](dataset-generation/03_enrich_queries.py)

- Enriches queries with extracted features using Gemini API based on the items retrieved on query generation step. Params remain the same. Only query_text, query_description is changed
- Runs search on enriched queries
- Includes retry logic for API failures
- Outputs: `queries.json`, `search_results.json`

### 4. [AI Reranking](dataset-generation/04_reranking.py)

- Uses RankerAPI Engine for reranking
- Outputs: `ranked_results.json`

### Vector Database Integration

- **Qdrant Integration**: Uses Qdrant for efficient vector storage and search
- **Multiple Embedding Models**: Uses SPLADE, QWEN for Hybrid-Search
- **Comples Reranking**: Uses Distribution-Based Score Fusion to combine the results

### Data Structure

#### Input Data

- `data/input/price_mapping.jsonl`: Product class to price range mappings (min/max prices for 400+ product categories)
- `data/input/feature_weights.jsonl`: Feature-based pricing weights for materials, styles, countries, and ratings
- `data/input/product.csv`: Raw product data
- `data/input/synthetic_products.json`: Additional synthetic products
- `data/input/synthetic_queries.json`: Pre-defined synthetic queries
- `data/input/unique_categories.json`: Unique category mappings

#### Output Data

- `data/output/products.jsonl`: Processed product data in jsonl
- `data/output/products.csv`: Processed product data in csv
- `data/output/product_embs.json`: Product embeddings for vector search
- `data/output/queries.json`: Generated queries
- `data/output/search_results.json`: Search results
- `data/output/ranked_results.json`: AI-ranked results
- `data/output/nlp_params_queries.json`: Queries with extracted parameters

### Query Types

- **Simple**: "decorative bowls in rustic style"
- **Negation**: "waiting room chairs in modern style, not leather material, great for reception areas"
- **Number**: "silver pendant lights, under $1220, need clear glass paneling"
- **Complex**: "unique affordable gray dining linens"
- **Synthetic**: Generated based on synthetic (LLM generated) products to test correctness

All queries use the following scheme as a skeleton: product_class + extracted feature from enrichment step (usage, adjective, feature).

- Simple queries have additional material, style or color
- Negation queries are simple queries with negated style, material or color
- Number queries are simple queries with numeric requirement
- Complex queries are simple queries with specific numeric ranking requirements e.g. "cheap", "popular"
- Synthetic queries may be seen in `data/input/synthetic_queries.json`, corresponding products in `data/input/synthetic_products.json`. Each synthetic query ideally targets 10 products

### Configuration

The pipeline uses a centralized configuration `shared_config.py` and local one in `dataset-generation/config.py`:

#### Key Configuration Areas

- **Paths**: Input/output directory configuration
- **Models**: Embedding and ranking model settings
- **APIs**: Google Cloud and OpenAI API configuration
- **Qdrant**: Vector database connection settings
- **Processing**: Batch sizes, worker counts, and retry settings
- **Query Generation**: Query type counts and validation thresholds

### Evaluation

Run the evaluation script to compare different retrieval systems against the ground-truth rankings produced in the pipeline.

#### Supported systems

- #### [Azure](/azure)
  
  🔗 https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search

  Microsoft black box search with built-in Semantic Ranking
  Support nlq parameters extraction and plain text search

- #### [Superlinked](superlinked_app)
  
  🔗 https://superlinked.com/

  Mixture of Embedders

- #### [VertexAI](vertex-ai)

  🔗 https://cloud.google.com/enterprise-search?hl=en

  Google black box search with RerankingAPI
  Support nlq parameters extraction and plain text search

- #### [VertexAI Hybrid Search](vertex-ai/hybrid-search/)

  Combines SPLADE for sparse encoding and Gemini for dense encoding, uses Qdrant as the VDB, and applies RerankingAPI for final ranking.
  Supports nlq parameters extraction

Configure dataset version and paths in `evaluation/config.py`. Update `data_id` to switch datasets; local result paths can also be adjusted there.

#### How to run

The `--k` flag controls the cutoff for Precision@k, Recall@k, MRR@k, and NDCG@k.
The `--system` flag defines the system you want to evaluate.

List of systems:

- `azure-nlq`
- `azure-gt-params`
- `azure-full-text`
- `hybrid-search`
- `vertex-ai-nlq`
- `vertex-ai-gt-params`
- `vertex-ai-full-text`
- `sl-nlq`
- `sl-gt-params`
- `sl-full-text`


Example:

```bash
python evaluation/app.py --system sl-nlq --k 10
```

#### Output

The script prints a JSON-like dictionary with per-query-type metrics and an aggregated block. Example:

```
{
  "overall": {"precision@10": 0.4498, "recall@10": 0.3945, "mrr@10": 0.8087, "ndcg@10": 0.4984}, 
  "simple": {"precision@10": 0.592, "recall@10": 0.3093, "mrr@10": 0.8192, "ndcg@10": 0.5457},
  ... 
}

```

#### Notes

- For GCS-based runs (`sl-*` systems), ensure `google-cloud-storage` is installed and your environment has credentials to access the bucket defined in `evaluation/config.py`.
- If you produce rankings from a new system, write them as the JSON mapping above and point a new local path in `evaluation/config.py`, then run with a new `--system` choice after adding it to `evaluation/app.py`.

### Metrics

The system evaluates search quality using:

- **Precision@k** - Relevance of top results
- **Recall@k** - Coverage of relevant items  
- **MRR@k** - Mean reciprocal rank of first relevant result
- **NDCG@k** - Position-weighted relevance scoring

Results are segmented by query type with both category-specific and aggregated reporting.
