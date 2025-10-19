# Hybrid Search Pipeline

This directory contains a hybrid search implementation that combines sparse (SPLADE) and dense (Gemini) embeddings to improve search relevance. The pipeline uses Qdrant as the vector database and includes query feature extraction, embedding generation, search execution, and result reranking.

## Overview

The hybrid search approach leverages:

- **SPLADE (Sparse Lexical and Expansion)**: Generates sparse embeddings that capture lexical and expansion information
- **Gemini Embeddings**: Produces dense embeddings that capture semantic meaning
- **Qdrant Vector Database**: Stores and searches both embedding types efficiently
- **Vertex AI Ranking**: Reranks search results using Google's Discovery Engine

## Pipeline Components

### 1. Query Feature Extraction (`query_feature_extraction.py`)

Extracts structured parameters from natural language queries using Gemini 2.0 Flash

### 2. Embedding Generation (`create_embs.py`)

Creates both sparse and dense embeddings for all products

### 3. Qdrant Search (`run_qdrant.py`)

Sets up and runs hybrid search using Qdrant

### 4. Result Reranking (`rerank_products.py`)

Reranks search results using Vertex AI Discovery Engine

## Setup

### Prerequisites

- Docker (for Qdrant)
- Google Cloud credentials
- Google API key for Gemini

## Configuration

Key configuration parameters in `config.py`:

- **Models**: SPLADE model, Gemini embedding model, ranking model
- **Qdrant**: Host, port, collection name, top-k results
- **Processing**: Batch size, number of workers, device (MPS for Apple Silicon)
- **Data Sources**: Google Cloud Storage bucket paths for products and queries

## Usage

### 1. Extract Query Features

```bash
# from the root
python vertex-ai/hybrid-search/query_feature_extraction.py
```

### 2. Generate Product Embeddings

```bash
#from the root
python vertex-ai/hybrid-search/create_embs.py
```

### 3. Run Hybrid Search

```bash
#from the root
python vertex-ai/hybrid-search/ranking.py
```

### 4. Rerank Results

```bash
python vertex-ai/hybrid-search/rerank_products.py
```

## Output Files

- `vertex-ai/hybrid-search/output/product_embs.json`: Product embeddings (SPLADE + Gemini)
- `vertex-ai/hybrid-searchoutput/results.json`: Initial search results from Qdrant
- `vertex-ai/hybrid-search/output/results_reranked.json`: Final reranked results
- `vertex-ai/hybrid-search/output/nlq_params_queries.json`: Extracted query parameters
