# Superlinked Search Pipeline

This module contains a Superlinked implementation that leverages a mixture of embedders for advanced semantic search capabilities. The pipeline supports query feature extraction, embedding generation, and configurable search approaches with multiple embedding models.

## Overview

Superlinked is a vector database platform that provides:

- **Multi-Embedder Support**: Combines multiple embedding models for enhanced search relevance
- **Flexible Schema Design**: Supports multiple spaces
- **Advanced Filtering**: Provides sophisticated filtering capabilities (soft and hard)
- **Query Parameter Integration**: Uses extracted query parameters for enhanced search precision

## Available Apps

The module supports two main applications:

### 1. SOTA App (`sota_app`)

- State-of-the-art implementation with advanced features
- Supports NLQ (Natural Language Query) parameter extraction
- Uses ground truth parameters for baseline comparison
- Full text search capabilities

### 2. SOTA App Baseline (`sota_app_baseline`)

- Baseline implementation for comparison (stringify-and-embedd)
- Simplified configuration
- Focus on full text search

## Usage

Make sure you have updated config files (local ones depending on your app) and a [global one](../shared_config.py) with you GCP and Redis credentials and paths

Run the module using Python's module execution:

```bash
# Run the SOTA app
python -m superlinked_app.app sota_app

# Run the baseline app (stringify-and-embedd)
python -m superlinked_app.app sota_app_baseline
```

**Note:** if you encounter problems with Superlinked and Redis, then run this line

```bash
pip install 'superlinked[redis]'
```
## Search Approaches

The implementation supports multiple search approaches:

### 1. NLQ Parameters

- Uses parameters extracted by LLM from natural language queries
- Applies dynamic filtering based on extracted attributes
- Combines multi-embedder search with structured filtering

### 2. Ground Truth Parameters

- Uses predefined ground truth parameters
- Applies exact filtering based on known query parameters
- Provides baseline performance comparison

### 3. Full Text Search

- Uses only the query text for search
- Leverages Stringify-and-embedd approach
- No additional filtering applied

## Module Structure

### Core Components

- **`app.py`**: Main application entry point
- **`registry.py`**: Module registry for different app configurations
- **`util/`**: Utility functions and enums

### App-Specific Components

Each app (`sota_app`, `sota_app_baseline`) contains:

- **`config.py`**: Configuration settings and parameters
- **`data_prep.py`**: Data preprocessing functions
- **`index.py`**: Superlinked index creation and configuration
- **`query.py`**: Query configuration and nlq descriptions
- **`nlq.py`**: Natural Language Query configuration

## Setup

### Prerequisites

- GCP bucket
- OpenAI API key for NLQ
- Redis instance for vector storage

### Configuration

Each app has its own configuration in `apps/{app_name}/config.py`:

- **Superlinked**:  Embedder settings, Query mode (USE_FULL_QUERY_TEXT, USE_GROUND_TRUTH_QUERY_INPUTS, USE_NLQ), Reingest option, Redo nlq option
- **Data Sources**: Google Cloud Storage bucket paths for products and queries
- **Output Files**: Paths for results and query parameters
- **Redis**: Vector database connection settings