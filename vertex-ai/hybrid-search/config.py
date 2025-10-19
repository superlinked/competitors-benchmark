#!/usr/bin/env python3
"""
Hybrid Search Configuration

This module provides hybrid search-specific configuration settings
that extend the shared base configuration.
"""

import os
import sys

# Add project root to path to import shared config
# Go up from hybrid-search -> vertex-ai -> project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from shared_config import *

# ============================================================================
# HYBRID SEARCH-SPECIFIC CONFIGURATION
# ============================================================================

# ============================================================================
# QDRANT CONFIGURATION
# ============================================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6334
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hybrid_search"
TOP_K = 20

# ============================================================================
# HYBRID SEARCH-SPECIFIC FILE PATHS
# ============================================================================

HYBRID_SEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_SEARCH_OUTPUT_DIR = os.path.join(HYBRID_SEARCH_DIR, "output")

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

EMBEDDINGS_FILE = os.path.join(HYBRID_SEARCH_OUTPUT_DIR, "product_embs.json")
SPLADE_MODEL = "naver/splade-v3"
GEMINI_MODEL = "gemini-embedding-001"
DEVICE_SPLADE = "cpu"
N_WORKERS = 5
BATCH_SIZE = 100
VECTOR_SIZE = 3072  # Gemini embedding dimension

RESULTS_FILE = os.path.join(HYBRID_SEARCH_OUTPUT_DIR, "search_results.json")
RANKED_RESULTS_FILE = os.path.join(HYBRID_SEARCH_OUTPUT_DIR, "ranked_results.json")
NLQ_PARAMS_QUERIES = os.path.join(HYBRID_SEARCH_OUTPUT_DIR, "nlq_params_queries.json")
