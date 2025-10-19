#!/usr/bin/env python3
"""
Azure Search Configuration

This module provides Azure-specific configuration settings
that extend the shared base configuration.
"""

import os
import sys

# Add parent directory to path to import shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_config import *

# ============================================================================
# AZURE-SPECIFIC CONFIGURATION
# ============================================================================

AZURE_SEARCH_ENDPOINT = "https://wands.search.windows.net"
AZURE_SEARCH_INDEX_NAME = "products-index"
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEMANTIC_CONFIGURATION_NAME = "default"

# ============================================================================
# AZURE OPENAI CONFIGURATION
# ============================================================================

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT = "o4-mini"

# ============================================================================
# AZURE-SPECIFIC FILE PATHS
# ============================================================================

# Get the directory of this config file (azure directory)
AZURE_DIR = os.path.dirname(os.path.abspath(__file__))
AZURE_OUTPUT_DIR = os.path.join(AZURE_DIR, "output")

GT_PARAMS_OUTPUT_RESULTS = os.path.join(AZURE_OUTPUT_DIR, "gt_params_results.json")
NLQ_OUTPUT_RESULTS = os.path.join(AZURE_OUTPUT_DIR, "nlq_results.json")
FULL_TEXT_OUTPUT_RESULTS = os.path.join(AZURE_OUTPUT_DIR, "full_text_results.json")
NLQ_PARAMS_QUERIES = os.path.join(AZURE_OUTPUT_DIR, "nlq_params_queries.json")
