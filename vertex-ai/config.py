#!/usr/bin/env python3
"""
Vertex AI Configuration

This module provides Vertex AI-specific configuration settings
that extend the shared base configuration.
"""

import os
import sys

# Add parent directory to path to import shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_config import *

# ============================================================================
# VERTEX AI-SPECIFIC CONFIGURATION
# ============================================================================

ENGINE_ID = "google-wands-search_1754904910477"

# ============================================================================
# VERTEX AI-SPECIFIC FILE PATHS
# ============================================================================

# Get the directory of this config file (vertex-ai directory)
VERTEX_AI_DIR = os.path.dirname(os.path.abspath(__file__))

VERTEX_AI_OUTPUT_DIR = os.path.join(VERTEX_AI_DIR, "output")

NLQ_PARAMS_QUERIES = os.path.join(VERTEX_AI_OUTPUT_DIR, "nlq_params_queries.json")
GT_PARAMS_QUERIES = os.path.join(OUTPUT_DIR, "queries.json")
GT_PARAMS_OUTPUT_RESULTS = os.path.join(VERTEX_AI_OUTPUT_DIR, "gt_params_results.json")
NLQ_OUTPUT_RESULTS = os.path.join(VERTEX_AI_OUTPUT_DIR, "nlq_results.json")
FULL_TEXT_OUTPUT_RESULTS = os.path.join(VERTEX_AI_OUTPUT_DIR, "full_text_results.json")

# ============================================================================
# GCS BLOB PATHS (VERTEX AI-SPECIFIC)
# ============================================================================

GCS_BUCKET_NAME_DATA = GCS_BUCKET_NAME
GCS_BUCKET_PATH_DATA = GCS_BUCKET_FULL_PATH
