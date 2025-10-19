#!/usr/bin/env python3
"""
Evaluation Configuration

This module provides evaluation-specific configuration settings
that extend the shared base configuration.
"""

import os
import sys

# Add parent directory to path to import shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_config import *

# ============================================================================
# EVALUATION-SPECIFIC FILE PATHS
# ============================================================================

# Get the directory of this config file (evaluation directory)
EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVALUATION_DIR)

# ============================================================================
# SUPERLINKED RESULTS BLOB PATHS
# ============================================================================

GCS_SL_GT_PARAMS_RESULTS_BLOB = f"{GCS_BUCKET_PATH.rstrip('/')}/outputs/{GCS_BUCKET_VERSION}/{GCS_BUCKET_VERSION}_sota_app_gt_params/evaluation_results.pkl"
GCS_SL_NLQ_RESULTS_BLOB = f"{GCS_BUCKET_PATH.rstrip('/')}/outputs/{GCS_BUCKET_VERSION}/{GCS_BUCKET_VERSION}_sota_app_nlq/evaluation_results.pkl"
GCS_SL_FULL_TEXT_RESULTS_BLOB = f"{GCS_BUCKET_PATH.rstrip('/')}/outputs/{GCS_BUCKET_VERSION}/{GCS_BUCKET_VERSION}_sota_app_full_text/evaluation_results.pkl"

# ============================================================================
# HYBRID SEARCH RESULTS PATHS
# ============================================================================

HYBRID_SEARCH_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "vertex-ai", "hybrid-search", "output", "ranked_results.json"
)

# ============================================================================
# VERTEX AI RESULTS PATHS
# ============================================================================

VERTEX_AI_GT_PARAMS_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "vertex-ai", "output", "gt_params_results.json"
)
VERTEX_AI_NLQ_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "vertex-ai", "output", "nlq_results.json"
)
VERTEX_AI_FULL_TEXT_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "vertex-ai", "output", "full_text_results.json"
)

# ============================================================================
# AZURE RESULTS PATHS
# ============================================================================

AZURE_GT_PARAMS_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "azure", "output", "gt_params_results.json"
)
AZURE_NLQ_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "azure", "output", "nlq_results.json"
)
AZURE_FULL_TEXT_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "azure", "output", "full_text_results.json"
)
