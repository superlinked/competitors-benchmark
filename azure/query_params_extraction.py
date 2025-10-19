#!/usr/bin/env python3
"""
Azure OpenAI Query Feature Extraction

This module extracts structured search parameters from natural language queries
using Azure OpenAI. It processes queries and generates structured parameters
for enhanced search functionality.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config import (AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
                    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_ENDPOINT,
                    GCS_BUCKET_NAME, GCS_QUERIES_BLOB, NLQ_PARAMS_QUERIES,
                    PROMPT)
from google.cloud import storage
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_queries_from_bucket() -> List[Dict[str, Any]]:
    """
    Load queries from Google Cloud Storage bucket.

    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the queries file in the bucket

    Returns:
        List of query dictionaries

    Raises:
        Exception: If loading from bucket fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_QUERIES_BLOB)
        text = blob.download_as_text(encoding="utf-8")
        queries = json.loads(text)

        logger.info(f"Loaded {len(queries)} queries from bucket")
        return queries

    except Exception as e:
        logger.error(f"Failed to load queries from bucket: {e}")
        raise


def init_azure_openai_client() -> AzureOpenAI:
    """
    Initialize Azure OpenAI client.

    Returns:
        Configured AzureOpenAI client

    Raises:
        ValueError: If required API key is missing
    """
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

    client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    logger.info("Azure OpenAI client initialized")
    return client


def create_default_query_params() -> Dict[str, Any]:
    """
    Create default query parameters structure.

    Returns:
        Dictionary with default parameter values
    """
    return {
        "material": [],
        "color": [],
        "style": [],
        "product_class": [],
        "price_max": 0.0,
        "rating_count_min": 0.0,
        "rating_min": 0.0,
    }


def extract_query_params_with_llm(
    client: AzureOpenAI, query_text: str, query_id: str
) -> Dict[str, Any]:
    """
    Extract structured parameters from query using Azure OpenAI.

    Args:
        client: Azure OpenAI client
        query_text: Natural language query text
        query_id: Unique identifier for the query

    Returns:
        Dictionary containing extracted parameters
    """
    formatted_prompt = PROMPT().replace("{query_text}", query_text)

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured search parameters as strict JSON only.",
                },
                {"role": "user", "content": formatted_prompt},
            ],
        )

        response_text = ""
        if response.choices:
            response_text = (response.choices[0].message.content or "").strip()

        logger.debug(f"Raw LLM response for {query_id}: {response_text[:100]}...")
        return response_text

    except Exception as e:
        logger.error(f"Azure OpenAI request failed for query_id={query_id}: {e}")
        return ""


def clean_json_response(response_text: str) -> str:
    """
    Clean JSON response by removing markdown formatting.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Cleaned JSON string
    """
    if not response_text:
        return ""

    # Remove markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()


def parse_extracted_params(response_text: str) -> Dict[str, Any]:
    """
    Parse extracted parameters from LLM response.

    Args:
        response_text: Cleaned response text from LLM

    Returns:
        Dictionary containing parsed parameters or error info
    """
    if not response_text:
        return {"error": "Empty response from LLM"}

    try:
        parsed_params = json.loads(response_text)
        if isinstance(parsed_params, dict):
            return parsed_params
        else:
            return {"error": "Response is not a valid dictionary"}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {"error": "Failed to parse JSON"}


def process_single_query(client: AzureOpenAI, query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single query to extract parameters.

    Args:
        client: Azure OpenAI client
        query: Query dictionary containing query_id, query_text, etc.

    Returns:
        Dictionary with query info and extracted parameters
    """
    query_id = query["query_id"]
    query_text = query["query_text"]
    query_type = query.get("query_type")
    logger.info(f"Processing query: {query_text}")

    if query_type == "synth":
        return query

    # Create base query structure
    processed_query = {
        "query_id": query_id,
        "query_text": query_text,
        "query_type": query_type,
        "query_params": create_default_query_params(),
    }

    # Extract parameters using LLM
    response_text = extract_query_params_with_llm(client, query_text, query_id)

    if response_text:
        cleaned_response = clean_json_response(response_text)
        parsed_params = parse_extracted_params(cleaned_response)
        processed_query["query_params"] = parsed_params

    logger.info(f"Query params for {query_id}: {processed_query['query_params']}")
    return processed_query


def save_extracted_features(extracted_features: List[Dict[str, Any]]) -> None:
    """
    Save extracted features to output file.

    Args:
        extracted_features: List of processed queries with extracted parameters

    Raises:
        Exception: If saving fails
    """
    try:
        with open(NLQ_PARAMS_QUERIES, "w", encoding="utf-8") as f:
            json.dump(extracted_features, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Saved {len(extracted_features)} extracted features to {NLQ_PARAMS_QUERIES}"
        )

    except Exception as e:
        logger.error(f"Failed to save extracted features: {e}")
        raise


def main():
    try:
        # Initialize Azure OpenAI client
        client = init_azure_openai_client()

        # Load queries from bucket
        queries = load_queries_from_bucket()

        if not queries:
            logger.warning("No queries found to process")
            return

        # Process each query
        extracted_features = []
        for query in queries:
            try:
                processed_query = process_single_query(client, query)
                extracted_features.append(processed_query)
            except Exception as e:
                logger.error(
                    f"Failed to process query {query.get('query_id', 'unknown')}: {e}"
                )
                # Add query with error info
                error_query = {
                    "query_id": query.get("query_id", "unknown"),
                    "query_text": query.get("query_text", ""),
                    "query_type": query.get("query_type", "unknown"),
                    "query_params": {"error": f"Processing failed: {str(e)}"},
                }
                extracted_features.append(error_query)

        # Save results
        save_extracted_features(extracted_features)

        logger.info(f"Successfully processed {len(extracted_features)} queries")

    except Exception as e:
        logger.error(f"Query feature extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
