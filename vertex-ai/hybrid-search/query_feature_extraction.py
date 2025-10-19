#!/usr/bin/env python3
"""
Hybrid Search Query Feature Extraction

This module extracts structured search parameters from natural language queries
using Google's Gemini model. It processes queries and generates structured parameters
for enhanced hybrid search functionality with Qdrant.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config import (GCS_BUCKET_NAME, GCS_QUERIES_BLOB, GEMINI_MODEL_NAME,
                    GOOGLE_API_KEY, NLQ_PARAMS_QUERIES, PROMPT)
from google.cloud import storage
from google.genai import Client

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


def create_gemini_client() -> Client:
    """
    Create and initialize Gemini client.

    Returns:
        Configured Gemini client

    Raises:
        ValueError: If required API key is missing
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    client = Client(api_key=GOOGLE_API_KEY)
    logger.info("Gemini client initialized")
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


def extract_query_params_with_gemini(
    client: Client, query_text: str, query_id: str
) -> str:
    """
    Extract structured parameters from query using Gemini.

    Args:
        client: Gemini client
        query_text: Natural language query text
        query_id: Unique identifier for the query

    Returns:
        Response text from Gemini

    Raises:
        Exception: If Gemini request fails
    """
    try:
        formatted_prompt = PROMPT().replace("{query_text}", query_text)

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=formatted_prompt,
        )

        response_text = response.text.strip()
        logger.debug(f"Raw Gemini response for {query_id}: {response_text[:100]}...")
        return response_text

    except Exception as e:
        logger.error(f"Gemini request failed for query_id={query_id}: {e}")
        raise


def clean_json_response(response_text: str) -> str:
    """
    Clean JSON response by removing markdown formatting.

    Args:
        response_text: Raw response text from Gemini

    Returns:
        Cleaned JSON string
    """
    if not response_text:
        return ""

    # Remove markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    if response_text.startswith("```"):
        response_text = response_text[3:]  # Remove ```
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove ```

    return response_text.strip()


def parse_extracted_params(response_text: str) -> Dict[str, Any]:
    """
    Parse extracted parameters from Gemini response.

    Args:
        response_text: Cleaned response text from Gemini

    Returns:
        Dictionary containing parsed parameters or error info
    """
    if not response_text:
        return {"error": "Empty response from Gemini"}

    try:
        parsed_params = json.loads(response_text)
        if isinstance(parsed_params, dict):
            return parsed_params
        else:
            return {"error": "Response is not a valid dictionary"}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {"error": "Failed to parse JSON"}


def process_single_query(client: Client, query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single query to extract parameters.

    Args:
        client: Gemini client
        query: Query dictionary containing query_id, query_text, etc.

    Returns:
        Dictionary with query info and extracted parameters
    """
    query_id = query["query_id"]
    query_text = query["query_text"]

    logger.info(f"Processing query: {query_text}")

    # Create base query structure
    processed_query = {
        "query_id": query_id,
        "query_text": query_text,
        "query_type": query.get("query_type", "unknown"),
        "query_params": create_default_query_params(),
    }

    try:
        # Extract parameters using Gemini
        response_text = extract_query_params_with_gemini(client, query_text, query_id)

        if response_text:
            cleaned_response = clean_json_response(response_text)
            parsed_params = parse_extracted_params(cleaned_response)
            processed_query["query_params"] = parsed_params

        logger.info(f"Query params for {query_id}: {processed_query['query_params']}")
        return processed_query

    except Exception as e:
        logger.error(f"Failed to process query {query_id}: {e}")
        # Return query with error info
        processed_query["query_params"] = {"error": f"Processing failed: {str(e)}"}
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
        # Initialize Gemini client
        client = create_gemini_client()

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
