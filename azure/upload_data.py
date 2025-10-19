#!/usr/bin/env python3
"""
Azure Search Data Upload

This module handles uploading product data to Azure Cognitive Search index.
Supports batch uploading for efficient data transfer.
"""

import json
import logging
from typing import Any, Dict, List

from config import (AZURE_SEARCH_API_KEY, AZURE_SEARCH_ENDPOINT,
                    AZURE_SEARCH_INDEX_NAME, PRODUCTS_FILE)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_products_from_file() -> List[Dict[str, Any]]:
    """
    Load products from JSONL file.

    Args:
        file_path: Path to the products JSONL file

    Returns:
        List of product dictionaries

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    products = []
    try:
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    product = json.loads(line)
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"Products file not found: {PRODUCTS_FILE}")
        raise

    logger.info(f"Loaded {len(products)} products from {PRODUCTS_FILE}")
    return products


def create_search_client() -> SearchClient:
    """
    Create and return Azure Search client.

    Returns:
        Configured SearchClient instance

    Raises:
        ValueError: If required credentials are missing
    """
    if not AZURE_SEARCH_API_KEY:
        raise ValueError("AZURE_SEARCH_API_KEY environment variable is required")

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )

    logger.info(f"Created search client for index: {AZURE_SEARCH_INDEX_NAME}")
    return search_client


def upload_products_batch(
    search_client: SearchClient, products: List[Dict[str, Any]], batch_size: int = 1000
) -> None:
    """
    Upload products to Azure Search in batches.

    Args:
        search_client: Azure Search client
        products: List of product dictionaries
        batch_size: Number of documents per batch

    Raises:
        Exception: If upload fails
    """
    total_batches = (len(products) + batch_size - 1) // batch_size
    logger.info(f"Uploading {len(products)} products in {total_batches} batches")

    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            search_client.upload_documents(documents=batch)
            logger.info(
                f"Uploaded batch {batch_num}/{total_batches}: {len(batch)} documents"
            )
        except Exception as e:
            logger.error(f"Failed to upload batch {batch_num}: {e}")
            raise

    logger.info(f"Successfully uploaded {len(products)} products")


def upload_products() -> None:
    """
    Main function to upload products to Azure Search.

    Loads products from file and uploads them to the configured Azure Search index.
    """
    try:
        # Load products from file
        products = load_products_from_file()

        if not products:
            logger.warning("No products to upload")
            return

        # Create search client
        search_client = create_search_client()

        # Upload products
        upload_products_batch(search_client, products)

        logger.info("Product upload completed successfully")

    except Exception as e:
        logger.error(f"Product upload failed: {e}")
        raise


if __name__ == "__main__":
    upload_products()
