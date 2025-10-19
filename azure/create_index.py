#!/usr/bin/env python3
"""
Azure Cognitive Search Index Creation

This module creates the Azure Cognitive Search index with semantic configuration
and comprehensive field definitions for product search.
"""

import logging
from typing import Any, Dict, List

import requests
from config import (AZURE_SEARCH_API_KEY, AZURE_SEARCH_ENDPOINT,
                    AZURE_SEARCH_INDEX_NAME)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def create_search_index_definition() -> Dict[str, Any]:
    """
    Create the Azure Search index definition with all required fields.

    Returns:
        Dictionary containing the complete index definition
    """
    return {
        "name": AZURE_SEARCH_INDEX_NAME,
        "fields": [
            {
                "name": "product_id",
                "type": "Edm.String",
                "key": True,
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
            {
                "name": "product_name",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
            {
                "name": "product_description",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": False,
                "facetable": False,
            },
            {
                "name": "product_class",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
            {
                "name": "material",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "sortable": False,
                "facetable": True,
            },
            {
                "name": "style",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "sortable": False,
                "facetable": True,
            },
            {
                "name": "color",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "sortable": False,
                "facetable": True,
            },
            {
                "name": "rating_count",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
            {
                "name": "average_rating",
                "type": "Edm.Double",
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
            {
                "name": "countryoforigin",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "sortable": False,
                "facetable": True,
            },
            {
                "name": "price",
                "type": "Edm.Double",
                "filterable": True,
                "sortable": True,
                "facetable": True,
            },
        ],
        "semantic": {
            "configurations": [
                {
                    "name": "default",
                    "prioritizedFields": {
                        "titleField": {"fieldName": "product_name"},
                        "prioritizedContentFields": [
                            {"fieldName": "product_description"},
                            {"fieldName": "product_class"},
                        ],
                        "prioritizedKeywordsFields": [
                            {"fieldName": "product_class"},
                            {"fieldName": "material"},
                            {"fieldName": "style"},
                            {"fieldName": "color"},
                            {"fieldName": "countryoforigin"},
                        ],
                    },
                }
            ]
        },
    }


def create_search_client() -> SearchIndexClient:
    """
    Create Azure Search Index client.

    Returns:
        Configured SearchIndexClient instance

    Raises:
        ValueError: If required credentials are missing
    """
    if not AZURE_SEARCH_API_KEY:
        raise ValueError("AZURE_SEARCH_API_KEY environment variable is required")

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT, credential=credential
    )

    logger.info(f"Created search index client for endpoint: {AZURE_SEARCH_ENDPOINT}")
    return index_client


def create_index_via_rest_api(index_definition: Dict[str, Any]) -> bool:
    """
    Create index using REST API (more reliable for complex configurations).

    Args:
        index_definition: Complete index definition dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version=2023-11-01"
        headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}

        logger.info(f"Creating index '{AZURE_SEARCH_INDEX_NAME}' via REST API")
        response = requests.put(url, headers=headers, json=index_definition)

        if response.status_code in [200, 201]:
            logger.info(
                f"Semantic index '{AZURE_SEARCH_INDEX_NAME}' created successfully"
            )
            return True
        else:
            logger.error(f"Failed to create index: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"REST API request failed: {e}")
        return False


def check_index_exists() -> bool:
    """
    Check if the index already exists.

    Returns:
        True if index exists, False otherwise
    """
    try:
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version=2023-11-01"
        headers = {"api-key": AZURE_SEARCH_API_KEY}

        response = requests.get(url, headers=headers)
        return response.status_code == 200

    except Exception as e:
        logger.warning(f"Could not check index existence: {e}")
        return False


def create_products_index() -> bool:
    """
    Create the products search index with semantic configuration.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if index already exists
        if check_index_exists():
            logger.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists")
            return True

        # Create index definition
        index_definition = create_search_index_definition()

        # Create search client (for potential SDK usage)
        index_client = create_search_client()

        # Create index using REST API (most reliable method)
        success = create_index_via_rest_api(index_definition)

        if success:
            logger.info("Index creation completed successfully")
        else:
            logger.error("Index creation failed")

        return success

    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        return False


def main():
    try:
        success = create_products_index()

        if success:
            logger.info("Azure Search index setup completed successfully")
        else:
            logger.error("Azure Search index setup failed")
            raise Exception("Index creation failed")

    except Exception as e:
        logger.error(f"Index creation process failed: {e}")
        raise


if __name__ == "__main__":
    main()
