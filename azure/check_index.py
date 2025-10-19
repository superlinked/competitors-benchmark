import json
import logging

import requests
from config import (AZURE_SEARCH_API_KEY, AZURE_SEARCH_ENDPOINT,
                    AZURE_SEARCH_INDEX_NAME)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def check_index():
    """Check the index definition using REST API"""

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}?api-version=2023-11-01"
    headers = {"api-key": AZURE_SEARCH_API_KEY}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        index_def = response.json()
        logger.info("Index definition:")
        print(f"Name: {index_def.get('name')}")
        print(f"Fields: {len(index_def.get('fields', []))}")

        # Check for semantic configuration
        if "semantic" in index_def:
            logger.info("Semantic configuration found!")
            semantic_config = index_def["semantic"]
            print(f"Configurations: {len(semantic_config.get('configurations', []))}")
            for config in semantic_config.get("configurations", []):
                print(f"  - {config.get('name')}")
                if "prioritizedFields" in config:
                    pf = config["prioritizedFields"]
                    if "titleField" in pf:
                        logger.info(f"    Title field: {pf['titleField']['fieldName']}")
                    if "prioritizedContentFields" in pf:
                        content_fields = [
                            f["fieldName"] for f in pf["prioritizedContentFields"]
                        ]
                        logger.info(f"    Content fields: {content_fields}")
                    if "prioritizedKeywordsFields" in pf:
                        keyword_fields = [
                            f["fieldName"] for f in pf["prioritizedKeywordsFields"]
                        ]
                        logger.info(f"    Keyword fields: {keyword_fields}")
        else:
            logger.info("No semantic configuration found in REST API response")

        # Print full response for debugging
        logger.info("\nFull index definition:")

        logger.info(json.dumps(index_def, indent=2))

    else:
        logger.info(f"Failed to get index: {response.status_code}")
        logger.info(response.text)


if __name__ == "__main__":
    check_index()
