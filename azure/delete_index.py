import logging

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


def delete_index():
    """Delete the existing index"""

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT, credential=credential
    )

    try:
        index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
        logger.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting index: {e}")


if __name__ == "__main__":
    delete_index()
