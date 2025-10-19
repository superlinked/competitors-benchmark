import json
import pickle
import re
from enum import Enum
from io import BytesIO

import pandas as pd
from google.cloud import storage


class ArtifactType(Enum):
    CODE = "code"
    OUTPUT = "output"
    MODEL = "model"


def get_unique_categories(series: pd.Series) -> list[str]:
    return series.dropna().unique().tolist()


def upload_pickle_to_gcs(data, bucket_name, destination_blob_name) -> None:
    buffer = BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(buffer)
    return None


def download_pickle_from_gcs(bucket_name, source_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    buffer = BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return pickle.load(buffer)


def download_text_from_gcs(bucket_name, file_path):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    return blob.download_as_text()


def upload_text_to_gcs(text, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(text, content_type="text/plain")

    return None


def clean_value(value: str) -> str:
    cleaned_value = (
        re.sub(r"[^a-zA-Z0-9]", " ", value) if isinstance(value, str) else value
    )
    cleaned_value = (
        re.sub(r"\s+", " ", cleaned_value).strip() if isinstance(value, str) else value
    )
    return cleaned_value
