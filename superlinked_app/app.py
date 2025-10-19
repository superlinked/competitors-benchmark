import json
import logging
import os
import time
from typing import cast

import pandas as pd
import typer
from dotenv import load_dotenv
from superlinked import framework as sl
from tqdm import tqdm

from superlinked_app.apps.sota_app.query import valid_query_params
from superlinked_app.registry import MODULE_REGISTRY
from superlinked_app.util.enum import QueryMode
from superlinked_app.util.util import (download_pickle_from_gcs,
                                       download_text_from_gcs,
                                       upload_pickle_to_gcs,
                                       upload_text_to_gcs)

logging.basicConfig(level=logging.INFO)


def run(
    superlinked_app_folder: str = typer.Argument(
        ..., help="The config folder name (e.g., 'superlinked_app')"
    )
):
    modules = MODULE_REGISTRY[superlinked_app_folder]
    data_prep = modules["data_prep"].data_prep
    settings = modules["config"].config
    index = modules["index"]
    query = modules["query"]

    load_dotenv(f"superlinked/apps/{superlinked_app_folder}/.env")

    product = index.product
    product_index = index.index

    product_dataframe_parser = sl.DataFrameParser(product)
    source = sl.InMemorySource(schema=product, parser=product_dataframe_parser)

    redis_vdb = sl.RedisVectorDatabase(
        settings.redis_url,
        settings.redis_port,
        username=settings.redis_username,
        password=os.environ["REDIS_PASSWORD"],
    )

    executor = sl.InteractiveExecutor(
        sources=[source], indices=[product_index], vector_database=redis_vdb
    )
    app = executor.run()

    if settings.reingest:
        product_data = pd.read_json(
            f"gs://{settings.bucket_name}/{settings.product_dataset_path}", lines=True
        )
        product_data = data_prep(product_data, settings)
        logging.info("Started ingestion.")
        source.put(product_data)

    input_queries = pd.DataFrame(
        json.loads(
            download_text_from_gcs(settings.bucket_name, settings.query_dataset_path)
        )
    ).set_index("query_id")

    search_params: dict[str, dict[str, str | float | int | list[str]]] = {}
    results = {}
    query_latencies: dict[str, float] = {}
    match settings.query_mode:
        case QueryMode.USE_FULL_QUERY_TEXT:
            search_params = {
                idx: {settings.query_text_colname: val}
                for idx, val in zip(
                    input_queries.index,
                    input_queries.loc[:, settings.query_text_colname].tolist(),
                )
            }
        case QueryMode.USE_GROUND_TRUTH_QUERY_INPUTS:
            search_params = input_queries[settings.query_params_colname].to_dict()
            search_params = {
                query_id: {
                    param: value
                    for param, value in query_params.items()
                    if param in valid_query_params
                }
                for query_id, query_params in search_params.items()
            }

        case QueryMode.USE_NLQ:
            if settings.redo_nlq:
                logging.info("Starting from a fresh set of NLQ results.")
            else:
                try:
                    search_params = download_pickle_from_gcs(
                        settings.bucket_name, settings.search_param_output_path
                    )
                    logging.info("NLQ results loaded from cache.")
                except Exception as e:
                    logging.warning(f"Unable to load results due to exception: {e}")
                    search_params = {}
                    logging.info("Starting from a fresh set of NLQ results.")
            for query_id, query_text in tqdm(
                input_queries["query_text"].items(),
                desc=f"Running {len(input_queries.index)} queries... (NLQ part)",
            ):
                query_id = cast(str, query_id)
                query_text = cast(str, query_text)
                if query_id in search_params.keys():
                    continue
                try:
                    st_time = time.time()
                    query_result = app.query(
                        query.product_query, natural_query=query_text
                    )
                    end_time = time.time()
                    search_params[query_id] = query_result.metadata.search_params
                    results[query_id] = query_result
                    query_latencies[query_id] = end_time - st_time
                except Exception as e:
                    logging.error(e)
                    upload_text_to_gcs(
                        str(e),
                        settings.bucket_name,
                        f"{settings.error_output_path}/{query_id}.txt",
                    )
            upload_text_to_gcs(
                json.dumps(query_latencies),
                settings.bucket_name,
                settings.query_latency_output_path,
            )

    for query_id in tqdm(
        input_queries["query_text"].index,
        desc=f"Running {len(input_queries.index)} queries... (SL query part)",
    ):
        if query_id in results.keys():
            continue

        filtered_search_params = {
            query_id: {
                param: value
                for param, value in query_search_params.items()
                if not param == "natural_query"
            }
            for query_id, query_search_params in search_params.items()
        }
        results[query_id] = app.query(
            query.product_query, **filtered_search_params[query_id]
        )

    upload_pickle_to_gcs(
        search_params, settings.bucket_name, settings.search_param_output_path
    )

    evaluation_results = []
    for query_id, result in results.items():
        for i, entry in enumerate(result.entries):
            evaluation_results.append(
                (
                    query_id,
                    entry.id,
                    i + 1,
                    input_queries.loc[query_id, settings.query_type_colname],
                )
            )

    upload_pickle_to_gcs(
        evaluation_results, settings.bucket_name, settings.eval_result_output_path
    )


def main():
    typer.run(run)


if __name__ == "__main__":
    main()
