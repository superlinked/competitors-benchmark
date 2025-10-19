import pandas as pd

from superlinked_app.apps.sota_app.config import config


def data_prep(
    product_df: pd.DataFrame,
    settings=config,
) -> pd.DataFrame:
    product_df["product_id"] = product_df["product_id"].astype("str")
    for float_col in settings.float_cols:
        product_df[float_col] = product_df[float_col].astype("float32")

    product_df["product_description"] = (
        product_df["product_name"] + " " + product_df["product_description"]
    )

    return product_df
