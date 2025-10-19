import pandas as pd

from superlinked_app.apps.sota_app_baseline.config import config


def data_prep(
    product_df: pd.DataFrame,
    configuration=config,
) -> pd.DataFrame:
    product_df["product_id"] = product_df["product_id"].astype("str")
    product_df["product_info"] = product_df.apply(
        lambda row: ", ".join(
            [f"{col}: {str(row[col])}" for col in configuration.join_cols]
        ),
        axis=1,
    )
    return product_df[["product_id", "product_info"]]
