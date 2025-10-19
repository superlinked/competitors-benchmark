from superlinked import framework as sl

from superlinked_app.apps.sota_app_baseline.config import config
from superlinked_app.apps.sota_app_baseline.index import (index, info_space,
                                                          product)

product_query = (
    sl.Query(
        index,
        weights={
            info_space: 1.0,
        },
    )
    .find(product)
    .similar(
        info_space,
        sl.Param(config.query_text_colname),
        1.0,
    )
)
