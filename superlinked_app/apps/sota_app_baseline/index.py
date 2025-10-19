from superlinked import framework as sl

from superlinked_app.apps.sota_app_baseline.config import config


class Product(sl.Schema):
    product_id: sl.IdField
    product_info: sl.String | None


product = Product()

info_space = sl.TextSimilaritySpace(
    product.product_info, model=config.text_embedding_model
)

index = sl.Index(
    [info_space],
)
