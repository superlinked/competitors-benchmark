import json

from dotenv import load_dotenv
from superlinked import framework as sl

from superlinked_app.apps.sota_app.config import config
from superlinked_app.util.util import download_text_from_gcs

load_dotenv("./superlinked_app/apps/sota_app/.env")


class Product(sl.Schema):
    product_id: sl.IdField
    product_description: sl.String
    product_class: sl.String
    material: sl.StringList
    style: sl.StringList
    color: sl.StringList
    rating_count: sl.Float
    average_rating: sl.Float
    price: sl.Float


unique_categories: dict[str, list[str]] = json.loads(
    download_text_from_gcs(
        bucket_name=config.bucket_name, file_path=config.unique_categories_path
    )
)
UNCATEGORIZED_SETTING: bool = False

product = Product()

description_space = sl.TextSimilaritySpace(
    product.product_description, model=config.text_embedding_model
)
class_space = sl.TextSimilaritySpace(
    product.product_class, model=config.text_embedding_model
)
rating_count_space = sl.NumberSpace(
    product.rating_count,
    min_value=0.0,
    max_value=8000.0,
    mode=sl.Mode.MAXIMUM,
    scale=sl.LogarithmicScale(),
)
average_rating_space = sl.NumberSpace(
    product.average_rating, min_value=0.0, max_value=5.0, mode=sl.Mode.MAXIMUM
)
price_space = sl.NumberSpace(
    product.price,
    min_value=0.0,
    max_value=5900,
    mode=sl.Mode.MAXIMUM,
    scale=sl.LogarithmicScale(),
)

index = sl.Index(
    [
        description_space,
        class_space,
        rating_count_space,
        average_rating_space,
        price_space,
    ],
    fields=[
        product.rating_count,
        product.price,
        product.average_rating,
        product.product_class,
        product.material,
        product.style,
        product.color,
    ],
)
