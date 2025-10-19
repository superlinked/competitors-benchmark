from superlinked import framework as sl

from superlinked_app.apps.sota_app.index import (average_rating_space,
                                                 class_space,
                                                 description_space, index,
                                                 price_space, product,
                                                 rating_count_space,
                                                 unique_categories)
from superlinked_app.apps.sota_app.nlq import openai_config

product_query = (
    sl.Query(
        index,
        weights={
            description_space: sl.Param(
                "description_space_weight", description="Doesn't need setting."
            ),
            class_space: sl.Param(
                "class_space_weight", description="Doesn't need setting."
            ),
            rating_count_space: sl.Param(
                "rating_count_weight",
                description="Set it to a positive number if user refers to popularity, lot of ratings, well known, etc.",
            ),
            average_rating_space: sl.Param(
                "rating_weight",
                description="Set it to a positive number if user refers to popularity, the product being acclaimed, high ratings, etc.",
            ),
            price_space: sl.Param(
                "price_weight",
                description="Cheap means a negative weight, expensive means a positive weight!",
            ),
        },
    )
    .find(product)
    .similar(
        description_space,
        sl.Param(
            "product_description",
            description="All textual information about a product. Feel free to include the full query text here!",
        ),
        sl.Param("description_similar_weight"),
    )
    .similar(
        class_space,
        sl.Param(
            "product_class",
            description="Category of a product from options. Only extract when there is an exact match. Be as specific as possible.",
            options=unique_categories["product_class"],
        ),
        sl.Param("class_similar_weight"),
    )
    .filter(
        product.price
        <= sl.Param("price_max", description="Maximum price for a product.")
    )
    .filter(
        product.average_rating
        >= sl.Param("rating_min", description="Minimum average rating for a product.")
    )
    .filter(
        product.rating_count
        >= sl.Param(
            "rating_count_min", description="Minimum number of ratings for a product."
        )
    )
    .filter(
        product.material.contains(
            sl.Param(
                "material",
                description="Materials from the options the query refers to as preferred or wanted.",
                options=unique_categories["material"],
            )
        )
    )
    .filter(
        product.color.contains(
            sl.Param(
                "color",
                description="Colors from the options the query refers to as preferred or wanted.",
                options=unique_categories["color"],
            )
        )
    )
    .filter(
        product.style.contains(
            sl.Param(
                "style",
                description="Styles from the options the query refers to as preferred or wanted.",
                options=unique_categories["style"],
            )
        )
    )
    .filter(
        product.material.not_contains(
            sl.Param(
                "material_negated",
                description="Materials from the options the query refers to as NOT wanted.",
                options=unique_categories["material"],
            )
        )
    )
    .filter(
        product.color.not_contains(
            sl.Param(
                "color_negated",
                description="Colors from the options the query refers to as NOT wanted.",
                options=unique_categories["color"],
            )
        )
    )
    .filter(
        product.style.not_contains(
            sl.Param(
                "style_negated",
                description="Styles from the options the query refers to as NOT wanted.",
                options=unique_categories["style"],
            )
        )
    )
    .with_natural_query(sl.Param("natural_query"), openai_config)
)

valid_query_params: list[str] = [
    "description_space_weight",
    "class_space_weight",
    "price_weight",
    "rating_count_weight",
    "rating_weight",
    "product_description",
    "description_similar_weight",
    "product_class",
    "class_similar_weight",
    "price_max",
    "rating_min",
    "rating_count_min",
    "material",
    "color",
    "style",
    "material_negated",
    "color_negated",
    "style_negated",
    "natural_query",
]
