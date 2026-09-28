"""Колонки запроса и объявления, как в исходных parquet."""

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]
