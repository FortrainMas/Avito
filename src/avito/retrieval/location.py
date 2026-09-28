"""Доли выборов по локациям и центры локаций поиска, посчитанные по train."""

import numpy as np
import polars as pl


class LocationPrior:
    """P(локация объявления | локация поиска) — доля выборов в train. Для поиска по городу масса почти вся
    на самом городе и пригородах, для поиска по региону раскладывается по городам региона."""

    def fit(self, pairs: pl.DataFrame) -> "LocationPrior":
        counts = pairs.group_by("search_location_id", "item_location_id").len("count")
        totals = counts.group_by("search_location_id").agg(pl.col("count").sum().alias("total"))
        counts = counts.join(totals, on="search_location_id")
        counts = counts.with_columns((pl.col("count") / pl.col("total")).alias("probability"))
        self.table = counts.select("search_location_id", "item_location_id", "probability")

        self.transitions = {}
        for group in self.table.partition_by("search_location_id"):
            search_location_id = group["search_location_id"][0]
            item_location_ids = group["item_location_id"].to_list()
            probabilities = group["probability"].to_list()
            self.transitions[search_location_id] = dict(zip(item_location_ids, probabilities))
        return self

    def probabilities(self, search_location_id: int, item_location_ids: pl.Series) -> np.ndarray:
        transitions = self.transitions.get(search_location_id, {})
        probabilities = item_location_ids.replace_strict(transitions, default=0.0, return_dtype=pl.Float32)
        return probabilities.to_numpy()


def fit_search_centroids(pairs: pl.DataFrame) -> pl.DataFrame:
    """Центр локации поиска — медиана координат объявлений, выбранных при поиске в ней."""
    return pairs.group_by("search_location_id").agg(
        pl.col("item_latitude").median().alias("search_latitude"),
        pl.col("item_longitude").median().alias("search_longitude"),
    )


def fit_location_stats(train: pl.DataFrame, items: pl.DataFrame) -> tuple[LocationPrior, pl.DataFrame]:
    items = items.select("item_id", "item_location_id", "item_latitude", "item_longitude")
    pairs = train.select("search_location_id", "item_id").join(items, on="item_id")
    return LocationPrior().fit(pairs), fit_search_centroids(pairs)
