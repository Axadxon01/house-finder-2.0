"""Loading and cleaning the Ames housing dataset."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import inputs_from_raw


def load_raw(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    # "NA" is a real category in Ames (e.g. "no garage"), so keep it as text.
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"PID": str})


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop the documented partial-sale outliers (huge houses sold cheaply)."""
    mask = (raw["Gr Liv Area"] > 4000) & (raw["SalePrice"] < 300000)
    return raw.loc[~mask].reset_index(drop=True)


def _jitter(pid: str, scale: float = 0.0045) -> tuple[float, float]:
    seed = zlib.crc32(str(pid).encode())
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-scale, scale, size=2))  # type: ignore[return-value]


def coordinates(neighborhood: str, pid: str) -> tuple[float, float]:
    """Approximate, deterministic map position for a house."""
    _, lat, lon = config.NEIGHBORHOODS.get(neighborhood, ("", *config.DEFAULT_CENTER))
    dlat, dlon = _jitter(pid)
    return lat + dlat, lon + dlon


def build_market(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per real sale: model inputs + price + derived analytics fields."""
    market = inputs_from_raw(raw)
    market["pid"] = raw["PID"].astype(str).values
    market["sale_price"] = raw["SalePrice"].astype(float).values
    market["yr_sold"] = raw["Yr Sold"].astype(int).values
    market["mo_sold"] = raw["Mo Sold"].astype(int).values
    market["sale_date"] = pd.to_datetime(
        dict(year=market["yr_sold"], month=market["mo_sold"], day=1)
    )
    market["total_sf"] = market["gr_liv_area"] + market["total_bsmt_sf"]
    market["total_baths"] = market["full_bath"] + 0.5 * market["half_bath"] + market["bsmt_full_bath"]
    market["price_per_sqft"] = market["sale_price"] / market["gr_liv_area"]
    coords = [coordinates(n, p) for n, p in zip(market["neighborhood"], market["pid"])]
    market["lat"] = [c[0] for c in coords]
    market["lon"] = [c[1] for c in coords]
    return market


def load_market() -> pd.DataFrame:
    return build_market(clean(load_raw()))
