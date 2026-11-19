"""Feature schema shared by training, prediction, search and the UI.

A house is described by a flat dict of *inputs* (what a user can type in a
form). ``engineer`` turns a DataFrame of inputs into the exact model matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}


@dataclass(frozen=True)
class NumericSpec:
    key: str
    min: float
    max: float
    step: float = 1
    is_area: bool = False  # stored in square feet, shown in the chosen unit


NUMERIC: list[NumericSpec] = [
    NumericSpec("gr_liv_area", 300, 6000, 10, True),
    NumericSpec("total_bsmt_sf", 0, 6000, 10, True),
    NumericSpec("bsmt_fin_sf", 0, 6000, 10, True),
    NumericSpec("lot_area", 1000, 220000, 100, True),
    NumericSpec("porch_deck_sf", 0, 2000, 10, True),
    NumericSpec("overall_qual", 1, 10),
    NumericSpec("overall_cond", 1, 10),
    NumericSpec("year_built", 1870, 2025),
    NumericSpec("year_remod", 1870, 2025),
    NumericSpec("bedrooms", 0, 8),
    NumericSpec("tot_rooms", 2, 15),
    NumericSpec("full_bath", 0, 4),
    NumericSpec("half_bath", 0, 2),
    NumericSpec("bsmt_full_bath", 0, 3),
    NumericSpec("garage_cars", 0, 5),
    NumericSpec("fireplaces", 0, 4),
]
NUMERIC_BY_KEY = {s.key: s for s in NUMERIC}

# Rated 1 (poor) .. 5 (excellent); basement quality 0 means "no basement".
ORDINAL = ["exter_qual", "kitchen_qual", "bsmt_qual", "heating_qc"]
BINARY = ["central_air"]
CATEGORICAL = [
    "neighborhood",
    "ms_zoning",
    "bldg_type",
    "house_style",
    "garage_type",
    "foundation",
    "sale_condition",
]
ENGINEERED = ["total_sf", "total_baths", "qual_sf", "house_age_at_remod"]

INPUT_KEYS = [s.key for s in NUMERIC] + ORDINAL + BINARY + CATEGORICAL
MODEL_FEATURES = INPUT_KEYS + ENGINEERED

# Real-estate common sense enforced in the model: more area, quality, baths,
# garage, newer construction, etc. can never lower the price (all else equal).
# Bedrooms and room counts stay free: at fixed area, more rooms can mean less value.
INCREASING = {
    "gr_liv_area", "total_bsmt_sf", "bsmt_fin_sf", "lot_area", "porch_deck_sf", "overall_qual",
    "overall_cond", "year_built", "year_remod", "full_bath", "half_bath", "bsmt_full_bath",
    "garage_cars", "fireplaces", "exter_qual", "kitchen_qual", "bsmt_qual", "heating_qc",
    "central_air", "total_sf", "total_baths", "qual_sf",
}
MONOTONE_CONSTRAINTS = tuple(1 if f in INCREASING else 0 for f in MODEL_FEATURES)

# Column mapping from the raw Ames CSV.
_RAW_NUMERIC = {
    "gr_liv_area": ["Gr Liv Area"],
    "total_bsmt_sf": ["Total Bsmt SF"],
    "bsmt_fin_sf": ["BsmtFin SF 1", "BsmtFin SF 2"],
    "lot_area": ["Lot Area"],
    "porch_deck_sf": ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"],
    "overall_qual": ["Overall Qual"],
    "overall_cond": ["Overall Cond"],
    "year_built": ["Year Built"],
    "year_remod": ["Year Remod/Add"],
    "bedrooms": ["Bedroom AbvGr"],
    "tot_rooms": ["TotRms AbvGrd"],
    "full_bath": ["Full Bath"],
    "half_bath": ["Half Bath"],
    "bsmt_full_bath": ["Bsmt Full Bath"],
    "garage_cars": ["Garage Cars"],
    "fireplaces": ["Fireplaces"],
}
_RAW_ORDINAL = {
    "exter_qual": "Exter Qual",
    "kitchen_qual": "Kitchen Qual",
    "bsmt_qual": "Bsmt Qual",
    "heating_qc": "Heating QC",
}
_RAW_CATEGORICAL = {
    "neighborhood": "Neighborhood",
    "ms_zoning": "MS Zoning",
    "bldg_type": "Bldg Type",
    "house_style": "House Style",
    "garage_type": "Garage Type",
    "foundation": "Foundation",
    "sale_condition": "Sale Condition",
}


def inputs_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Ames CSV rows into the input schema."""
    out = pd.DataFrame(index=raw.index)
    for key, cols in _RAW_NUMERIC.items():
        out[key] = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in cols)
    for key, col in _RAW_ORDINAL.items():
        default = 0 if key == "bsmt_qual" else 3
        out[key] = raw[col].map(QUALITY_MAP).fillna(default).astype(int)
    out["central_air"] = (raw["Central Air"] == "Y").astype(int)
    for key, col in _RAW_CATEGORICAL.items():
        out[key] = raw[col].fillna("NA").astype(str).str.strip()
    return out


def engineer(inputs: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Build the model matrix. ``categories`` pins the category order seen in training."""
    df = inputs.copy()
    for key in [s.key for s in NUMERIC] + ORDINAL + BINARY:
        df[key] = pd.to_numeric(df[key], errors="coerce").astype(float)
    df["total_sf"] = df["gr_liv_area"] + df["total_bsmt_sf"]
    df["total_baths"] = df["full_bath"] + 0.5 * df["half_bath"] + df["bsmt_full_bath"]
    df["qual_sf"] = df["overall_qual"] * df["total_sf"]
    df["house_age_at_remod"] = (df["year_remod"] - df["year_built"]).clip(lower=0)
    for key in CATEGORICAL:
        df[key] = pd.Categorical(df[key].astype(str), categories=categories[key])
    return df[MODEL_FEATURES]


def default_inputs(market: pd.DataFrame, neighborhood: str | None = None) -> dict:
    """Typical house (median / mode), optionally within one neighbourhood."""
    pool = market
    if neighborhood is not None:
        subset = market[market["neighborhood"] == neighborhood]
        if len(subset) >= 5:
            pool = subset
    result: dict = {}
    for spec in NUMERIC:
        result[spec.key] = float(np.round(pool[spec.key].median() / spec.step) * spec.step)
    for key in ORDINAL + BINARY:
        result[key] = int(pool[key].median())
    for key in CATEGORICAL:
        result[key] = str(pool[key].mode().iloc[0])
    result["sale_condition"] = "Normal"
    if neighborhood is not None:
        result["neighborhood"] = neighborhood
    return result


def normalize_inputs(inputs: dict) -> dict:
    """Make a user-entered house internally consistent (never raises)."""
    out = dict(inputs)
    for spec in NUMERIC:
        value = float(out.get(spec.key, spec.min) or 0)
        out[spec.key] = float(min(max(value, spec.min), spec.max))
    out["year_remod"] = max(out["year_remod"], out["year_built"])
    out["bsmt_fin_sf"] = min(out["bsmt_fin_sf"], out["total_bsmt_sf"])
    if out["total_bsmt_sf"] <= 0:
        out["bsmt_qual"] = 0
        out["bsmt_full_bath"] = 0.0
    elif int(out.get("bsmt_qual", 0)) == 0:
        out["bsmt_qual"] = 3
    if out["garage_cars"] <= 0:
        out["garage_type"] = "NA"
    elif out.get("garage_type", "NA") == "NA":
        out["garage_type"] = "Attchd"
    out["tot_rooms"] = max(out["tot_rooms"], out["bedrooms"] + 1)
    for key in ORDINAL + BINARY:
        out[key] = int(out.get(key, 0))
    for key in CATEGORICAL:
        out[key] = str(out.get(key, "NA"))
    return out
