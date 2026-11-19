"""Paths and static reference data."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATASET_PATH = DATA_DIR / "AmesHousing.csv"
MODEL_PATH = MODELS_DIR / "price_model.ubj"
METADATA_PATH = MODELS_DIR / "metadata.json"
DB_PATH = Path(os.environ.get("HOUSEFINDER_DB", DATA_DIR / "housefinder.db"))

SQFT_PER_M2 = 10.7639
RANDOM_STATE = 42
HOLDOUT_SIZE = 0.15
DEMO_LISTINGS = 60
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Ames, Iowa city centre.
DEFAULT_CENTER = (42.0308, -93.6319)

# Neighbourhood code -> (full name, approximate centroid lat, lon).
# Coordinates are approximate centroids used only for map display.
NEIGHBORHOODS: dict[str, tuple[str, float, float]] = {
    "Blmngtn": ("Bloomington Heights", 42.0590, -93.6400),
    "Blueste": ("Bluestem", 42.0100, -93.6460),
    "BrDale": ("Briardale", 42.0525, -93.6288),
    "BrkSide": ("Brookside", 42.0300, -93.6300),
    "ClearCr": ("Clear Creek", 42.0360, -93.6700),
    "CollgCr": ("College Creek", 42.0210, -93.6850),
    "Crawfor": ("Crawford", 42.0170, -93.6380),
    "Edwards": ("Edwards", 42.0190, -93.6650),
    "Gilbert": ("Gilbert", 42.0650, -93.6450),
    "Greens": ("Greens", 42.0380, -93.6410),
    "GrnHill": ("Green Hills", 42.0000, -93.6550),
    "IDOTRR": ("Iowa DOT & Rail Road", 42.0220, -93.6150),
    "Landmrk": ("Landmark", 42.0440, -93.6120),
    "MeadowV": ("Meadow Village", 41.9920, -93.6030),
    "Mitchel": ("Mitchell", 41.9900, -93.6000),
    "NAmes": ("North Ames", 42.0450, -93.6170),
    "NoRidge": ("Northridge", 42.0480, -93.6500),
    "NPkVill": ("Northpark Villa", 42.0500, -93.6270),
    "NridgHt": ("Northridge Heights", 42.0600, -93.6560),
    "NWAmes": ("Northwest Ames", 42.0490, -93.6350),
    "OldTown": ("Old Town", 42.0290, -93.6140),
    "SWISU": ("South & West of ISU", 42.0170, -93.6500),
    "Sawyer": ("Sawyer", 42.0340, -93.6770),
    "SawyerW": ("Sawyer West", 42.0340, -93.6850),
    "Somerst": ("Somerset", 42.0520, -93.6440),
    "StoneBr": ("Stone Brook", 42.0600, -93.6370),
    "Timber": ("Timberland", 41.9970, -93.6490),
    "Veenker": ("Veenker", 42.0400, -93.6510),
}


def neighborhood_name(code: str) -> str:
    return NEIGHBORHOODS.get(code, (code, 0, 0))[0]
