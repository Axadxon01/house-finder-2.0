"""Price model: training, persistence, valuation, explanations and comparables.

The model is a gradient-boosted tree ensemble (XGBoost) trained on
log(SalePrice). Working in log space makes errors proportional to price and
lets each feature's contribution be read as a percentage effect.

Uncertainty comes from split-conformal calibration: quantiles of out-of-fold
residuals give a prediction interval whose coverage is then checked on a
hold-out set the model never saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from . import config
from .features import CATEGORICAL, MODEL_FEATURES, MONOTONE_CONSTRAINTS, engineer, normalize_inputs

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=2,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    monotone_constraints=MONOTONE_CONSTRAINTS,
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
)
INTERVAL_LEVELS = (0.5, 0.8, 0.9)
LEGACY_FEATURES = ["gr_liv_area", "bedrooms", "year_built", "garage_cars", "lot_area", "overall_qual"]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mape": float(np.mean(np.abs(predicted - actual) / actual) * 100),
        "median_ape": float(np.median(np.abs(predicted - actual) / actual) * 100),
    }


def _categories(market: pd.DataFrame) -> dict[str, list[str]]:
    cats = {}
    for key in CATEGORICAL:
        values = set(market[key].astype(str).unique()) | {"NA"}
        cats[key] = sorted(values)
    return cats


def train(market: pd.DataFrame, n_splits: int = 5) -> tuple[xgb.Booster, dict]:
    """Train on ~85% of sales, calibrate with CV, evaluate on the rest."""
    categories = _categories(market)
    X = engineer(market, categories)
    y = np.log(market["sale_price"].to_numpy())
    idx = np.arange(len(market))
    train_idx, test_idx = train_test_split(
        idx, test_size=config.HOLDOUT_SIZE, random_state=config.RANDOM_STATE
    )

    # Out-of-fold predictions -> honest CV metrics and conformal residuals.
    oof = np.zeros(len(train_idx))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for fit_part, val_part in folds.split(train_idx):
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X.iloc[train_idx[fit_part]], y[train_idx[fit_part]])
        oof[val_part] = model.predict(X.iloc[train_idx[val_part]])
    residuals = y[train_idx] - oof
    quantiles = {
        str(level): [
            float(np.quantile(residuals, (1 - level) / 2)),
            float(np.quantile(residuals, 1 - (1 - level) / 2)),
        ]
        for level in INTERVAL_LEVELS
    }

    final = xgb.XGBRegressor(**PARAMS)
    final.fit(X.iloc[train_idx], y[train_idx])
    test_log = final.predict(X.iloc[test_idx])
    test_actual = np.exp(y[test_idx])
    test_pred = np.exp(test_log)
    coverage = {
        level: float(np.mean((y[test_idx] >= test_log + lo) & (y[test_idx] <= test_log + hi)))
        for level, (lo, hi) in quantiles.items()
    }

    # The original app's model (6 raw features, 200 trees, dollar target), same split.
    legacy_X = market[LEGACY_FEATURES]
    legacy = xgb.XGBRegressor(n_estimators=200, random_state=config.RANDOM_STATE)
    legacy.fit(legacy_X.iloc[train_idx], np.exp(y[train_idx]))
    legacy_pred = legacy.predict(legacy_X.iloc[test_idx])

    booster = final.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(X.iloc[test_idx], enable_categorical=True), pred_contribs=True
    )
    importance = dict(
        zip(MODEL_FEATURES, np.abs(contribs[:, :-1]).mean(axis=0).round(5).tolist())
    )

    meta = {
        "version": 3,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "features": MODEL_FEATURES,
        "categories": categories,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items() if k != "n_jobs"},
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "residual_quantiles": quantiles,
        "metrics": {
            "cv": {**_metrics(np.exp(y[train_idx]), np.exp(oof)),
                   "rmsle": float(np.sqrt(np.mean(residuals**2)))},
            "holdout": _metrics(test_actual, test_pred),
            "legacy_holdout": _metrics(test_actual, legacy_pred),
            "coverage": coverage,
        },
        "importance": importance,
        # Reference predictions: a loaded model must reproduce these exactly.
        "check": {
            "pid": market["pid"].iloc[test_idx[:10]].tolist(),
            "log_pred": [float(v) for v in test_log[:10]],
        },
        "holdout": {
            "pid": market["pid"].iloc[test_idx].tolist(),
            "actual": test_actual.round(0).tolist(),
            "predicted": test_pred.round(0).tolist(),
        },
    }
    return booster, meta


def save(booster: xgb.Booster, meta: dict, model_path: Path = config.MODEL_PATH,
         meta_path: Path = config.METADATA_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


@dataclass
class Valuation:
    price: float
    low: float
    high: float
    level: float
    base_price: float
    contributions: pd.DataFrame
    inputs: dict = field(default_factory=dict)

    @property
    def price_per_sqft(self) -> float:
        return self.price / max(self.inputs.get("gr_liv_area", 1), 1)


class Valuator:
    """Inference wrapper around a trained booster and its metadata."""

    def __init__(self, booster: xgb.Booster, meta: dict):
        self.booster = booster
        self.meta = meta
        self.categories: dict[str, list[str]] = meta["categories"]

    @classmethod
    def load(cls, model_path: Path = config.MODEL_PATH,
             meta_path: Path = config.METADATA_PATH) -> "Valuator":
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(booster, meta)

    def is_consistent(self, market: pd.DataFrame) -> bool:
        """True if this booster reproduces the predictions recorded at training time.

        Guards against XGBoost version drift, which can load a model without error
        but silently produce wrong numbers.
        """
        check = self.meta.get("check")
        if not check:
            return False
        rows = market.set_index("pid").loc[check["pid"]].reset_index()
        return bool(np.allclose(self.predict_log(rows), check["log_pred"], atol=1e-4))

    @classmethod
    def load_or_train(cls, market: pd.DataFrame) -> "Valuator":
        try:
            valuator = cls.load()
            if valuator.is_consistent(market):
                return valuator
        except Exception:
            pass
        booster, meta = train(market)
        try:
            save(booster, meta)
        except OSError:
            pass  # read-only deploy: keep the in-memory model
        return cls(booster, meta)

    # ---- core prediction -------------------------------------------------
    def _matrix(self, inputs: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(engineer(inputs, self.categories), enable_categorical=True)

    def predict_log(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._matrix(inputs))

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(inputs))

    def interval(self, log_pred: np.ndarray, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.meta["residual_quantiles"][str(level)]
        return np.exp(log_pred + lo), np.exp(log_pred + hi)

    def value(self, inputs: dict, level: float = 0.8, adjustment: float = 0.0) -> Valuation:
        """Full valuation of one house. ``adjustment`` is a market change in % (e.g. 25)."""
        house = normalize_inputs(inputs)
        frame = pd.DataFrame([house])
        dmatrix = self._matrix(frame)
        contribs = self.booster.predict(dmatrix, pred_contribs=True)[0]
        factor = 1 + adjustment / 100
        log_pred = float(contribs.sum())
        low, high = self.interval(np.array([log_pred]), level)
        table = pd.DataFrame({"feature": MODEL_FEATURES, "log_effect": contribs[:-1]})
        table["pct"] = (np.exp(table["log_effect"]) - 1) * 100
        table = table.reindex(table["log_effect"].abs().sort_values(ascending=False).index)
        return Valuation(
            price=float(np.exp(log_pred) * factor),
            low=float(low[0] * factor),
            high=float(high[0] * factor),
            level=level,
            base_price=float(np.exp(contribs[-1]) * factor),
            contributions=table.reset_index(drop=True),
            inputs=house,
        )

    def what_if(self, inputs: dict, changes: dict) -> float:
        return self.value({**inputs, **changes}).price

    # ---- comparables -----------------------------------------------------
    @staticmethod
    def comparables(inputs: dict, market: pd.DataFrame, k: int = 6) -> pd.DataFrame:
        """Most similar real sales (sales comparison approach used by appraisers)."""
        house = normalize_inputs(inputs)
        q = pd.Series({
            "total_sf": house["gr_liv_area"] + house["total_bsmt_sf"],
            "overall_qual": house["overall_qual"],
            "year_built": house["year_built"],
            "total_baths": house["full_bath"] + 0.5 * house["half_bath"] + house["bsmt_full_bath"],
            "garage_cars": house["garage_cars"],
            "log_lot": np.log1p(house["lot_area"]),
        })
        m = market.assign(log_lot=np.log1p(market["lot_area"]))
        cols = list(q.index)
        weights = np.array([2.0, 1.5, 1.0, 0.7, 0.5, 0.4])
        z = (m[cols] - q) / m[cols].std()
        distance = np.sqrt(((z**2) * weights).sum(axis=1))
        distance += np.where(m["neighborhood"] == house["neighborhood"], 0.0, 1.5)
        distance += np.where(m["bldg_type"] == house["bldg_type"], 0.0, 0.75)
        result = m.assign(distance=distance).nsmallest(k, "distance").copy()
        result["similarity"] = (100 * np.exp(-result["distance"] / 3)).round(0)
        return result.drop(columns=["log_lot"])

    @staticmethod
    def comparables_estimate(inputs: dict, comps: pd.DataFrame) -> float:
        """Median comparable price per above-ground sq ft x subject size."""
        return float(comps["price_per_sqft"].median() * normalize_inputs(inputs)["gr_liv_area"])
