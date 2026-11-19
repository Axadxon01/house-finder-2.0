import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housefinder import auth, db, finance
from housefinder.data import load_market
from housefinder.features import (CATEGORICAL, INPUT_KEYS, MODEL_FEATURES, default_inputs, engineer,
                                  normalize_inputs)
from housefinder.i18n import T
from housefinder.model import Valuator

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def market():
    return load_market()


@pytest.fixture(scope="session")
def valuator(market):
    return Valuator.load_or_train(market)


def test_market_is_clean(market):
    assert len(market) > 2900
    assert not ((market["gr_liv_area"] > 4000) & (market["sale_price"] < 300000)).any()
    assert market[INPUT_KEYS].notna().all().all()
    assert market["lat"].between(41.9, 42.1).all()


def test_engineer_matches_schema(market, valuator):
    X = engineer(market.head(20), valuator.categories)
    assert list(X.columns) == MODEL_FEATURES
    for key in CATEGORICAL:
        assert isinstance(X[key].dtype, pd.CategoricalDtype)
    assert (X["total_sf"] == market.head(20)["gr_liv_area"] + market.head(20)["total_bsmt_sf"]).all()


def test_saved_model_is_consistent(market, valuator):
    assert valuator.is_consistent(market)
    broken = Valuator(valuator.booster, {**valuator.meta, "check": {**valuator.meta["check"],
                                                                     "log_pred": [0.0] * 10}})
    assert not broken.is_consistent(market)


def test_model_quality(valuator):
    m = valuator.meta["metrics"]
    assert m["holdout"]["r2"] > 0.92
    assert m["holdout"]["mape"] < 9
    assert m["holdout"]["mae"] < m["legacy_holdout"]["mae"]  # beats the original app's model
    assert 0.72 < m["coverage"]["0.8"] < 0.88  # calibrated interval


def test_valuation_is_consistent(market, valuator):
    house = default_inputs(market, "NridgHt")
    result = valuator.value(house)
    assert result.low < result.price < result.high
    # SHAP contributions reconstruct the prediction exactly.
    total = np.log(result.base_price) + result.contributions["log_effect"].sum()
    assert np.isclose(np.exp(total), result.price, rtol=1e-4)
    # Market adjustment scales everything.
    adjusted = valuator.value(house, adjustment=50)
    assert np.isclose(adjusted.price, result.price * 1.5, rtol=1e-6)


def test_model_responds_sensibly(market, valuator):
    house = default_inputs(market, "NAmes")
    base = valuator.value(house).price
    assert valuator.what_if(house, {"overall_qual": house["overall_qual"] + 2}) > base
    assert valuator.what_if(house, {"gr_liv_area": house["gr_liv_area"] * 1.5}) > base
    expensive = valuator.what_if(house, {"neighborhood": "StoneBr"})
    cheap = valuator.what_if(house, {"neighborhood": "MeadowV"})
    assert expensive > cheap


@pytest.mark.parametrize("hood", ["NAmes", "OldTown", "CollgCr", "NridgHt", "Edwards"])
def test_upgrades_never_lower_value(market, valuator, hood):
    house = default_inputs(market, hood)
    base = valuator.value(house).price
    upgrades = [
        {"overall_qual": min(house["overall_qual"] + 1, 10)}, {"kitchen_qual": 5}, {"exter_qual": 5},
        {"bsmt_fin_sf": house["total_bsmt_sf"]}, {"full_bath": house["full_bath"] + 1},
        {"garage_cars": house["garage_cars"] + 1}, {"fireplaces": house["fireplaces"] + 1},
        {"heating_qc": 5}, {"gr_liv_area": house["gr_liv_area"] * 1.2},
        {"porch_deck_sf": house["porch_deck_sf"] + 300},
    ]
    for change in upgrades:
        assert valuator.what_if(house, change) >= base - 1e-6, change


def test_comparables(market, valuator):
    house = default_inputs(market, "CollgCr")
    comps = valuator.comparables(house, market, k=6)
    assert len(comps) == 6
    assert (comps["neighborhood"] == "CollgCr").all()
    estimate = valuator.comparables_estimate(house, comps)
    assert 0.7 < estimate / valuator.value(house).price < 1.3


def test_normalize_inputs_fixes_inconsistencies(market):
    house = default_inputs(market)
    house.update(year_built=2005, year_remod=1990, total_bsmt_sf=0, bsmt_fin_sf=500, garage_cars=0,
                 gr_liv_area=10)
    fixed = normalize_inputs(house)
    assert fixed["year_remod"] == 2005
    assert fixed["bsmt_fin_sf"] == 0 and fixed["bsmt_qual"] == 0
    assert fixed["garage_type"] == "NA"
    assert fixed["gr_liv_area"] >= 300


def test_finance():
    # 200k at 6% for 30 years -> 1199.10 / month (standard amortisation table).
    assert finance.monthly_payment(200_000, 6, 30) == pytest.approx(1199.10, abs=0.01)
    assert finance.monthly_payment(120_000, 0, 10) == pytest.approx(1000)
    schedule = finance.amortization(200_000, 6, 30)
    assert schedule["principal"].sum() == pytest.approx(200_000, rel=1e-6)
    assert schedule["balance"].iloc[-1] == pytest.approx(0, abs=1e-6)
    r = finance.rental_metrics(200_000, 1_600, 35, 5, 20, 6, 30)
    assert r["gross_yield"] == pytest.approx(9.6)
    assert r["cap_rate"] == pytest.approx(1_600 * 12 * 0.95 * 0.65 / 200_000 * 100)
    proj = finance.projection(100_000, {"a": 10}, 2)
    assert proj["a"].iloc[-1] == pytest.approx(121_000)
    assert finance.roi(15_000, 10_000) == pytest.approx(50)


def test_password_hashing():
    stored = auth.hash_password("secret123")
    assert stored != auth.hash_password("secret123")  # salted
    assert auth.verify_password("secret123", stored)
    assert not auth.verify_password("wrong", stored)
    assert not auth.verify_password("x", "garbage")


def test_database_flow(tmp_path, market, valuator):
    path = tmp_path / "test.db"
    db.init(path)
    assert db.create_user("ali", "secret1", "ali@example.com", path=path) is None
    assert db.create_user("ALI", "secret1", path=path) == "err_user_exists"
    assert db.create_user("a", "secret1", path=path) == "err_username"
    assert db.create_user("vali", "123", path=path) == "err_password"
    user = db.authenticate("ali", "secret1", path=path)
    assert user and db.authenticate("ali", "nope", path=path) is None

    seeded = db.seed_demo_listings(market, valuator.meta["holdout"]["pid"], n=10, path=path)
    assert seeded == 10
    assert db.seed_demo_listings(market, valuator.meta["holdout"]["pid"], n=10, path=path) == 0
    house = default_inputs(market)
    lid = db.add_listing(user["id"], 199_000, house, title="Test uy", image=b"\x89PNG", image_mime="image/png",
                         path=path)
    listings = db.active_listings(path)
    assert len(listings) == 11 and set(INPUT_KEYS) <= set(listings.columns)
    assert db.listing_image(lid, path) == (b"\x89PNG", "image/png")
    assert db.toggle_favorite(user["id"], lid, path) is True
    assert db.favorite_ids(user["id"], path) == {lid}
    assert db.toggle_favorite(user["id"], lid, path) is False
    assert db.add_interest(lid, "s:1", path) and not db.add_interest(lid, "s:1", path)
    assert db.extend_listing(lid, user["id"], 10, path)
    assert not db.delete_listing(lid, user["id"] + 1, path)  # only the owner can delete
    assert db.delete_listing(lid, user["id"], path)
    assert len(db.user_listings(user["id"], path)) == 0


def test_every_translation_key_exists():
    used = set()
    for file in list((ROOT / "views").glob("*.py")) + [ROOT / "app.py"] + list((ROOT / "housefinder").glob("*.py")):
        used |= set(re.findall(r"""\bt\(\s*["']([a-z0-9_]+)["']\s*[,)]""", file.read_text(encoding="utf-8")))
    used |= {f"feat_{k}" for k in MODEL_FEATURES}
    missing = sorted(k for k in used if k not in T)
    assert not missing, missing
    for key, values in T.items():
        assert len(values) == 3 and all(values), key
