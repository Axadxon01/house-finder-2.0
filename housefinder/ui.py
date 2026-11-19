"""Shared Streamlit helpers: cached resources, formatting, the house form."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from . import config, data, db
from .features import CATEGORICAL, INPUT_KEYS, NUMERIC, NUMERIC_BY_KEY, default_inputs
from .i18n import t
from .model import Valuator

# Chart colours (validated reference palette; see .streamlit/config.toml).
BLUE = "#2a78d6"
BLUE_DARK = "#1c5cab"
ORANGE = "#eb6834"
GRAY = "#8f8d85"
UP = "#2a78d6"     # factor raises price
DOWN = "#e34948"   # factor lowers price
STATUS = {"deal": "#0ca30c", "fair": "#8f8d85", "over": "#d03b3b"}
STATUS_ICON = {"deal": "🟢", "fair": "⚪", "over": "🔴"}
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

CSS = """
<style>
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.hf-hero {padding: 1.6rem 1.8rem; border-radius: 1rem; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(42,120,214,.14), rgba(27,175,122,.10));
  border: 1px solid rgba(42,120,214,.25);}
.hf-hero h1 {margin: 0 0 .3rem 0; font-size: 2.3rem;}
.hf-hero p {margin: 0; font-size: 1.08rem; opacity: .85;}
.hf-price {font-size: 2.6rem; font-weight: 750; line-height: 1.1; font-variant-numeric: tabular-nums;}
.hf-range {font-size: .95rem; opacity: .75;}
.hf-badge {display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .8rem;
  font-weight: 600; color: #fff;}
.hf-muted {opacity: .7; font-size: .9rem;}
.hf-card-price {font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums;}
.hf-photo {height: 150px; border-radius: .6rem; display: flex; align-items: center;
  justify-content: center; font-size: 3.2rem; margin-bottom: .5rem;
  background: linear-gradient(135deg, rgba(42,120,214,.18), rgba(235,104,52,.12));}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---- cached resources ---------------------------------------------------
@st.cache_data(show_spinner=False)
def get_market() -> pd.DataFrame:
    return data.load_market()


@st.cache_resource(show_spinner=False)
def get_valuator() -> Valuator:
    return Valuator.load_or_train(get_market())


@st.cache_resource(show_spinner=False)
def _ensure_db(path: str) -> bool:
    db.init(path)
    db.seed_demo_listings(get_market(), get_valuator().meta["holdout"]["pid"], path=path)
    return True


def ensure_db() -> bool:
    """Create tables and demo listings once per database file."""
    return _ensure_db(str(config.DB_PATH))


# ---- session helpers ---------------------------------------------------
def current_user() -> dict | None:
    return st.session_state.get("user")


def viewer_id() -> str:
    user = current_user()
    if user:
        return f"u:{user['id']}"
    if "viewer" not in st.session_state:
        st.session_state.viewer = f"s:{np.random.default_rng().integers(1e12)}"
    return st.session_state.viewer


def area_unit() -> str:
    return st.session_state.get("unit") or "m2"  # segmented control can be cleared


# ---- formatting --------------------------------------------------------
def money(x: float) -> str:
    return f"${x:,.0f}"


def money_short(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def signed_money(x: float) -> str:
    return ("+" if x >= 0 else "−") + money(abs(x))


def to_unit(sqft: float) -> float:
    return sqft / config.SQFT_PER_M2 if area_unit() == "m2" else sqft


def from_unit(value: float) -> float:
    return value * config.SQFT_PER_M2 if area_unit() == "m2" else value


def unit_label() -> str:
    return "m²" if area_unit() == "m2" else "sq ft"


def area(sqft: float) -> str:
    return f"{to_unit(sqft):,.0f} {unit_label()}"


def per_area(price: float, sqft: float) -> str:
    return f"{money(price / max(to_unit(sqft), 1e-9))}/{unit_label()}"


def esc(text) -> str:
    return html.escape(str(text or ""))


def badge(text: str, color: str) -> str:
    return f"<span class='hf-badge' style='background:{color}'>{esc(text)}</span>"


def neighborhood_label(code: str) -> str:
    return config.neighborhood_name(code)


def option_label(key: str, value) -> str:
    """Human label for a categorical / ordinal option."""
    if key in ("exter_qual", "kitchen_qual", "bsmt_qual", "heating_qc"):
        return t(f"q{int(value)}")
    if key == "neighborhood":
        return neighborhood_label(value)
    if key == "central_air":
        return t("yes") if int(value) else t("no")
    label = t(f"opt_{key}_{value}")
    return value if label.startswith("opt_") else label


def feature_value(key: str, value) -> str:
    """Human-readable value of a model feature."""
    if key in CATEGORICAL or key in ("exter_qual", "kitchen_qual", "bsmt_qual", "heating_qc", "central_air"):
        return option_label(key, value)
    spec = NUMERIC_BY_KEY.get(key)
    if (spec and spec.is_area) or key == "total_sf":
        return area(float(value))
    if key == "qual_sf":
        return f"{float(value):,.0f}"
    if key in ("year_built", "year_remod"):
        return f"{int(value)}"
    return f"{float(value):g}"


def deal_status(price: float, low: float, high: float) -> str:
    if price < low:
        return "deal"
    if price > high:
        return "over"
    return "fair"


def listing_title(row) -> str:
    title = row.get("title") if hasattr(row, "get") else None
    if isinstance(title, str) and title.strip():
        return title.strip()
    return t("auto_title", beds=int(row["bedrooms"]), style=option_label("house_style", row["house_style"]),
             hood=neighborhood_label(row["neighborhood"]))


# ---- the house description form ---------------------------------------
_AREA_KEYS = [s.key for s in NUMERIC if s.is_area]
_QUALITY = [1, 2, 3, 4, 5]


def _k(prefix: str, key: str) -> str:
    return f"{prefix}_{key}"


def seed_form(prefix: str, inputs: dict, force: bool = False) -> None:
    """Put a house into the form's widget state (in the current display unit)."""
    for key, value in inputs.items():
        wkey = _k(prefix, key)
        if force or wkey not in st.session_state:
            spec = NUMERIC_BY_KEY.get(key)
            if key == "central_air":
                value = bool(int(value))
            elif spec and spec.is_area:
                value = int(round(to_unit(float(value))))
            elif spec:
                value = int(value)
            st.session_state[wkey] = value
    st.session_state[_k(prefix, "_unit")] = area_unit()


def reset_form(prefix: str, inputs: dict | None = None) -> None:
    """Forget a form's widget values (optionally replacing them with ``inputs``)."""
    for key in [*INPUT_KEYS, "_unit"]:
        st.session_state.pop(_k(prefix, key), None)
    if inputs is not None:
        seed_form(prefix, inputs)


def _sync_unit(prefix: str) -> None:
    """Convert stored area values if the user switched m² <-> sq ft."""
    stored = st.session_state.get(_k(prefix, "_unit"))
    if stored and stored != area_unit():
        factor = config.SQFT_PER_M2 if stored == "m2" else 1 / config.SQFT_PER_M2
        for key in _AREA_KEYS:
            wkey = _k(prefix, key)
            if wkey in st.session_state:
                st.session_state[wkey] = int(round(st.session_state[wkey] * factor))
        st.session_state[_k(prefix, "_unit")] = area_unit()


def house_form(prefix: str, categories: dict[str, list[str]], defaults: dict,
               advanced_open: bool = False) -> dict:
    """Render house inputs (call inside ``st.form``). Returns inputs in model units."""
    seed_form(prefix, defaults)
    _sync_unit(prefix)
    u = unit_label()

    def num(key, label_key=None, col=None):
        spec = NUMERIC_BY_KEY[key]
        lo, hi = spec.min, spec.max
        if spec.is_area:
            lo, hi = int(to_unit(lo)), int(np.ceil(to_unit(hi)))
            lo = 0 if spec.min == 0 else max(lo, 1)
        target = col or st
        return target.number_input(f"{t(label_key or 'feat_' + key)} ({u})" if spec.is_area
                                   else t(label_key or "feat_" + key),
                                   min_value=int(lo), max_value=int(hi), step=1, key=_k(prefix, key),
                                   help=t(f"help_{key}") if not t(f"help_{key}").startswith("help_") else None)

    def select(key, col=None, options=None):
        target = col or st
        opts = options or [c for c in categories[key] if c != "NA" or key == "garage_type"]
        if st.session_state.get(_k(prefix, key)) not in opts:
            st.session_state[_k(prefix, key)] = opts[0]
        return target.selectbox(t(f"feat_{key}"), opts, key=_k(prefix, key),
                                format_func=lambda v: option_label(key, v))

    def quality(key, col=None, with_none=False):
        target = col or st
        opts = ([0] if with_none else []) + _QUALITY
        if st.session_state.get(_k(prefix, key)) not in opts:
            st.session_state[_k(prefix, key)] = 3
        return target.select_slider(t(f"feat_{key}"), opts, key=_k(prefix, key),
                                    format_func=lambda v: option_label(key, v))

    hoods = sorted([c for c in categories["neighborhood"] if c != "NA"], key=neighborhood_label)
    c1, c2 = st.columns(2)
    select("neighborhood", c1, hoods)
    select("bldg_type", c2)
    c1, c2, c3 = st.columns(3)
    num("gr_liv_area", col=c1)
    num("total_bsmt_sf", col=c2)
    num("lot_area", col=c3)
    st.slider(t("feat_overall_qual"), 1, 10, key=_k(prefix, "overall_qual"), help=t("help_overall_qual"))
    c1, c2, c3 = st.columns(3)
    num("year_built", col=c1)
    num("bedrooms", col=c2)
    num("garage_cars", col=c3)
    c1, c2, c3 = st.columns(3)
    num("full_bath", col=c1)
    num("half_bath", col=c2)
    quality("kitchen_qual", c3)

    with st.expander(t("advanced"), expanded=advanced_open):
        c1, c2 = st.columns(2)
        select("house_style", c1)
        select("garage_type", c2)
        c1, c2, c3 = st.columns(3)
        num("year_remod", col=c1)
        num("tot_rooms", col=c2)
        num("fireplaces", col=c3)
        c1, c2, c3 = st.columns(3)
        num("bsmt_fin_sf", col=c1)
        num("bsmt_full_bath", col=c2)
        num("porch_deck_sf", col=c3)
        st.slider(t("feat_overall_cond"), 1, 10, key=_k(prefix, "overall_cond"), help=t("help_overall_cond"))
        c1, c2 = st.columns(2)
        quality("exter_qual", c1)
        quality("bsmt_qual", c2, with_none=True)
        c1, c2 = st.columns(2)
        quality("heating_qc", c1)
        select("foundation", c2)
        c1, c2 = st.columns(2)
        select("ms_zoning", c1)
        select("sale_condition", c2)
        st.toggle(t("feat_central_air"), key=_k(prefix, "central_air"))

    return read_form(prefix)


def read_form(prefix: str) -> dict:
    """Collect current widget values for ``prefix`` in model units."""
    result = {}
    for spec in NUMERIC:
        value = float(st.session_state[_k(prefix, spec.key)])
        result[spec.key] = from_unit(value) if spec.is_area else value
    for key in ("exter_qual", "kitchen_qual", "bsmt_qual", "heating_qc"):
        result[key] = int(st.session_state[_k(prefix, key)])
    result["central_air"] = int(bool(st.session_state[_k(prefix, "central_air")]))
    for key in CATEGORICAL:
        result[key] = st.session_state[_k(prefix, key)]
    return result


def typical_house(neighborhood: str | None = None) -> dict:
    return default_inputs(get_market(), neighborhood)


def need_login(message_key: str = "login_required") -> None:
    st.info(t(message_key), icon="🔐")
    st.page_link("views/account.py", label=t("go_login"), icon="👤")
