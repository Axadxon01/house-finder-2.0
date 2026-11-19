import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from housefinder import config, db, ui
from housefinder.features import INPUT_KEYS
from housefinder.i18n import t

valuator = ui.get_valuator()
user = ui.current_user()

st.title("🔎 " + t("search_title"))
st.caption(t("search_sub"))

listings = db.active_listings()
if listings.empty:
    st.info(t("no_listings"))
    st.stop()

# AI fair value for every listing (one batched prediction).
log_pred = valuator.predict_log(listings)
low, high = valuator.interval(log_pred, 0.8)
listings["fair"] = np.exp(log_pred)
listings["low"], listings["high"] = low, high
listings["diff_pct"] = (listings["price"] / listings["fair"] - 1) * 100
listings["status"] = [ui.deal_status(p, lo, hi) for p, lo, hi in zip(listings["price"], low, high)]
listings["baths"] = listings["full_bath"] + 0.5 * listings["half_bath"] + listings["bsmt_full_bath"]
favorites = db.favorite_ids(user["id"]) if user else set()

# ---- filters -------------------------------------------------------------
with st.container(border=True):
    pmin, pmax = int(listings["price"].min() // 10000 * 10000), int(np.ceil(listings["price"].max() / 10000) * 10000)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    price_range = c1.slider(t("filter_price"), pmin, pmax, (pmin, pmax), step=5000, format="$%d", key="s_price")
    min_beds = c2.number_input(t("filter_min_beds"), 0, 8, 0, key="s_beds")
    min_baths = c3.number_input(t("filter_min_baths"), 0.0, 6.0, 0.0, step=0.5, format="%.1f", key="s_baths")
    min_area = c4.number_input(f"{t('filter_min_area')} ({ui.unit_label()})", 0, 10000, 0, step=10, key="s_area")
    c1, c2, c3 = st.columns([1.5, 2, 1])
    hoods = sorted(listings["neighborhood"].unique(), key=ui.neighborhood_label)
    sel_hoods = c1.multiselect(t("feat_neighborhood"), hoods, format_func=ui.neighborhood_label, key="s_hoods",
                               placeholder=t("all"))
    deal = c2.pills(t("filter_deal"), ["deal", "fair", "over"], selection_mode="multi", key="s_deal",
                    format_func=lambda s: f"{ui.STATUS_ICON[s]} {t('short_' + s)}")
    sort = c3.selectbox(t("sort_by"), ["best_deal", "newest", "price_asc", "price_desc", "area_desc"],
                        format_func=lambda s: t("sort_" + s), key="s_sort")
    only_fav = st.toggle("❤️ " + t("only_favorites"), key="s_fav", disabled=not user)

mask = (listings["price"].between(*price_range) & (listings["bedrooms"] >= min_beds)
        & (listings["baths"] >= min_baths) & (listings["gr_liv_area"] >= ui.from_unit(min_area)))
if sel_hoods:
    mask &= listings["neighborhood"].isin(sel_hoods)
if deal:
    mask &= listings["status"].isin(deal)
if only_fav:
    mask &= listings["id"].isin(favorites)
results = listings[mask]
order = {"best_deal": ("diff_pct", True), "newest": ("created_at", False), "price_asc": ("price", True),
         "price_desc": ("price", False), "area_desc": ("gr_liv_area", False)}[sort]
results = results.sort_values(order[0], ascending=order[1])

counts = results["status"].value_counts()
st.markdown(
    f"**{t('found_n', n=len(results))}** &nbsp; "
    + " &nbsp; ".join(ui.badge(f"{t('deal_' + s)}: {counts.get(s, 0)}", ui.STATUS[s]) for s in ("deal", "fair", "over")),
    unsafe_allow_html=True,
)
if results.empty:
    st.warning(t("no_matches"))
    st.stop()

# ---- map -------------------------------------------------------------------
with st.expander("🗺️ " + t("map_view"), expanded=True):
    mapdf = results.assign(
        label=[ui.listing_title(r) for _, r in results.iterrows()],
        status_name=results["status"].map(lambda s: t("deal_" + s)),
        price_fmt=results["price"].map(ui.money),
        fair_fmt=results["fair"].map(ui.money),
    )
    fig = px.scatter_map(
        mapdf, lat="lat", lon="lon", color="status_name", hover_name="label",
        color_discrete_map={t("deal_" + s): c for s, c in ui.STATUS.items()},
        hover_data={"lat": False, "lon": False, "status_name": False, "price_fmt": True, "fair_fmt": True},
        labels={"price_fmt": t("asking_price"), "fair_fmt": t("ai_fair_value"), "status_name": ""},
        zoom=11.3, center={"lat": config.DEFAULT_CENTER[0], "lon": config.DEFAULT_CENTER[1]},
    )
    fig.update_traces(marker=dict(size=13))
    fig.update_layout(map_style="open-street-map", height=420, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, key="search_map")

# ---- cards -------------------------------------------------------------------
PER_PAGE = 12
pages = max(1, int(np.ceil(len(results) / PER_PAGE)))
page = st.columns([1, 5])[0].number_input(t("page") + f" (1–{pages})", 1, pages, 1, key="s_page") if pages > 1 else 1
chunk = results.iloc[(page - 1) * PER_PAGE: page * PER_PAGE]


def card(row: pd.Series) -> None:
    lid = int(row["id"])
    with st.container(border=True):
        image = db.listing_image(lid) if row["has_image"] else None
        if image:
            st.image(image[0], width="stretch")
        else:
            st.markdown("<div class='hf-photo'>🏡</div>", unsafe_allow_html=True)
        st.markdown(f"**{ui.esc(ui.listing_title(row))}**")
        st.markdown(
            f"<span class='hf-card-price'>{ui.money(row['price'])}</span> &nbsp;"
            + ui.badge(f"{ui.STATUS_ICON[row['status']]} {t('deal_' + row['status'])}", ui.STATUS[row["status"]]),
            unsafe_allow_html=True,
        )
        st.caption(
            f"🛏 {int(row['bedrooms'])} · 🛁 {row['baths']:g} · 📐 {ui.area(row['gr_liv_area'])} · "
            f"🏗 {int(row['year_built'])} · 🚗 {int(row['garage_cars'])}"
        )
        st.markdown(t("fair_line", fair=ui.money(row["fair"]), diff=f"{row['diff_pct']:+.1f}%"))
        b1, b2 = st.columns(2)
        fav = lid in favorites
        if b1.button("❤️" if fav else "🤍", key=f"fav_{lid}", help=t("favorite"), width="stretch",
                     disabled=not user):
            db.toggle_favorite(user["id"], lid)
            st.rerun()
        if b2.button(f"👋 {int(row['interest_count'])}", key=f"int_{lid}", help=t("interest"), width="stretch"):
            if db.add_interest(lid, ui.viewer_id()):
                st.toast(t("interest_saved"), icon="👋")
            else:
                st.toast(t("interest_already"), icon="ℹ️")
            st.rerun()
        with st.expander(t("details")):
            st.markdown(f"📍 {ui.neighborhood_label(row['neighborhood'])} · "
                        f"{ui.option_label('bldg_type', row['bldg_type'])} · {ui.option_label('house_style', row['house_style'])}")
            st.markdown(t("range_80", low=ui.money(row["low"]), high=ui.money(row["high"])))
            if isinstance(row.get("description"), str) and row["description"]:
                st.write(row["description"])
            if row["is_demo"]:
                st.caption("🧪 " + t("demo_listing"))
            else:
                st.caption(f"👤 {row['seller']} · 🗓 {str(row['created_at'])[:10]}")
                if row.get("contact"):
                    if user:
                        st.markdown(f"📞 {ui.esc(row['contact'])}")
                    else:
                        st.caption("📞 " + t("contact_login"))
            if st.button("🔮 " + t("open_in_predictor"), key=f"val_{lid}", width="stretch"):
                st.session_state["p_saved"] = {k: row[k] for k in INPUT_KEYS}
                ui.reset_form("p")
                st.switch_page("views/predict.py")


for i in range(0, len(chunk), 3):
    cols = st.columns(3)
    for col, (_, row) in zip(cols, chunk.iloc[i:i + 3].iterrows()):
        with col:
            card(row)

if not user:
    st.caption("🔐 " + t("login_for_favorites"))
