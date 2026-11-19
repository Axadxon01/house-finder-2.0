import streamlit as st

from housefinder import db, ui
from housefinder.i18n import t

market = ui.get_market()
valuator = ui.get_valuator()
metrics = valuator.meta["metrics"]["holdout"]

st.markdown(
    f"<div class='hf-hero'><h1>🏠 House Finder Pro</h1><p>{ui.esc(t('hero_sub'))}</p></div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("kpi_accuracy"), f"{100 - metrics['median_ape']:.1f}%", help=t("kpi_accuracy_help"))
c2.metric(t("kpi_r2"), f"{metrics['r2']:.3f}", help=t("kpi_r2_help"))
c3.metric(t("kpi_sales"), f"{len(market):,}")
c4.metric(t("kpi_listings"), f"{len(db.active_listings()):,}")

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.subheader("⚡ " + t("quick_title"))
    st.caption(t("quick_sub"))
    hoods = sorted(market["neighborhood"].unique(), key=ui.neighborhood_label)
    with st.form("quick"):
        hood = st.selectbox(t("feat_neighborhood"), hoods, format_func=ui.neighborhood_label,
                            index=hoods.index("NAmes"))
        a, b = st.columns(2)
        living = a.number_input(f"{t('feat_gr_liv_area')} ({ui.unit_label()})", 30, 600 if ui.area_unit() == "m2" else 6000,
                                int(round(ui.to_unit(1500))))
        year = b.number_input(t("feat_year_built"), 1870, 2025, 1985)
        qual = st.slider(t("feat_overall_qual"), 1, 10, 6, help=t("help_overall_qual"))
        go = st.form_submit_button("🔮 " + t("quick_button"), type="primary", width="stretch")
    if go:
        house = ui.typical_house(hood)
        house.update(gr_liv_area=ui.from_unit(living), year_built=year, year_remod=max(year, house["year_remod"]),
                     overall_qual=qual)
        result = valuator.value(house)
        st.markdown(f"<div class='hf-price'>{ui.money(result.price)}</div>"
                    f"<div class='hf-range'>{t('range_80', low=ui.money(result.low), high=ui.money(result.high))}</div>",
                    unsafe_allow_html=True)
        st.caption(t("quick_note"))
        st.session_state["p_saved"] = result.inputs
        ui.reset_form("p")
        st.page_link("views/predict.py", label=t("quick_more"), icon="➡️")

with right:
    st.subheader("✨ " + t("what_you_can"))
    features = [
        ("views/predict.py", "🔮", "card_predict"),
        ("views/market.py", "📈", "card_market"),
        ("views/search.py", "🔎", "card_search"),
        ("views/invest.py", "💰", "card_invest"),
        ("views/sell.py", "📢", "card_sell"),
        ("views/insights.py", "🧠", "card_insights"),
    ]
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for col, (path, icon, key) in zip(cols, features[i:i + 2]):
            with col.container(border=True):
                st.page_link(path, label=t(f"{key}_title"), icon=icon)
                st.caption(t(f"{key}_text"))

st.divider()
st.subheader(t("how_title"))
s1, s2, s3 = st.columns(3)
for col, n in zip((s1, s2, s3), (1, 2, 3)):
    with col:
        st.markdown(f"**{n}. {t(f'how_{n}_title')}**")
        st.caption(t(f"how_{n}_text"))

st.caption("ℹ️ " + t("data_note"))
