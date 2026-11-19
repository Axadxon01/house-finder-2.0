"""House Finder Pro — Streamlit entry point.

Run locally:  streamlit run app.py
"""

import streamlit as st

from housefinder import __version__, ui
from housefinder.i18n import LANGUAGES, t

st.set_page_config(page_title="House Finder Pro", page_icon="🏠", layout="wide",
                   initial_sidebar_state="expanded")

st.session_state.setdefault("lang", "uz")
st.session_state.setdefault("unit", "m2")
ui.inject_css()

with st.spinner(t("loading_model")):
    ui.get_market()
    ui.get_valuator()
    ui.ensure_db()

pages = {
    "": [
        st.Page("views/home.py", title=t("nav_home"), icon="🏠", default=True),
    ],
    t("nav_section_ai"): [
        st.Page("views/predict.py", title=t("nav_predict"), icon="🔮", url_path="predict"),
        st.Page("views/invest.py", title=t("nav_invest"), icon="💰", url_path="invest"),
        st.Page("views/insights.py", title=t("nav_insights"), icon="🧠", url_path="model"),
    ],
    t("nav_section_market"): [
        st.Page("views/market.py", title=t("nav_market"), icon="📈", url_path="market"),
        st.Page("views/search.py", title=t("nav_search"), icon="🔎", url_path="search"),
        st.Page("views/sell.py", title=t("nav_sell"), icon="📢", url_path="sell"),
    ],
    t("nav_section_account"): [
        st.Page("views/account.py", title=t("nav_account"), icon="👤", url_path="account"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.selectbox("🌐 " + t("language"), list(LANGUAGES), key="lang",
                 format_func=lambda code: LANGUAGES[code])
    st.segmented_control("📐 " + t("area_unit"), ["m2", "sqft"], key="unit",
                         format_func=lambda u: "m²" if u == "m2" else "sq ft")
    user = ui.current_user()
    if user:
        st.success(t("signed_in_as", name=user["username"]), icon="👤")
    else:
        st.page_link("views/account.py", label=t("sign_in_cta"), icon="🔐")
    st.caption(f"House Finder Pro v{__version__} · {t('data_note_short')}")

nav.run()
