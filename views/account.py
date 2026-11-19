import numpy as np
import pandas as pd
import streamlit as st

from housefinder import db, ui
from housefinder.i18n import t

valuator = ui.get_valuator()
user = ui.current_user()

st.title("👤 " + t("account_title"))

if not user:
    tab_login, tab_register = st.tabs(["🔐 " + t("login"), "📝 " + t("register")])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input(t("username"), key="a_user")
            password = st.text_input(t("password"), type="password", key="a_pass")
            if st.form_submit_button(t("login"), type="primary", width="stretch"):
                found = db.authenticate(username, password)
                if found:
                    st.session_state.user = found
                    st.rerun()
                else:
                    st.error(t("err_login"))
    with tab_register:
        with st.form("register_form"):
            username = st.text_input(t("username"), key="r_user", help=t("username_help"))
            email = st.text_input(t("email_optional"), key="r_email")
            password = st.text_input(t("password"), type="password", key="r_pass", help=t("password_help"))
            password2 = st.text_input(t("password_repeat"), type="password", key="r_pass2")
            if st.form_submit_button(t("register"), type="primary", width="stretch"):
                if password != password2:
                    st.error(t("err_password_match"))
                else:
                    error = db.create_user(username, password, email)
                    if error:
                        st.error(t(error))
                    else:
                        st.session_state.user = db.authenticate(username, password)
                        st.success(t("registered"))
                        st.rerun()
    st.caption("🔒 " + t("security_note"))
    st.stop()

mine = db.user_listings(user["id"])
favorites = db.favorite_ids(user["id"])
head, logout = st.columns([4, 1], vertical_alignment="center")
head.markdown(f"### {t('hello', name=ui.esc(user['username']))}")
if logout.button("🚪 " + t("logout"), width="stretch"):
    st.session_state.pop("user", None)
    st.rerun()

m1, m2, m3 = st.columns(3)
m1.metric(t("my_listings"), len(mine), border=True)
m2.metric(t("favorites"), len(favorites), border=True)
m3.metric(t("interest_received"), int(mine["interest_count"].sum()) if not mine.empty else 0, border=True)

tab_mine, tab_fav = st.tabs(["📢 " + t("my_listings"), "❤️ " + t("favorites")])


def with_valuation(frame: pd.DataFrame) -> pd.DataFrame:
    log_pred = valuator.predict_log(frame)
    frame = frame.copy()
    frame["fair"] = np.exp(log_pred)
    frame["diff_pct"] = (frame["price"] / frame["fair"] - 1) * 100
    return frame


with tab_mine:
    if mine.empty:
        st.info(t("no_my_listings"))
        st.page_link("views/sell.py", label=t("nav_sell"), icon="📢")
    else:
        mine = with_valuation(mine)
        now = pd.Timestamp.now()
        for _, row in mine.iterrows():
            lid = int(row["id"])
            expires = pd.Timestamp(row["expires_at"]) if row["expires_at"] else None
            active = expires is None or expires > now
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="center")
                diff = f"{row['diff_pct']:+.1f}%"
                c1.markdown(f"**{ui.esc(ui.listing_title(row))}**  \n"
                            f"{ui.money(row['price'])} · {t('fair_line', fair=ui.money(row['fair']), diff=diff)}")
                status = (t("active_until", date=str(row["expires_at"])[:10]) if active
                          else t("expired"))
                c2.markdown(f"{'🟢' if active else '⚪'} {status}  \n👋 {int(row['interest_count'])} · ❤️ {int(row['favorite_count'])}")
                b1, b2 = c3.columns(2)
                if b1.button("⏳ +30", key=f"ext_{lid}", help=t("extend"), width="stretch"):
                    db.extend_listing(lid, user["id"], 30)
                    st.toast(t("extended"), icon="⏳")
                    st.rerun()
                if b2.button("🗑️", key=f"del_{lid}", help=t("delete"), width="stretch"):
                    st.session_state[f"confirm_{lid}"] = True
                if st.session_state.get(f"confirm_{lid}"):
                    st.warning(t("confirm_delete"))
                    y, n = st.columns(2)
                    if y.button(t("yes_delete"), key=f"yes_{lid}", type="primary"):
                        db.delete_listing(lid, user["id"])
                        st.session_state.pop(f"confirm_{lid}", None)
                        st.rerun()
                    if n.button(t("cancel"), key=f"no_{lid}"):
                        st.session_state.pop(f"confirm_{lid}", None)
                        st.rerun()

with tab_fav:
    listings = db.active_listings()
    fav = listings[listings["id"].isin(favorites)] if not listings.empty else listings
    if fav.empty:
        st.info(t("no_favorites"))
        st.page_link("views/search.py", label=t("nav_search"), icon="🔎")
    else:
        fav = with_valuation(fav)
        for _, row in fav.iterrows():
            lid = int(row["id"])
            with st.container(border=True):
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                diff = f"{row['diff_pct']:+.1f}%"
                c1.markdown(f"**{ui.esc(ui.listing_title(row))}** — {ui.money(row['price'])}  \n"
                            f"{t('fair_line', fair=ui.money(row['fair']), diff=diff)}")
                if c2.button("💔", key=f"unfav_{lid}", help=t("remove_favorite"), width="stretch"):
                    db.toggle_favorite(user["id"], lid)
                    st.rerun()
