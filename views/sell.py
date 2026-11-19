import streamlit as st

from housefinder import config, data, db, ui
from housefinder.i18n import t

valuator = ui.get_valuator()
user = ui.current_user()

st.title("📢 " + t("sell_title"))
st.caption(t("sell_sub"))

if not user:
    ui.need_login("login_to_sell")
    st.stop()

defaults = st.session_state.get("s_saved") or (st.session_state.get("last_valuation") or {}).get("inputs") \
    or ui.typical_house("NAmes")

with st.form("sell_form"):
    st.subheader("1. " + t("sell_step_house"))
    inputs = ui.house_form("sl", valuator.categories, defaults)
    st.subheader("2. " + t("sell_step_listing"))
    title = st.text_input(t("listing_title"), max_chars=120, key="sl_title", placeholder=t("listing_title_ph"))
    description = st.text_area(t("listing_desc"), max_chars=2000, key="sl_desc")
    c1, c2, c3 = st.columns(3)
    price = c1.number_input(t("asking_price") + " ($)", 0, 5_000_000, 0, step=1000, key="sl_price",
                            help=t("asking_price_help"))
    contact = c2.text_input(t("contact"), max_chars=80, key="sl_contact", placeholder="+998 90 123 45 67")
    days = c3.slider(t("listing_days"), 7, 90, 30, key="sl_days")
    image = st.file_uploader(t("photo"), type=["jpg", "jpeg", "png", "webp"], key="sl_image")
    b1, b2 = st.columns(2)
    evaluate = b1.form_submit_button("🔮 " + t("evaluate_button"), width="stretch")
    publish = b2.form_submit_button("📢 " + t("publish_button"), type="primary", width="stretch")

if evaluate or publish:
    st.session_state["s_saved"] = inputs
result = valuator.value(inputs)

with st.container(border=True):
    st.markdown(f"**{t('ai_fair_value')}:** {ui.money(result.price)} &nbsp; "
                f"<span class='hf-muted'>{t('range_80', low=ui.money(result.low), high=ui.money(result.high))}</span>",
                unsafe_allow_html=True)
    quick, fair_hi = result.price * 0.97, result.high
    st.markdown(t("price_strategy", quick=ui.money(round(quick, -3)), fair=ui.money(round(result.price, -3)),
                  top=ui.money(round(fair_hi, -3))))
    if price:
        status = ui.deal_status(price, result.low, result.high)
        diff = (price / result.price - 1) * 100
        st.markdown(ui.badge(f"{ui.STATUS_ICON[status]} {t('deal_' + status)}", ui.STATUS[status])
                    + f" &nbsp; {t('your_price_vs', diff=f'{diff:+.1f}%')}", unsafe_allow_html=True)
        st.caption(t("advice_" + status))

if publish:
    errors = []
    if len(title.strip()) < 5:
        errors.append(t("err_title"))
    if price <= 0:
        errors.append(t("err_price"))
    blob = mime = None
    if image is not None:
        blob = image.getvalue()
        mime = image.type
        if len(blob) > config.MAX_IMAGE_BYTES:
            errors.append(t("err_image"))
    if errors:
        for e in errors:
            st.error(e)
    else:
        lat, lon = data.coordinates(result.inputs["neighborhood"], f"{user['id']}-{title}")
        lid = db.add_listing(user["id"], price, result.inputs, title=title, description=description,
                             lat=lat, lon=lon, contact=contact, image=blob, image_mime=mime, days=days)
        st.success(t("published", id=lid), icon="✅")
        st.balloons()
        st.session_state.pop("s_saved", None)
        c1, c2 = st.columns(2)
        c1.page_link("views/search.py", label=t("go_search"), icon="🔎")
        c2.page_link("views/account.py", label=t("go_my_listings"), icon="👤")
