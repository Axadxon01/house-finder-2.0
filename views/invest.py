import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from housefinder import finance, ui
from housefinder.i18n import t

market = ui.get_market()
valuator = ui.get_valuator()

st.title("💰 " + t("invest_title"))
st.caption(t("invest_sub"))

last = st.session_state.get("last_valuation")
house = last["inputs"] if last else ui.typical_house("NAmes")
default_price = int(round((last["price"] if last else market["sale_price"].median()), -3))
if last:
    st.success(t("using_last_valuation", price=ui.money(last["price"]),
                 hood=ui.neighborhood_label(house["neighborhood"])), icon="🔮")
else:
    st.info(t("no_valuation_yet"), icon="ℹ️")

price = st.number_input(t("house_price") + " ($)", 10_000, 5_000_000, default_price, step=1000, key="i_price")

tab_mort, tab_rent, tab_proj, tab_reno = st.tabs(
    ["🏦 " + t("tab_mortgage"), "🏘️ " + t("tab_rental"), "📈 " + t("tab_projection"), "🛠️ " + t("tab_renovation")]
)

with tab_mort:
    c1, c2, c3 = st.columns(3)
    down = c1.slider(t("down_payment"), 0, 100, 20, format="%d%%", key="i_down")
    rate = c2.number_input(t("interest_rate"), 0.0, 30.0, 6.5, step=0.1, format="%.2f", key="i_rate")
    years = c3.select_slider(t("loan_years"), [5, 10, 15, 20, 25, 30], value=30, key="i_years")
    loan = price * (1 - down / 100)
    payment = finance.monthly_payment(loan, rate, years)
    schedule = finance.amortization(loan, rate, years)
    total_interest = schedule["interest"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("monthly_payment"), ui.money(payment), border=True)
    m2.metric(t("loan_amount"), ui.money(loan), border=True)
    m3.metric(t("total_interest"), ui.money(total_interest), border=True)
    m4.metric(t("total_cost"), ui.money(price + total_interest), border=True)
    if loan > 0:
        fig = go.Figure()
        fig.add_bar(x=schedule["year"], y=schedule["principal"], name=t("principal"), marker_color=ui.BLUE,
                    hovertemplate="%{x}: $%{y:,.0f}<extra>" + t("principal") + "</extra>")
        fig.add_bar(x=schedule["year"], y=schedule["interest"], name=t("interest_part"), marker_color=ui.ORANGE,
                    hovertemplate="%{x}: $%{y:,.0f}<extra>" + t("interest_part") + "</extra>")
        fig.update_layout(barmode="stack", title=t("chart_payments"), height=360, bargap=0.2,
                          margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(tickprefix="$"),
                          xaxis=dict(title=t("year")), legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, key="amort")
        st.caption(t("payment_note"))

with tab_rent:
    c1, c2, c3 = st.columns(3)
    rent = c1.number_input(t("monthly_rent") + " ($)", 0, 50_000, int(round(price * 0.007, -1)), step=50,
                           key="i_rent", help=t("monthly_rent_help"))
    expenses = c2.slider(t("expenses"), 0, 70, 35, format="%d%%", key="i_exp", help=t("expenses_help"))
    vacancy = c3.slider(t("vacancy"), 0, 30, 5, format="%d%%", key="i_vac")
    r = finance.rental_metrics(price, rent, expenses, vacancy, st.session_state.get("i_down", 20),
                               st.session_state.get("i_rate", 6.5), st.session_state.get("i_years", 30))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("gross_yield"), f"{r['gross_yield']:.2f}%", border=True)
    m2.metric(t("cap_rate"), f"{r['cap_rate']:.2f}%", border=True, help=t("cap_rate_help"))
    m3.metric(t("monthly_cash_flow"), ui.signed_money(r["cash_flow"] / 12), border=True)
    m4.metric(t("cash_on_cash"), f"{r['cash_on_cash']:.1f}%", border=True, help=t("cash_on_cash_help"))
    verdict = "rent_good" if r["cash_flow"] > 0 and r["cap_rate"] >= 5 else "rent_ok" if r["cash_flow"] > 0 else "rent_bad"
    (st.success if verdict == "rent_good" else st.info if verdict == "rent_ok" else st.warning)(t(verdict))
    st.caption(t("rent_note"))

with tab_proj:
    c1, c2, c3, c4 = st.columns(4)
    pess = c1.number_input(t("scenario_pess"), -10.0, 20.0, 1.0, step=0.5, key="i_pess")
    base = c2.number_input(t("scenario_base"), -10.0, 20.0, 3.5, step=0.5, key="i_base")
    opt = c3.number_input(t("scenario_opt"), -10.0, 20.0, 6.0, step=0.5, key="i_opt")
    horizon = c4.slider(t("horizon"), 1, 30, 10, key="i_horizon")
    names = {t("scenario_pess"): pess, t("scenario_base"): base, t("scenario_opt"): opt}
    proj = finance.projection(price, names, horizon)
    fig = go.Figure()
    for name, color in zip(names, ["#8f8d85", ui.BLUE, "#1baf7a"]):
        fig.add_scatter(x=proj["year"], y=proj[name], name=name, mode="lines", line=dict(color=color, width=2),
                        hovertemplate=t("year") + " %{x}: $%{y:,.0f}<extra>" + name + "</extra>")
    fig.update_layout(title=t("chart_projection"), height=380, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(tickprefix="$"), xaxis=dict(title=t("year")), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, key="proj")
    loan = price * (1 - st.session_state.get("i_down", 20) / 100)
    sched = finance.amortization(loan, st.session_state.get("i_rate", 6.5), st.session_state.get("i_years", 30))
    balance = float(sched.loc[sched["year"] == min(horizon, len(sched)), "balance"].iloc[0]) if horizon <= len(sched) else 0.0
    end_value = proj[t("scenario_base")].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric(t("value_in_years", n=horizon), ui.money(end_value), f"{(end_value / price - 1) * 100:+.0f}%", border=True)
    m2.metric(t("loan_balance"), ui.money(balance), border=True)
    m3.metric(t("your_equity"), ui.money(end_value - balance), border=True)
    st.caption(t("projection_note"))

with tab_reno:
    st.caption(t("reno_help"))
    adjust = last.get("adjustment", 0) if last else 0
    base_value = valuator.value(house, adjustment=adjust).price
    unfinished = max(house["total_bsmt_sf"] - house["bsmt_fin_sf"], 0)
    upgrades = [
        ("reno_quality", {"overall_qual": min(house["overall_qual"] + 1, 10)}, 30000, house["overall_qual"] < 10),
        ("reno_kitchen", {"kitchen_qual": 5}, 25000, house["kitchen_qual"] < 5),
        ("reno_basement", {"bsmt_fin_sf": house["total_bsmt_sf"]}, int(round(unfinished * 30, -3)), unfinished > 50),
        ("reno_bath", {"full_bath": house["full_bath"] + 1, "tot_rooms": house["tot_rooms"] + 1}, 18000, house["full_bath"] < 4),
        ("reno_garage", {"garage_cars": house["garage_cars"] + 1}, 20000, house["garage_cars"] < 4),
        ("reno_fireplace", {"fireplaces": house["fireplaces"] + 1}, 6000, house["fireplaces"] < 3),
        ("reno_exterior", {"exter_qual": min(house["exter_qual"] + 1, 5)}, 18000, house["exter_qual"] < 5),
        ("reno_heating", {"heating_qc": 5}, 9000, house["heating_qc"] < 5),
        ("reno_air", {"central_air": 1}, 7000, house["central_air"] == 0),
        ("reno_porch", {"porch_deck_sf": house["porch_deck_sf"] + 300}, 12000, True),
    ]
    available = [u for u in upgrades if u[3]]
    st.markdown(f"**{t('reno_base')}: {ui.money(base_value)}**")
    rows = []
    cols = st.columns(2)
    for i, (key, change, cost, _) in enumerate(available):
        with cols[i % 2].container(border=True):
            on = st.checkbox(t(key), value=False, key=f"i_on_{key}")
            cost = st.number_input(t("reno_cost") + " ($)", 0, 500_000, cost, step=500, key=f"i_cost_{key}")
        new_value = valuator.value({**house, **change}, adjustment=adjust).price
        gain = new_value - base_value
        rows.append({"key": key, "on": on, "change": change, "cost": cost, "gain": gain,
                     "roi": finance.roi(gain, cost)})
    table = pd.DataFrame(rows)
    picked = table[table["on"]]
    if not picked.empty:
        combined = {}
        for change in picked["change"]:
            combined.update(change)
        package_value = valuator.value({**house, **combined}, adjustment=adjust).price
        package_cost = picked["cost"].sum()
        package_gain = package_value - base_value
        m1, m2, m3 = st.columns(3)
        m1.metric(t("reno_new_value"), ui.money(package_value), ui.signed_money(package_gain), border=True)
        m2.metric(t("reno_total_cost"), ui.money(package_cost), border=True)
        m3.metric("ROI", f"{finance.roi(package_gain, package_cost):+.0f}%", border=True,
                  help=t("roi_help"))
    else:
        st.info(t("reno_pick"), icon="☑️")
    show = table.sort_values("roi", ascending=False)
    fig = go.Figure()
    fig.add_bar(y=[t(k) for k in show["key"]], x=show["gain"], name=t("reno_gain"), orientation="h",
                marker_color=ui.BLUE, hovertemplate="%{y}<br>" + t("reno_gain") + ": $%{x:,.0f}<extra></extra>")
    fig.add_bar(y=[t(k) for k in show["key"]], x=show["cost"], name=t("reno_cost"), orientation="h",
                marker_color=ui.ORANGE, hovertemplate="%{y}<br>" + t("reno_cost") + ": $%{x:,.0f}<extra></extra>")
    fig.update_layout(barmode="group", title=dict(text=t("chart_reno"), y=0.98), height=130 + 46 * len(show),
                      bargap=0.3, margin=dict(l=10, r=10, t=80, b=10), xaxis=dict(tickprefix="$"),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    st.plotly_chart(fig, key="reno")
    st.caption(t("reno_note"))
