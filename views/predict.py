import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from housefinder import ui
from housefinder.i18n import t

market = ui.get_market()
valuator = ui.get_valuator()
categories = valuator.categories

st.title("🔮 " + t("predict_title"))
st.caption(t("predict_sub"))

saved = st.session_state.get("p_saved") or ui.typical_house("NAmes")

form_col, result_col = st.columns([1.05, 1], gap="large")

with form_col:
    with st.container(border=True):
        preset_cols = st.columns([2, 1], vertical_alignment="bottom")
        hoods = sorted(market["neighborhood"].unique(), key=ui.neighborhood_label)
        preset = preset_cols[0].selectbox(t("preset_label"), hoods, format_func=ui.neighborhood_label,
                                          index=hoods.index(saved["neighborhood"]) if saved["neighborhood"] in hoods else 0,
                                          key="p_preset")
        if preset_cols[1].button(t("preset_button"), width="stretch"):
            st.session_state["p_saved"] = ui.typical_house(preset)
            ui.reset_form("p", st.session_state["p_saved"])
            st.rerun()

    with st.form("predict_form", border=True):
        inputs = ui.house_form("p", categories, saved)
        adjustment = st.number_input(t("market_adjust"), -50, 200, 0, step=5, key="p_adjust",
                                     help=t("market_adjust_help"))
        level = st.select_slider(t("confidence_level"), [0.5, 0.8, 0.9], value=0.8, key="p_level",
                                 format_func=lambda v: f"{int(v * 100)}%")
        submitted = st.form_submit_button("🔮 " + t("predict_button"), type="primary", width="stretch")

if submitted:
    st.session_state["p_saved"] = inputs
current = st.session_state.get("p_saved") or inputs
result = valuator.value(current, level=st.session_state.get("p_level", 0.8),
                        adjustment=st.session_state.get("p_adjust", 0))
st.session_state["last_valuation"] = {"inputs": result.inputs, "price": result.price, "low": result.low,
                                      "high": result.high, "adjustment": st.session_state.get("p_adjust", 0)}
comps = valuator.comparables(result.inputs, market, k=6)
comps_value = valuator.comparables_estimate(result.inputs, comps) * (1 + st.session_state.get("p_adjust", 0) / 100)

with result_col:
    with st.container(border=True):
        st.markdown(f"**{t('estimated_value')}**")
        st.markdown(
            f"<div class='hf-price'>{ui.money(result.price)}</div>"
            f"<div class='hf-range'>{t('range_level', level=int(result.level * 100), low=ui.money(result.low), high=ui.money(result.high))}</div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric(t("price_per_area"), ui.per_area(result.price, result.inputs["gr_liv_area"]))
        diff = (comps_value - result.price) / result.price * 100
        m2.metric(t("comps_method"), ui.money(comps_value), f"{diff:+.1f}% {t('vs_ai')}", delta_color="off",
                  help=t("comps_method_help"))
        spread = (result.high - result.low) / result.price * 100
        confidence = t("conf_high") if spread < 25 else t("conf_mid") if spread < 40 else t("conf_low")
        hood_prices = market.loc[market["neighborhood"] == result.inputs["neighborhood"], "sale_price"]
        pct_rank = (hood_prices < result.price).mean() * 100 if len(hood_prices) else np.nan
        st.markdown(t("summary_line", conf=confidence, hood=ui.neighborhood_label(result.inputs["neighborhood"]),
                      pct=f"{pct_rank:.0f}"))
        if st.session_state.get("p_adjust", 0):
            st.caption(t("adjusted_note", pct=st.session_state["p_adjust"]))

    # Waterfall: how the typical house becomes this house.
    contrib = result.contributions
    top = contrib.head(8)
    rest = contrib.iloc[8:]["log_effect"].sum()
    labels, deltas = [t("base_value")], []
    running = result.base_price
    for _, row in top.iterrows():
        new = running * np.exp(row["log_effect"])
        deltas.append(new - running)
        labels.append(t(f"feat_{row['feature']}"))
        running = new
    deltas.append(running * np.exp(rest) - running)
    labels.append(t("other_factors"))
    levels = np.cumsum([result.base_price] + deltas)
    x_lo, x_hi = levels.min() * 0.88, levels.max() * 1.08
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=labels + [t("estimated_value")],
        x=[result.base_price] + deltas + [0],
        measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
        increasing={"marker": {"color": ui.UP}},
        decreasing={"marker": {"color": ui.DOWN}},
        totals={"marker": {"color": ui.GRAY}},
        connector={"line": {"width": 1, "color": "rgba(128,128,128,.4)"}},
        text=[ui.money_short(result.base_price)] + [("+" if d >= 0 else "−") + ui.money_short(abs(d)) for d in deltas]
        + [ui.money_short(result.price)],
        textposition=["inside"] + ["outside"] * len(deltas) + ["inside"],
        insidetextfont=dict(color="#ffffff"),
        cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(height=430, margin=dict(l=10, r=50, t=40, b=10), showlegend=False,
                      title=dict(text=t("waterfall_title"), font=dict(size=15)),
                      yaxis=dict(autorange="reversed"),
                      xaxis=dict(tickprefix="$", showgrid=True, range=[x_lo, x_hi]))
    st.plotly_chart(fig, key="waterfall")

tab_factors, tab_comps, tab_hood, tab_report = st.tabs(
    ["📊 " + t("tab_factors"), "🏘️ " + t("tab_comps"), "📍 " + t("tab_hood"), "📄 " + t("tab_report")]
)

with tab_factors:
    st.caption(t("factors_help"))
    rows = []
    matrix_values = {**result.inputs,
                     "total_sf": result.inputs["gr_liv_area"] + result.inputs["total_bsmt_sf"],
                     "total_baths": result.inputs["full_bath"] + 0.5 * result.inputs["half_bath"] + result.inputs["bsmt_full_bath"],
                     "qual_sf": result.inputs["overall_qual"] * (result.inputs["gr_liv_area"] + result.inputs["total_bsmt_sf"]),
                     "house_age_at_remod": max(result.inputs["year_remod"] - result.inputs["year_built"], 0)}
    for _, row in contrib.iterrows():
        key = row["feature"]
        value = matrix_values[key]
        shown = f"{value:g}" if key in ("total_baths", "house_age_at_remod") else ui.feature_value(key, value)
        rows.append({t("col_factor"): t(f"feat_{key}"), t("col_value"): shown,
                     t("col_effect"): row["pct"]})
    table = pd.DataFrame(rows)
    st.dataframe(
        table, hide_index=True, height=420,
        column_config={t("col_effect"): st.column_config.NumberColumn(format="%+.1f%%")},
    )

with tab_comps:
    st.caption(t("comps_help"))
    show = pd.DataFrame({
        t("col_hood"): comps["neighborhood"].map(ui.neighborhood_label),
        t("col_sold"): comps["sale_date"].dt.strftime("%Y-%m"),
        t("col_price") + " ($)": comps["sale_price"].astype(int),
        t("col_area"): comps["gr_liv_area"].map(ui.to_unit).round(0),
        t("col_ppa") + " ($)": (comps["sale_price"] / comps["gr_liv_area"].map(ui.to_unit)).round(0).astype(int),
        t("feat_overall_qual"): comps["overall_qual"],
        t("feat_year_built"): comps["year_built"],
        t("feat_bedrooms"): comps["bedrooms"].astype(int),
        t("col_similarity"): comps["similarity"],
    })
    st.dataframe(show, hide_index=True, column_config={
        t("col_price") + " ($)": st.column_config.NumberColumn(format="localized"),
        t("col_area"): st.column_config.NumberColumn(format=f"%.0f {ui.unit_label()}"),
        t("col_ppa") + " ($)": st.column_config.NumberColumn(format="localized"),
        t("col_similarity"): st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    })
    st.info(t("comps_estimate_line", value=ui.money(comps_value), ppa=ui.per_area(
        comps["price_per_sqft"].median() * result.inputs["gr_liv_area"], result.inputs["gr_liv_area"])), icon="🧮")

with tab_hood:
    hood = result.inputs["neighborhood"]
    hood_sales = market[market["neighborhood"] == hood]
    st.caption(t("hood_compare_help", hood=ui.neighborhood_label(hood), n=len(hood_sales)))
    fig = go.Figure()
    fig.add_histogram(x=hood_sales["sale_price"], nbinsx=25, marker_color=ui.BLUE, opacity=0.85,
                      name=ui.neighborhood_label(hood), hovertemplate="%{x}<br>%{y}<extra></extra>")
    fig.add_vline(x=result.price, line_width=3, line_color=ui.ORANGE,
                  annotation_text=t("this_house"), annotation_position="top")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis=dict(tickprefix="$", title=t("col_price")), yaxis=dict(title=t("col_count")),
                      bargap=0.06)
    st.plotly_chart(fig, key="hood_hist")
    h1, h2, h3 = st.columns(3)
    h1.metric(t("hood_median"), ui.money(hood_sales["sale_price"].median()))
    h2.metric(t("hood_ppa"), ui.per_area(hood_sales["sale_price"].median(), hood_sales["gr_liv_area"].median()))
    h3.metric(t("hood_area"), ui.area(hood_sales["gr_liv_area"].median()))

with tab_report:
    report = {
        "estimated_value": round(result.price),
        "interval": {"level": result.level, "low": round(result.low), "high": round(result.high)},
        "comparables_estimate": round(comps_value),
        "market_adjustment_pct": st.session_state.get("p_adjust", 0),
        "house": result.inputs,
        "top_factors_pct": {r["feature"]: round(r["pct"], 2) for _, r in contrib.head(10).iterrows()},
        "model": {"trained_at": valuator.meta["trained_at"],
                  "holdout_r2": round(valuator.meta["metrics"]["holdout"]["r2"], 4),
                  "holdout_mape_pct": round(valuator.meta["metrics"]["holdout"]["mape"], 2)},
    }
    st.json(report, expanded=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ JSON", json.dumps(report, indent=2), "valuation.json", "application/json",
                       width="stretch")
    csv = pd.DataFrame([{**result.inputs, "estimated_value": round(result.price),
                         "low": round(result.low), "high": round(result.high)}]).to_csv(index=False)
    d2.download_button("⬇️ CSV", csv, "valuation.csv", "text/csv", width="stretch")

st.divider()
n1, n2, n3 = st.columns(3)
n1.page_link("views/invest.py", label=t("next_invest"), icon="💰")
n2.page_link("views/sell.py", label=t("next_sell"), icon="📢")
n3.page_link("views/insights.py", label=t("next_model"), icon="🧠")
