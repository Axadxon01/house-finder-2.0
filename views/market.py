import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from housefinder import config, ui
from housefinder.i18n import t

market = ui.get_market()

st.title("📈 " + t("market_title"))
st.caption(t("market_sub"))

years = sorted(market["yr_sold"].unique())
types = sorted(market["bldg_type"].unique())
f1, f2 = st.columns([1, 1.4])
sel_years = f1.pills(t("filter_years"), years, selection_mode="multi", default=years, key="m_years")
sel_types = f2.pills(t("feat_bldg_type"), types, selection_mode="multi", default=types, key="m_types",
                     format_func=lambda v: ui.option_label("bldg_type", v))
df = market[market["yr_sold"].isin(sel_years or years) & market["bldg_type"].isin(sel_types or types)]
if df.empty:
    st.warning(t("no_data"))
    st.stop()

# ---- KPIs ----------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric(t("kpi_sales"), f"{len(df):,}", border=True)
k2.metric(t("median_price"), ui.money(df["sale_price"].median()), border=True)
k3.metric(t("median_ppa"), ui.per_area(df["sale_price"].median(), df["gr_liv_area"].median()), border=True)
by_year = df.groupby("yr_sold")["sale_price"].median()
if len(by_year) >= 2:
    change = (by_year.iloc[-1] / by_year.iloc[-2] - 1) * 100
    k4.metric(f"{t('yoy_change')} ({by_year.index[-2]}→{by_year.index[-1]})", f"{change:+.1f}%", border=True,
              help=t("yoy_help"))
else:
    k4.metric(t("yoy_change"), "—", border=True)

# ---- neighbourhood ranking + map ----------------------------------------
hood = (df.groupby("neighborhood")
          .agg(median_price=("sale_price", "median"), sales=("sale_price", "size"),
               area=("gr_liv_area", "median"), qual=("overall_qual", "median"))
          .reset_index())
hood["name"] = hood["neighborhood"].map(ui.neighborhood_label)
hood["ppa"] = hood["median_price"] / hood["area"].map(ui.to_unit)
hood["lat"] = hood["neighborhood"].map(lambda c: config.NEIGHBORHOODS.get(c, ("", *config.DEFAULT_CENTER))[1])
hood["lon"] = hood["neighborhood"].map(lambda c: config.NEIGHBORHOODS.get(c, ("", *config.DEFAULT_CENTER))[2])
hood = hood.sort_values("median_price")

c1, c2 = st.columns([1, 1.1], gap="large")
with c1:
    fig = go.Figure(go.Bar(
        x=hood["median_price"], y=hood["name"], orientation="h", marker_color=ui.BLUE,
        customdata=hood[["sales", "ppa"]],
        hovertemplate="<b>%{y}</b><br>" + t("median_price") + ": $%{x:,.0f}<br>"
        + t("kpi_sales") + ": %{customdata[0]}<br>" + t("price_per_area") + ": $%{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=t("chart_hood_rank"), height=640, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(tickprefix="$"), bargap=0.25)
    st.plotly_chart(fig, key="hood_rank")
with c2:
    fig = px.scatter_map(
        hood, lat="lat", lon="lon", size="sales", color="median_price", hover_name="name",
        color_continuous_scale=ui.SEQUENTIAL, size_max=38, zoom=11.2,
        center={"lat": config.DEFAULT_CENTER[0], "lon": config.DEFAULT_CENTER[1]},
        hover_data={"lat": False, "lon": False, "median_price": ":$,.0f", "sales": True},
        labels={"median_price": t("median_price"), "sales": t("kpi_sales")},
    )
    fig.update_layout(map_style="open-street-map", height=640, margin=dict(l=0, r=0, t=50, b=0),
                      title=t("chart_hood_map"), coloraxis_colorbar=dict(tickprefix="$", title=""))
    st.plotly_chart(fig, key="hood_map")
    st.caption("📍 " + t("approx_coords"))

# ---- trend + distribution ----------------------------------------------
c1, c2 = st.columns(2, gap="large")
with c1:
    trend = (df.groupby(pd.Grouper(key="sale_date", freq="QS"))["sale_price"]
               .agg(["median", "size"]).reset_index())
    fig = go.Figure(go.Scatter(x=trend["sale_date"], y=trend["median"], mode="lines+markers",
                               line=dict(color=ui.BLUE, width=2), marker=dict(size=8),
                               customdata=trend["size"],
                               hovertemplate="%{x|%Y Q%q}<br>$%{y:,.0f}<br>n=%{customdata}<extra></extra>"))
    fig.update_layout(title=t("chart_trend"), height=360, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(tickprefix="$"))
    st.plotly_chart(fig, key="trend")
with c2:
    fig = go.Figure(go.Histogram(x=df["sale_price"], nbinsx=45, marker_color=ui.BLUE,
                                 hovertemplate="$%{x}<br>%{y}<extra></extra>"))
    fig.add_vline(x=df["sale_price"].median(), line_dash="dash", line_color=ui.ORANGE,
                  annotation_text=t("median_price"))
    fig.update_layout(title=t("chart_distribution"), height=360, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(tickprefix="$"), bargap=0.05)
    st.plotly_chart(fig, key="dist")

c1, c2 = st.columns(2, gap="large")
with c1:
    fig = px.box(df, x="overall_qual", y="sale_price", points=False, color_discrete_sequence=[ui.BLUE],
                 labels={"overall_qual": t("feat_overall_qual"), "sale_price": t("col_price")})
    fig.update_layout(title=t("chart_quality"), height=380, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(tickprefix="$"))
    st.plotly_chart(fig, key="quality_box")
with c2:
    sample = df.sample(min(len(df), 1500), random_state=1).assign(area_u=lambda d: d["gr_liv_area"].map(ui.to_unit))
    fig = px.scatter(sample, x="area_u", y="sale_price", color="overall_qual", opacity=0.75,
                     color_continuous_scale=ui.SEQUENTIAL,
                     labels={"area_u": f"{t('feat_gr_liv_area')} ({ui.unit_label()})",
                             "sale_price": t("col_price"), "overall_qual": t("feat_overall_qual")})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,.6)")))
    fig.update_layout(title=t("chart_area_price"), height=380, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(tickprefix="$"))
    st.plotly_chart(fig, key="area_scatter")

# ---- neighbourhood profile ------------------------------------------------
st.subheader("📍 " + t("hood_profile"))
codes = hood.sort_values("name")["neighborhood"].tolist()
pick = st.selectbox(t("feat_neighborhood"), codes, format_func=ui.neighborhood_label,
                    index=codes.index("NAmes") if "NAmes" in codes else 0, key="m_hood")
sub = df[df["neighborhood"] == pick]
p1, p2, p3, p4 = st.columns(4)
p1.metric(t("kpi_sales"), f"{len(sub):,}")
p2.metric(t("median_price"), ui.money(sub["sale_price"].median()))
p3.metric(t("hood_area"), ui.area(sub["gr_liv_area"].median()))
p4.metric(t("median_year"), f"{int(sub['year_built'].median())}")
table = pd.DataFrame({
    t("col_sold"): sub["sale_date"].dt.strftime("%Y-%m"),
    t("col_price") + " ($)": sub["sale_price"].astype(int),
    t("col_area"): sub["gr_liv_area"].map(ui.to_unit).round(0),
    t("feat_overall_qual"): sub["overall_qual"],
    t("feat_year_built"): sub["year_built"],
    t("feat_bedrooms"): sub["bedrooms"].astype(int),
    t("feat_bldg_type"): sub["bldg_type"].map(lambda v: ui.option_label("bldg_type", v)),
}).sort_values(t("col_sold"), ascending=False)
st.dataframe(table, hide_index=True, height=300, column_config={
    t("col_price") + " ($)": st.column_config.NumberColumn(format="localized"),
    t("col_area"): st.column_config.NumberColumn(format=f"%.0f {ui.unit_label()}"),
})
st.download_button("⬇️ " + t("download_csv"), table.to_csv(index=False), f"ames_{pick}.csv", "text/csv")
