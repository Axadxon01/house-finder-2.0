import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from housefinder import ui
from housefinder.i18n import t

valuator = ui.get_valuator()
meta = valuator.meta
m = meta["metrics"]
new, old = m["holdout"], m["legacy_holdout"]

st.title("🧠 " + t("insights_title"))
st.caption(t("insights_sub", n_train=meta["n_train"], n_test=meta["n_test"]))

k1, k2, k3, k4 = st.columns(4)
k1.metric("R²", f"{new['r2']:.3f}", f"{new['r2'] - old['r2']:+.3f} {t('vs_old')}", border=True,
          help=t("kpi_r2_help"))
k2.metric(t("mae"), ui.money(new["mae"]), f"{(new['mae'] / old['mae'] - 1) * 100:+.0f}% {t('vs_old')}",
          delta_color="inverse", border=True, help=t("mae_help"))
k3.metric(t("mape"), f"{new['mape']:.1f}%", f"{new['mape'] - old['mape']:+.1f} pp {t('vs_old')}",
          delta_color="inverse", border=True, help=t("mape_help"))
k4.metric(f"{t('coverage')} ({t('coverage_target')})", f"{m['coverage']['0.8'] * 100:.1f}%",
          border=True, help=t("coverage_help"))

c1, c2 = st.columns(2, gap="large")
with c1:
    comp = pd.DataFrame({
        t("col_metric"): ["R²", t("mae"), "RMSE", t("mape"), t("median_ape")],
        t("model_old"): [f"{old['r2']:.3f}", ui.money(old["mae"]), ui.money(old["rmse"]), f"{old['mape']:.2f}%",
                         f"{old['median_ape']:.2f}%"],
        t("model_new"): [f"{new['r2']:.3f}", ui.money(new["mae"]), ui.money(new["rmse"]), f"{new['mape']:.2f}%",
                         f"{new['median_ape']:.2f}%"],
    })
    st.subheader(t("old_vs_new"))
    st.dataframe(comp, hide_index=True)
    fig = go.Figure()
    fig.add_bar(x=[t("mae"), "RMSE"], y=[old["mae"], old["rmse"]], name=t("model_old"), marker_color=ui.GRAY,
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + t("model_old") + "</extra>")
    fig.add_bar(x=[t("mae"), "RMSE"], y=[new["mae"], new["rmse"]], name=t("model_new"), marker_color=ui.BLUE,
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + t("model_new") + "</extra>")
    fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3,
                      yaxis=dict(tickprefix="$", title=t("error_lower_better")), legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, key="old_new")

with c2:
    st.subheader(t("pred_vs_actual"))
    h = meta["holdout"]
    actual, pred = np.array(h["actual"]), np.array(h["predicted"])
    err = (pred / actual - 1) * 100
    lim = [0, max(actual.max(), pred.max()) * 1.05]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=ui.GRAY, dash="dash", width=2),
                    name=t("perfect_line"), hoverinfo="skip")
    fig.add_scatter(x=actual, y=pred, mode="markers", name=t("holdout_houses"),
                    marker=dict(size=8, color=ui.BLUE, opacity=0.7, line=dict(width=1, color="rgba(255,255,255,.7)")),
                    customdata=err, hovertemplate=t("actual") + ": $%{x:,.0f}<br>" + t("predicted")
                    + ": $%{y:,.0f}<br>" + t("error") + ": %{customdata:+.1f}%<extra></extra>")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0),
                      xaxis=dict(title=t("actual"), tickprefix="$", range=lim),
                      yaxis=dict(title=t("predicted"), tickprefix="$", range=lim))
    st.plotly_chart(fig, key="pva")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader(t("error_dist"))
    fig = go.Figure(go.Histogram(x=err, nbinsx=40, marker_color=ui.BLUE,
                                 hovertemplate="%{x:.0f}%<br>%{y}<extra></extra>"))
    fig.add_vline(x=0, line_color=ui.GRAY, line_dash="dash")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), bargap=0.05,
                      xaxis=dict(title=t("error_pct"), ticksuffix="%"), yaxis=dict(title=t("col_count")))
    st.plotly_chart(fig, key="err_hist")
    within10 = np.mean(np.abs(err) <= 10) * 100
    within20 = np.mean(np.abs(err) <= 20) * 100
    st.markdown(t("within_line", p10=f"{within10:.0f}", p20=f"{within20:.0f}"))

with c2:
    st.subheader(t("importance_title"))
    imp = pd.Series(meta["importance"]).sort_values(ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=[t(f"feat_{k}") for k in imp.index], orientation="h",
                           marker_color=ui.BLUE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), bargap=0.25,
                      xaxis=dict(title=t("importance_axis")))
    st.plotly_chart(fig, key="importance")

st.subheader("📘 " + t("method_title"))
st.markdown(t("method_text", n_train=meta["n_train"], n_test=meta["n_test"],
              cv_r2=f"{m['cv']['r2']:.3f}", trees=meta["params"]["n_estimators"]))
st.caption(t("trained_at", when=meta["trained_at"][:16].replace("T", " "), xgb=meta["xgboost_version"]))
