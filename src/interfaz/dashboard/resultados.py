"""Pestaña Resultados: gráficos comparativos a partir del CSV de la ejecución por lotes."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.interfaz.dashboard.comun import historico_ego


def renderizar_pestana_resultados(
    df_historico,
    ego_ids: list[int],
    estrategia_keys: list[str],
    estrategia_labels: dict[str, str],
) -> None:
    """UI de resultados: métricas agregadas y gráficos por ego."""
    st.subheader("Comparación de estrategias")

    if df_historico is None or df_historico.empty:
        st.warning(
            "No se encontró resultados/resultados_estrategias.csv. "
            "Ejecuta main.py para generar la comparativa histórica."
        )
        return

    estrategia_filtro = st.multiselect(
        "Estrategias a comparar",
        estrategia_keys,
        default=estrategia_keys,
        format_func=lambda key: estrategia_labels[key],
    )
    df_comparativa = (
        df_historico[df_historico["estrategia"].isin(estrategia_filtro)]
        if estrategia_filtro
        else df_historico.iloc[0:0]
    )

    if df_comparativa.empty:
        st.warning("Selecciona al menos una estrategia.")
        return

    resumen_global = df_comparativa.groupby("estrategia", as_index=False)[
        ["f1", "precision", "sensibilidad", "exactitud"]
    ].mean()
    resumen_global["Estrategia"] = resumen_global["estrategia"].map(estrategia_labels).fillna(resumen_global["estrategia"])
    resumen_global = resumen_global.sort_values("f1", ascending=False)

    r1, r2, r3 = st.columns(3)
    r1.metric("Mejor F1 global", str(resumen_global.iloc[0]["Estrategia"]))
    r2.metric("Egos evaluados", int(df_comparativa["ego_id"].nunique()))
    r3.metric("Estrategias visibles", int(df_comparativa["estrategia"].nunique()))

    plot_df = resumen_global.melt(
        id_vars=["estrategia", "Estrategia"],
        value_vars=["f1", "precision", "sensibilidad"],
        var_name="Metrica",
        value_name="Valor",
    )
    fig = px.bar(
        plot_df,
        x="Estrategia",
        y="Valor",
        color="Metrica",
        barmode="group",
        color_discrete_map={
            "f1": "#2563eb",
            "precision": "#16a34a",
            "sensibilidad": "#f97316",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Media", legend_title="", margin=dict(l=8, r=8, t=18, b=8))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Ranking global")
    st.dataframe(
        resumen_global.rename(
            columns={
                "f1": "F1",
                "precision": "Precisión",
                "sensibilidad": "Sensibilidad",
                "exactitud": "Exactitud",
            }
        )[["Estrategia", "F1", "Precisión", "Sensibilidad", "Exactitud"]],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Detalle por ego")
    ego_resultados = st.selectbox(
        "Ego",
        ego_ids,
        index=0,
        format_func=lambda value: f"Ego {value}",
        key="ego_resultados",
    )
    historial_ego_df = historico_ego(df_comparativa, int(ego_resultados))
    if historial_ego_df is None or historial_ego_df.empty:
        st.caption("Este ego no aparece para las estrategias seleccionadas.")
        return

    historial_ego_df["Estrategia"] = historial_ego_df["estrategia"].map(estrategia_labels).fillna(historial_ego_df["estrategia"])
    historial_ego_df = historial_ego_df.sort_values("f1", ascending=False)
    st.dataframe(
        historial_ego_df.rename(
            columns={
                "f1": "F1",
                "precision": "Precisión",
                "sensibilidad": "Sensibilidad",
                "exactitud": "Exactitud",
            }
        )[["Estrategia", "F1", "Precisión", "Sensibilidad", "Exactitud"]],
        width="stretch",
        hide_index=True,
    )
