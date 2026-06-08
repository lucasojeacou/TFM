"""Utilidades compartidas entre pestañas: leyenda, influencias y filtros de atributos."""

from __future__ import annotations

import math

import streamlit as st

from src.interfaz.analisis import top_contribuyentes
from src.interfaz.indicadores import renderizar_filas_influencia
from src.interfaz.inferencia import etiqueta_atributo, generar_paleta
from src.interfaz.ui import atributo_label, render_card_grid
from src.interfaz.visual import tarjeta_leyenda


def historico_ego(df_historico, ego_id: int):
    """Filas del CSV histórico correspondientes a un ego."""
    if df_historico is None or df_historico.empty:
        return None
    return df_historico[df_historico["ego_id"].astype(int) == int(ego_id)].copy()


def atributos_enfocados_desde_seleccion(seleccion, atributo_enfocado=None) -> list[int]:
    """Índices de atributos a resaltar en XAI a partir de la fila de auditoría."""
    if atributo_enfocado is None:
        atributo = seleccion.get("ganador_idx")
        if atributo is not None and not (isinstance(atributo, float) and math.isnan(atributo)):
            return [int(atributo)]
        pred_indices = [int(idx) for idx in list(seleccion["pred_indices"])]
        return [pred_indices[0]] if pred_indices else []
    if isinstance(atributo_enfocado, (list, tuple, set)):
        return [
            int(at)
            for at in atributo_enfocado
            if at is not None and not (isinstance(at, float) and math.isnan(at))
        ]
    if isinstance(atributo_enfocado, float) and math.isnan(atributo_enfocado):
        return []
    return [int(atributo_enfocado)]


def renderizar_influencias(resultado, seleccion, snapshot, atributo_enfocado=None) -> None:
    """Gráfico de barras con contribuciones XAI de la categoría seleccionada."""
    atributos = atributos_enfocados_desde_seleccion(seleccion, atributo_enfocado)
    if not atributos:
        st.caption(
            "No hay contribuciones positivas para esta categoría. "
            "Suele ocurrir cuando la estrategia no infiere ningún atributo claro."
        )
        return

    if len(atributos) > 1:
        tab_labels = [atributo_label(snapshot, int(at)) for at in atributos]
        tabs = st.tabs(tab_labels)
        for i, atributo in enumerate(atributos):
            with tabs[i]:
                contribuyentes = top_contribuyentes(resultado, atributo)
                if contribuyentes.empty:
                    st.caption("Sin aportaciones registradas.")
                    continue
                renderizar_filas_influencia(
                    [
                        {"Rango": idx + 1, "Nodo": row["Nodo"], "Peso": row["Peso"], "Porcentaje": row["Porcentaje"]}
                        for idx, row in contribuyentes.iterrows()
                    ]
                )
    else:
        atributo = atributos[0]
        contribuyentes = top_contribuyentes(resultado, atributo)
        if not contribuyentes.empty:
            st.markdown(f"##### {atributo_label(snapshot, int(atributo))}")
            renderizar_filas_influencia(
                [
                    {"Rango": idx + 1, "Nodo": row["Nodo"], "Peso": row["Peso"], "Porcentaje": row["Porcentaje"]}
                    for idx, row in contribuyentes.iterrows()
                ]
            )


def _renderizar_bloque_tarjetas(tarjetas: list, titulo: str | None, caption: str | None = None) -> None:
    """Renderiza un bloque de tarjetas de leyenda si hay contenido."""
    if not tarjetas:
        return
    render_card_grid(tarjetas, titulo=titulo, caption=caption)


def renderizar_leyenda_grafo(
    strategy_key: str,
    mapeador,
    category_key: str,
    circulos,
    atributos_amigos,
    vecinos_validos,
    phase: int,
    atributos_enfocados: list[int] | None = None,
    mostrar_leyenda_valores: bool = True,
    inferidos: list[int] | None = None,
    elegido_por_categoria: dict | None = None,
    estrategia=None,
    ego_id: int | None = None,
) -> None:
    """Delega en el plugin de estrategia y pinta leyenda de nodos, XAI y aristas."""
    from src.interfaz.plugins import get_strategy_ui

    plugin = get_strategy_ui(strategy_key)
    circle_palette = (
        generar_paleta(len(circulos)) if strategy_key == "circulos" and circulos else []
    )
    leyenda = plugin.render_legend(
        mapeador,
        category_key,
        strategy_key,
        circulos,
        circle_palette,
        atributos_amigos,
        vecinos_validos,
        phase,
        inferidos=inferidos,
        elegido_por_categoria=elegido_por_categoria,
        estrategia=estrategia,
        ego_id=ego_id,
    )

    titulo_nodos = str(leyenda.get("title") or "Leyenda del grafo")
    cards_nodos = list(leyenda.get("cards", [])) if mostrar_leyenda_valores else []
    caption = str(leyenda["caption"]) if leyenda.get("caption") else None
    _renderizar_bloque_tarjetas(cards_nodos, titulo_nodos, caption)

    if phase >= 4 and atributos_enfocados and len(atributos_enfocados) > 1:
        paleta = generar_paleta(len(atributos_enfocados))
        cards_xai = [
            tarjeta_leyenda(paleta[pos], etiqueta_atributo(mapeador, int(atributo)))
            for pos, atributo in enumerate(atributos_enfocados)
        ]
        _renderizar_bloque_tarjetas(cards_xai, "Atributos explicados")

    edge_cards = list(leyenda.get("edge_cards") or [])
    _renderizar_bloque_tarjetas(edge_cards, "Aristas" if edge_cards else None)
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)


__all__ = [
    "historico_ego",
    "atributos_enfocados_desde_seleccion",
    "renderizar_influencias",
    "renderizar_leyenda_grafo",
]
