"""Componentes HTML/Streamlit reutilizables: tarjetas, fases y tabla de auditoría."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Mapping

import streamlit as st

from src.interfaz.configuracion import BASE_NODE_SIZE, NEUTRAL_OTHER_COLOR


def render_card(texto: str, color: str) -> str:
    """HTML de una tarjeta de leyenda con indicador de color."""
    texto = (texto or "").strip()
    return (
        "<div style='display:flex;align-items:center;justify-content:center;gap:0.55rem;"
        "padding:0.65rem 0.75rem;border:1px solid rgba(148,163,184,0.25);border-radius:12px;"
        "background:rgba(15,23,42,0.35);min-height:56px;width:100%;box-sizing:border-box;"
        "text-align:center;'>"
        f"<span style='width:12px;height:12px;border-radius:999px;background:{color};flex:0 0 auto;'></span>"
        f"<span style='font-size:0.88rem;font-weight:700;color:#f8fafc;line-height:1.25;'>{texto}</span>"
        "</div>"
    )


def _columnas_leyenda(cantidad: int) -> int:
    """Número de columnas de la cuadrícula según la cantidad de tarjetas."""
    if cantidad <= 1:
        return 1
    if cantidad == 2:
        return 2
    if cantidad <= 4:
        return cantidad
    return 4


def _contenedor_tarjetas_html(tarjetas_html: list[str]) -> str:
    """HTML con grid responsive para un conjunto de tarjetas."""
    cantidad = len(tarjetas_html)
    columnas = _columnas_leyenda(cantidad)
    celdas = "".join(f"<div class='legend-card-cell'>{html}</div>" for html in tarjetas_html)
    return (
        "<style>"
        ".legend-grid{display:grid;gap:0.65rem;width:100%;margin:0.2rem 0 0.45rem 0;"
        f"grid-template-columns:repeat({columnas},minmax(0,1fr));"
        ".legend-card-cell{min-width:0;width:100%;}"
        "@media (max-width:768px){"
        ".legend-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}"
        "}"
        "</style>"
        f"<div class='legend-grid'>{celdas}</div>"
    )


def render_card_grid(
    cards: Iterable[Mapping[str, object]],
    titulo: str | None = None,
    caption: str | None = None,
    mostrar_progreso: bool = False,
    clave_progreso: str = "score",
) -> None:
    """Cuadrícula responsive de tarjetas de leyenda en Streamlit."""
    cards_list = list(cards)
    if not cards_list:
        if caption:
            st.caption(caption)
        return

    if titulo:
        st.markdown(f"##### {titulo}")

    tarjetas_html = [
        render_card(
            str(card.get("texto") or card.get("title") or ""),
            str(card.get("color", "#8b5cf6")),
        )
        for card in cards_list
    ]
    st.markdown(_contenedor_tarjetas_html(tarjetas_html), unsafe_allow_html=True)

    if mostrar_progreso:
        for card in cards_list:
            progreso = float(card.get(clave_progreso, 0.0))
            if progreso > 1.0:
                progreso /= 100.0
            st.progress(min(1.0, max(0.0, progreso)))

    if caption:
        st.caption(caption)


def construir_estilo_nodo(
    x: float,
    y: float,
    color: str = NEUTRAL_OTHER_COLOR,
    label: str = "",
    size: int = BASE_NODE_SIZE,
    title: str = "",
    fixed: bool = True,
) -> dict:
    """Propiedades visuales de un nodo para PyVis."""
    return {
        "x": float(x),
        "y": float(y),
        "fixed": fixed,
        "shape": "dot",
        "size": size,
        "label": label,
        "color": color,
        "title": title,
    }


def hex_to_rgba(hex_val: str, opacity: float) -> str:
    """Convierte color hex a cadena CSS rgba."""
    h = hex_val.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity:.3f})"


def construir_estilo_arista(color: str, width: float = 0.4, opacity: float = 0.14) -> dict:
    """Propiedades visuales de una arista para PyVis."""
    return {"color": hex_to_rgba(color, opacity), "width": width}


def renderizar_tarjetas_fase(phase: int) -> None:
    """Indicador visual de las cuatro fases del simulador."""
    items = [
        ("1", "Reposo", "Grafo neutro con el ego en el centro."),
        ("2", "Vecindad", "Localización de amigos relevantes."),
        ("3", "Atributos", "Coloreado por categorías visibles."),
        ("4", "Veredicto", "Inferencia resultante sobre la red."),
    ]
    cards = []
    for number, title, copy in items:
        active = " step-card-active" if int(number) == int(phase) else ""
        cards.append(
            f"<div class='step-card{active}'><div class='step-title'>{number}. {title}</div><div class='step-copy'>{copy}</div></div>"
        )
    st.markdown(f"<div class='step-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def renderizar_encabezado_fase(phase: int, titulo: str) -> None:
    """Título y badge de fase actual."""
    st.markdown(
        f"<div class='phase-header'><h4>Paso actual: {titulo}</h4><span class='phase-badge'>Fase {phase}/4</span></div>",
        unsafe_allow_html=True,
    )


def renderizar_categoria_seleccionada(seleccion) -> None:
    """Bloque Real vs Inferido de la categoría seleccionada en auditoría."""
    estado = str(seleccion.get("Estado", "Sin datos"))
    estado_clase = {
        "Correcto": "status-ok",
        "OK": "status-ok",
        "Parcial": "status-mid",
        "Fallido": "status-bad",
        "Sin datos": "status-none",
    }.get(estado, "status-none")
    st.markdown(f"<span class='status-pill {estado_clase}'>{estado}</span>", unsafe_allow_html=True)
    col_real, col_pred = st.columns(2)
    col_real.markdown("**Dato real**")
    col_real.write(str(seleccion.get("Real", "-")))
    col_pred.markdown("**Inferido**")
    col_pred.write(str(seleccion.get("Inferido", "-")))


def tabla_auditoria(auditoria):
    """Columnas visibles de la tabla de auditoría, ordenadas por categoría."""
    if auditoria is None:
        return auditoria
    if hasattr(auditoria, "empty") and auditoria.empty:
        return auditoria
    return auditoria[["Categoria", "Real", "Inferido", "Estado", "Confianza"]].sort_values("Categoria")


def tabla_auditoria_coloreada(auditoria):
    """Tabla de auditoría con colores en la columna Estado."""
    tabla = tabla_auditoria(auditoria)
    if tabla is None or (hasattr(tabla, "empty") and tabla.empty):
        return tabla

    def color_estado(row):
        """Estilo CSS de la celda Estado según el resultado de auditoría."""
        color = {
            "Correcto": "background-color: #16a34a; color: #ffffff; font-weight: 800;",
            "OK": "background-color: #16a34a; color: #ffffff; font-weight: 800;",
            "Parcial": "background-color: #f97316; color: #ffffff; font-weight: 800;",
            "Fallido": "background-color: #dc2626; color: #ffffff; font-weight: 800;",
            "Sin datos": "background-color: #475569; color: #ffffff; font-weight: 800;",
        }.get(row["Estado"], "")
        return [color if col == "Estado" else "" for col in row.index]

    return tabla.style.apply(color_estado, axis=1)


def atributo_label(snapshot: SimpleNamespace, idx: int) -> str:
    """Etiqueta legible de un atributo a partir del mapeador del snapshot."""
    from src.interfaz.inferencia import etiqueta_atributo

    return etiqueta_atributo(snapshot.mapeador, int(idx))


def categoria_desde_tabla(auditoria, tabla_evento):
    """Categoría seleccionada en la tabla interactiva de auditoría."""
    tabla = tabla_auditoria(auditoria)
    if tabla is None or (hasattr(tabla, "empty") and tabla.empty):
        return None
    filas = tabla_evento.get("selection", {}).get("rows", []) if tabla_evento else []
    if filas:
        tabla_reset = tabla.reset_index(drop=True)
        return tabla_reset.iloc[int(filas[0])]["Categoria"]
    return tabla.iloc[0]["Categoria"]


def categoria_clave_por_nombre(auditoria, nombre_categoria):
    """Clave interna 'categoria' a partir de la etiqueta de display."""
    if auditoria is None or (hasattr(auditoria, "empty") and auditoria.empty):
        return None
    fila = auditoria[auditoria["Categoria"] == nombre_categoria]
    if fila.empty:
        return auditoria.iloc[0].get("categoria", auditoria.iloc[0].get("Categoria"))
    return fila.iloc[0].get("categoria", fila.iloc[0].get("Categoria"))
