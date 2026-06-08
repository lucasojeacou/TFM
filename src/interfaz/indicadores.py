"""Paneles de métricas resumen por estrategia para el simulador (fase 2)."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from src.interfaz.inferencia import etiqueta_atributo, indices_rango


def filas_frecuencia_categoria(
    vecinos_validos: List[int],
    atributos_amigos: Dict[int, List[int]],
    mapeador,
    category_key: str,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Top atributos más frecuentes en la vecindad para una categoría."""
    indices = indices_rango(mapeador, category_key)
    total_vecinos = max(1, len(vecinos_validos))
    conteo: Dict[int, int] = {idx: 0 for idx in indices}

    for vecino in vecinos_validos:
        attrs = atributos_amigos.get(vecino, [])
        for idx in indices:
            try:
                if int(attrs[idx]) == 1:
                    conteo[idx] += 1
            except IndexError:
                continue

    filas = []
    for idx, count in conteo.items():
        porcentaje = (100.0 * count) / total_vecinos
        filas.append({"label": etiqueta_atributo(mapeador, idx), "score": porcentaje, "detail": f"{count}/{total_vecinos} amigos"})

    return sorted(filas, key=lambda item: item["score"], reverse=True)[:limit]


def filas_resumen_circulos(circulos: Dict[str, List[int]], vecinos_validos: List[int], limit: int = 5) -> List[Dict[str, object]]:
    """Penetración de vecinos en cada círculo social SNAP."""
    total_vecinos = max(1, len(vecinos_validos))
    rows = []
    for circle_name, members in circulos.items():
        miembros_validos = [m for m in members if int(m) in vecinos_validos]
        if not miembros_validos:
            continue
        share = len(miembros_validos) / total_vecinos
        rows.append({"label": str(circle_name), "score": share, "detail": f"{len(miembros_validos)} amigos"})
    return sorted(rows, key=lambda item: item["score"], reverse=True)[:limit]


def filas_resumen_puentes(strategy, vecinos_validos: List[int], limit: int = 5) -> List[Dict[str, object]]:
    """Vecinos ordenados por centralidad de intermediación."""
    if not vecinos_validos:
        return []
    centralidad = strategy._obtener_centralidad() if hasattr(strategy, "_obtener_centralidad") else {}
    rows = []
    for vecino in vecinos_validos:
        score = float(centralidad.get(vecino, 0.0))
        rows.append({"label": f"Nodo {vecino}", "score": score, "detail": f"Intermediación {score:.3f}"})
    return sorted(rows, key=lambda item: item["score"], reverse=True)[:limit]


def filas_resumen_transitividad(
    grafo,
    ego_id: int,
    vecinos_validos: List[int],
    atributos_amigos: Dict[int, List[int]],
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Nodos a dos saltos más conectados con la vecindad directa."""
    vecinos_set = set(vecinos_validos)
    nodos_2hop = []
    for vecino in vecinos_validos:
        for candidato in grafo.neighbors(vecino):
            if int(candidato) == ego_id or int(candidato) in vecinos_set or int(candidato) not in atributos_amigos:
                continue
            nodos_2hop.append(int(candidato))
    if not nodos_2hop:
        return []
    conteo: Dict[int, int] = {}
    for nodo in nodos_2hop:
        conteo[nodo] = conteo.get(nodo, 0) + 1
    return sorted([{"label": f"Nodo {nodo}", "score": count / len(vecinos_validos), "detail": f"aparece en {count} caminos"} for nodo, count in conteo.items()], key=lambda item: item["score"], reverse=True)[:limit]


def filas_resumen_similitud(strategy, ego_id: int, limit: int = 5) -> List[Dict[str, object]]:
    """Top nodos por similitud estructural Adamic-Adar."""
    if not hasattr(strategy, "_obtener_similitudes_estructurales"):
        return []
    similitudes = strategy._obtener_similitudes_estructurales(ego_id)
    return [{"label": f"Nodo {nodo}", "score": float(score), "detail": "Adamic-Adar"} for nodo, score in similitudes[:limit]]


def filas_resumen_comunidades(strategy, ego_id: int, vecinos_validos: List[int], limit: int = 5) -> List[Dict[str, object]]:
    """Distribución de vecinos entre comunidades Louvain detectadas."""
    if not vecinos_validos or not hasattr(strategy, "_obtener_comunidades_subgrafo"):
        return []
    comunidades = strategy._obtener_comunidades_subgrafo(ego_id, vecinos_validos)
    conteo: Dict[int, int] = {}
    for nodo, comunidad_id in comunidades.items():
        conteo[int(comunidad_id)] = conteo.get(int(comunidad_id), 0) + 1
    total = max(1, len(vecinos_validos))
    rows = [{"label": f"Comunidad {cid}", "score": count / total, "detail": f"{count} amigos"} for cid, count in conteo.items()]
    return sorted(rows, key=lambda item: item["score"], reverse=True)[:limit]


def renderizar_filas_influencia(rows: List[Dict[str, object]]) -> None:
    """Gráfico de barras horizontales con las contribuciones de explicabilidad principales."""
    if not rows:
        return
    total = len(rows)
    limite = min(5, total)
    top_rows = rows[:limite]

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(top_rows)
    df = df.iloc[::-1]
    df["Texto"] = df["Porcentaje"].apply(lambda p: f"{p:.2f}%")

    fig = px.bar(
        df,
        x="Porcentaje",
        y="Nodo",
        orientation="h",
        text="Texto",
        color="Porcentaje",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        xaxis_title="Contribución relativa (%)",
        yaxis_title="",
        coloraxis_showscale=False,
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_traces(
        textposition="outside",
        marker_line_color='rgba(0,0,0,0)',
        marker_line_width=0,
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if total > limite:
        st.markdown(f"<p style='color:#64748b; font-size:0.85rem; margin-top:-0.5rem;'>Y {total - limite} vecinos más con contribución muy parecida.</p>", unsafe_allow_html=True)
