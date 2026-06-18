"""Renderizado PyVis del grafo según fase del simulador, estrategia y categoría."""

from __future__ import annotations

import math
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaz.analisis import InferenceResult

from src.interfaz.configuracion import (
    BASE_NODE_SIZE, EGO_SIZE, EGO_UNKNOWN_COLOR, EGO_WARNING_COLOR,
    NEUTRAL_NODE_COLOR, NEUTRAL_OTHER_COLOR, NEUTRAL_EDGE_COLOR
)
from src.interfaz.inferencia import (
    etiqueta_atributo,
    formato_atributos_activados,
    generar_paleta,
    indices_rango,
    numero_feat_dataset,
    paleta_categoria,
    resolver_indice_inferido_ego,
)
from src.interfaz.estilos_fase2 import (
    aplicar_estilos_aristas_fase2,
    aplicar_estilos_nodos_fase2,
)
from src.estrategias.base import obtener_vecinos_validos
from src.interfaz.ui import construir_estilo_nodo, construir_estilo_arista


PYVIS_OPTIONS_JSON = """
{
  "interaction": {
    "hover": true,
    "tooltipDelay": 120,
    "hideEdgesOnDrag": true,
    "hideEdgesOnZoom": true
  },
  "physics": {
    "enabled": false
  },
  "nodes": {
    "shape": "dot",
    "font": {"size": 5, "face": "arial"}
  },
  "edges": {
    "smooth": {"enabled": false}
  }
}
"""


def inicializar_aristas(grafo) -> dict[tuple[int, int], dict[str, object]]:
    """Inicializa el diccionario de propiedades visuales para todas las aristas del grafo."""
    edge_rows: dict[tuple[int, int], dict[str, object]] = {}
    for u, v in grafo.edges():
        edge_rows[tuple(sorted((int(u), int(v))))] = construir_estilo_arista(NEUTRAL_EDGE_COLOR)
    return edge_rows


def construir_estilo_nodo_categoria(nodo, category_key, mapeador, atributos_amigos, perfil_ego) -> tuple[str, str]:
    """Determina el color y la etiqueta visual del nodo vecino segun sus atributos activos de la categoria."""
    values = atributos_amigos.get(nodo, perfil_ego)
    if not values:
        return NEUTRAL_NODE_COLOR, str(nodo)
    palette = paleta_categoria(mapeador, category_key)
    for idx in indices_rango(mapeador, category_key):
        if idx < len(values) and int(values[idx]) == 1:
            return palette[idx], numero_feat_dataset(mapeador, idx)
    return NEUTRAL_NODE_COLOR, str(nodo)


def construir_estilo_ego(category_key, mapeador, elegido_por_categoria, inferidos=None) -> tuple[str, str, float]:
    """Asigna el color del ego segun el primer valor inferido principal de la categoria."""
    _, score = elegido_por_categoria.get(category_key, (None, 0.0))
    winner_idx = resolver_indice_inferido_ego(mapeador, category_key, elegido_por_categoria, inferidos)
    if winner_idx is None:
        return EGO_WARNING_COLOR, "Ego | Sin predicción", float(score)
    palette = paleta_categoria(mapeador, category_key)
    color = palette.get(int(winner_idx), EGO_UNKNOWN_COLOR)
    if inferidos is not None:
        titulo = f"Ego | Inferido: {formato_atributos_activados(mapeador, category_key, inferidos)}"
    else:
        titulo = f"Ego | Inferido: {numero_feat_dataset(mapeador, int(winner_idx))}"
    return color, titulo, float(score)


def procesar_propiedades_nodos(
    grafo,
    posiciones,
    strategy_key: str,
    phase: int,
    ego_id: int | None = None,
    vecinos_validos: Iterable[int] = (),
    circulos: dict[str, list[int]] | None = None,
    estrategia=None,
    atributos_amigos=None,
) -> dict[int, dict[str, object]]:
    """Inicializa el estado geométrico de los nodos y, en fase 2, delega el coloreado por estrategia."""
    node_rows: dict[int, dict[str, object]] = {}
    for nodo in grafo.nodes():
        nodo_int = int(nodo)
        x, y = posiciones.get(nodo_int, (0.0, 0.0))
        if ego_id is not None and nodo_int == int(ego_id):
            node_rows[nodo_int] = construir_estilo_nodo(x, y, color=EGO_WARNING_COLOR, size=EGO_SIZE, title=f"Ego {ego_id}")
        else:
            node_rows[nodo_int] = construir_estilo_nodo(x, y, title=str(nodo_int))

    if phase != 2:
        return node_rows

    return aplicar_estilos_nodos_fase2(
        node_rows,
        grafo,
        strategy_key,
        ego_id,
        vecinos_validos,
        circulos,
        estrategia,
        atributos_amigos,
    )


def inicializar_nodos(
    grafo,
    posiciones,
    strategy_key: str,
    phase: int = 1,
    ego_id: int | None = None,
    vecinos_validos: Iterable[int] = (),
    circulos: dict[str, list[int]] | None = None,
    estrategia=None,
    atributos_amigos=None,
) -> dict[int, dict[str, object]]:
    """Instancia el estado geometrico base de todos los nodos del grafo a partir de las coordenadas de layout."""
    return procesar_propiedades_nodos(
        grafo,
        posiciones,
        strategy_key,
        phase,
        ego_id=ego_id,
        vecinos_validos=vecinos_validos,
        circulos=circulos,
        estrategia=estrategia,
        atributos_amigos=atributos_amigos,
    )


def estilizar_vecinos(node_rows, vecinos_validos, category_key, mapeador, atributos_amigos, perfil_ego):
    """Aplica los colores correspondientes de la categoria a los nodos de la vecindad del ego."""
    for nodo in vecinos_validos:
        color, title = construir_estilo_nodo_categoria(nodo, category_key, mapeador, atributos_amigos, perfil_ego)
        node_rows.setdefault(nodo, {}).update({"color": color, "label": "", "title": title, "size": BASE_NODE_SIZE})


def estilizar_veredicto(node_rows, grafo, ego_id: int, vecinos_validos, category_key, mapeador, elegido_por_categoria, inferidos=None) -> dict[str, object]:
    """Modifica el diseño del grafo para resaltar el veredicto final sobre el ego y neutralizar el resto de nodos."""
    color, titulo, _ = construir_estilo_ego(category_key, mapeador, elegido_por_categoria, inferidos)
    vecinos_set = set(vecinos_validos)
    for nodo in grafo.nodes():
        nodo_int = int(nodo)
        if nodo_int == ego_id:
            node_rows[nodo_int] = {
                **node_rows.get(nodo_int, {}),
                "shape": "dot",
                "color": color,
                "size": EGO_SIZE,
                "label": "",
                "title": titulo,
            }
        elif nodo_int in vecinos_set:
            node_rows.setdefault(nodo_int, {}).update({"size": BASE_NODE_SIZE, "label": ""})
        else:
            node_rows.setdefault(nodo_int, {}).update({"color": NEUTRAL_OTHER_COLOR, "label": "", "title": str(nodo_int)})
    return {"color": color, "title": titulo}


def estilizar_aportes_xai(node_rows, grafo, ego_id: int, atributo_enfocado: int, contribuciones_xai, mapeador=None, atributos_amigos=None):
    """Enruta la estilización de explicabilidad enfocando un único atributo de la red."""
    estilizar_aportes_xai_multiple(node_rows, grafo, ego_id, [int(atributo_enfocado)], contribuciones_xai, mapeador, atributos_amigos)


def estilizar_aportes_xai_multiple(node_rows, grafo, ego_id: int, atributos_enfocados: Iterable[int], contribuciones_xai, mapeador=None, atributos_amigos=None):
    """Calcula la intensidad del tamaño y color de los vecinos basandose en sus aportaciones metricas de explicabilidad."""
    atributos = [int(atributo) for atributo in atributos_enfocados if atributo is not None]
    if not atributos:
        return
    paleta = generar_paleta(len(atributos))
    color_por_atributo = {atributo: paleta[pos] for pos, atributo in enumerate(atributos)}
    aportes_por_atributo = {atributo: dict(contribuciones_xai.get(int(atributo), {})) if contribuciones_xai else {} for atributo in atributos}
    max_peso_global = max((float(peso) for aportes in aportes_por_atributo.values() for peso in aportes.values()), default=1.0)

    for nodo in grafo.nodes():
        nodo_int = int(nodo)
        if nodo_int == ego_id:
            node_rows.setdefault(nodo_int, {}).update({"color": EGO_WARNING_COLOR, "size": EGO_SIZE, "label": "", "title": "Ego"})
            continue
        mejores_aportes: list[tuple[int, float]] = []
        for atributo in atributos:
            peso = float(aportes_por_atributo.get(atributo, {}).get(nodo_int, 0.0))
            if peso > 0.0:
                mejores_aportes.append((atributo, peso))
        if not mejores_aportes:
            valores = atributos_amigos.get(nodo_int, []) if atributos_amigos is not None else []
            atributos_activos = [
                atributo
                for atributo in atributos
                if 0 <= atributo < len(valores) and int(valores[atributo]) == 1
            ]
            if atributos_activos:
                atributo_ganador = atributos_activos[0]
                if mapeador is not None:
                    titulo = " | ".join(etiqueta_atributo(mapeador, atributo) for atributo in atributos_activos)
                else:
                    titulo = " | ".join(str(atributo) for atributo in atributos_activos)
                node_rows.setdefault(nodo_int, {}).update({
                    "color": color_por_atributo[atributo_ganador],
                    "label": "",
                    "title": titulo,
                    "size": BASE_NODE_SIZE,
                })
                continue
            node_rows.setdefault(nodo_int, {}).update({"color": NEUTRAL_OTHER_COLOR, "label": "", "title": str(nodo_int), "size": max(2, BASE_NODE_SIZE - 1)})
            continue
        atributo_ganador, peso_ganador = max(mejores_aportes, key=lambda item: item[1])
        intensidad = peso_ganador / max_peso_global if max_peso_global else 0.0
        size = BASE_NODE_SIZE + min(2, int(round(intensidad * 2)))
        if mapeador is not None:
            titulo = " | ".join(f"{etiqueta_atributo(mapeador, atributo)}: {peso:.3f}" for atributo, peso in sorted(mejores_aportes, key=lambda item: item[1], reverse=True))
        else:
            titulo = " | ".join(
                f"Atributo {atributo}: {peso:.3f}"
                for atributo, peso in sorted(mejores_aportes, key=lambda item: item[1], reverse=True)
            )
        node_rows.setdefault(nodo_int, {}).update({"color": color_por_atributo[atributo_ganador], "size": size, "label": "", "title": titulo})


def calcular_inferencia(estrategia, ego_id, circulos, phase, resultado_inferencia: InferenceResult | None = None):
    """Controla la ejecucion del algoritmo de inferencia bloqueando su computo en fases iniciales."""
    if phase < 4:
        return [], {}, None
    if resultado_inferencia is not None:
        return (
            list(resultado_inferencia.inferidos),
            dict(resultado_inferencia.elegido_por_categoria),
            dict(resultado_inferencia.contribuciones_xai or {}),
        )
    return estrategia.inferir(ego_id, circulos=circulos)


def normalizar_lista_atributos(atributo_enfocado) -> list[int]:
    """Normaliza y limpia los identificadores de atributos enfocados para evitar valores nulos."""
    if atributo_enfocado is None:
        return []
    if isinstance(atributo_enfocado, (list, tuple, set)):
        atributos = [int(at) for at in atributo_enfocado if at is not None and not (isinstance(at, float) and math.isnan(at))]
        return list(dict.fromkeys(atributos))
    if isinstance(atributo_enfocado, float) and math.isnan(atributo_enfocado):
        return []
    return [int(atributo_enfocado)]


def construir_insights(strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
    """Invoca los componentes secundarios del plugin para extraer las metricas resumen del subgrafo."""
    if phase < 2:
        return "", [], None
    from src.interfaz.plugins import get_strategy_ui
    plugin = get_strategy_ui(strategy_key)
    insight = plugin.render_insights(
        strategy_key,
        phase,
        grafo,
        ego_id,
        category_key,
        vecinos_validos,
        atributos_amigos,
        mapeador,
        estrategia,
        circulos,
    )
    return insight.get("title", ""), insight.get("rows", []), insight.get("note")


def renderizar_grafo(
    grafo: nx.Graph,
    posiciones,
    ego_id,
    atributos_amigos,
    perfil_ego,
    category_key,
    mapeador,
    phase,
    estrategia,
    strategy_key,
    circulos,
    atributo_enfocado=None,
    spacing_scale: float = 3.1,
    fixed_nodes: bool = True,
    resultado_inferencia: InferenceResult | None = None,
):
    """Ensambla los estilos de nodos y aristas segun la fase de simulacion y renderiza el iframe interactivo en Streamlit."""
    vecinos_validos = obtener_vecinos_validos(grafo, atributos_amigos, ego_id)
    inferidos, elegido_por_categoria, contribuciones_xai = calcular_inferencia(
        estrategia, ego_id, circulos, phase, resultado_inferencia=resultado_inferencia
    )
    node_rows = inicializar_nodos(
        grafo,
        posiciones,
        strategy_key,
        phase,
        ego_id=ego_id,
        vecinos_validos=vecinos_validos,
        circulos=circulos,
        estrategia=estrategia,
        atributos_amigos=atributos_amigos,
    )
    edge_rows = inicializar_aristas(grafo)
    from src.interfaz.plugins import get_strategy_ui
    plugin = get_strategy_ui(strategy_key)

    if phase == 2:
        aplicar_estilos_aristas_fase2(
            edge_rows,
            grafo,
            strategy_key,
            ego_id,
            vecinos_validos,
            atributos_amigos,
            estrategia,
            resaltar_aristas_ego=plugin.highlight_phase2_edges(),
        )
    atributos_enfocados = normalizar_lista_atributos(atributo_enfocado)

    if phase >= 3 and not atributos_enfocados:
        estilizar_vecinos(node_rows, vecinos_validos, category_key, mapeador, atributos_amigos, perfil_ego)
    ego_estilo_veredicto = None
    if phase >= 4 and not atributos_enfocados:
        ego_estilo_veredicto = estilizar_veredicto(
            node_rows, grafo, ego_id, vecinos_validos, category_key, mapeador, elegido_por_categoria, inferidos
        )
    if phase >= 4 and atributos_enfocados:
        if len(atributos_enfocados) == 1:
            estilizar_aportes_xai(node_rows, grafo, ego_id, int(atributos_enfocados[0]), contribuciones_xai, mapeador, atributos_amigos)
        else:
            estilizar_aportes_xai_multiple(node_rows, grafo, ego_id, atributos_enfocados, contribuciones_xai, mapeador, atributos_amigos)

    if phase >= 4 and not atributos_enfocados:
        spacing_scale = 2.0

    render_graph = grafo.copy()
    ego_int = int(ego_id)
    if ego_int in posiciones:
        cx, cy = posiciones[ego_int]
    else:
        xs = [posiciones.get(int(n), (0.0, 0.0))[0] for n in render_graph.nodes()]
        ys = [posiciones.get(int(n), (0.0, 0.0))[1] for n in render_graph.nodes()]
        if xs and ys:
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
        else:
            cx = cy = 0.0

    for nodo in list(render_graph.nodes()):
        nodo_int = int(nodo)
        x, y = posiciones.get(nodo_int, (0.0, 0.0))
        nx_pos = cx + (x - cx) * float(spacing_scale)
        ny_pos = cy + (y - cy) * float(spacing_scale)
        props = dict(node_rows.get(nodo_int, {}))
        props.update({"x": float(nx_pos), "y": float(ny_pos), "fixed": bool(fixed_nodes)})
        props.setdefault("shape", "dot")
        if ego_estilo_veredicto is not None and nodo_int == ego_int:
            props["shape"] = "dot"
            props["color"] = ego_estilo_veredicto["color"]
            props["size"] = EGO_SIZE
            props["label"] = ""
            props["title"] = str(ego_estilo_veredicto.get("title", f"Ego {ego_id}"))
        else:
            props.setdefault("size", BASE_NODE_SIZE)
            props.setdefault("label", "")
            props.setdefault("color", NEUTRAL_OTHER_COLOR)
        render_graph.nodes[nodo_int].update(props)

    for u, v in list(render_graph.edges()):
        render_graph.edges[u, v].update(edge_rows.get(tuple(sorted((int(u), int(v)))), {}))

    net = Network(height="760px", width="100%", bgcolor="#0f172a", font_color="#e5e7eb", directed=False, notebook=False, cdn_resources="in_line")
    net.from_nx(render_graph)
    net.set_options(PYVIS_OPTIONS_JSON)
    html = net.generate_html(notebook=False)

    try:
        st.iframe(srcdoc=html, height=780, scrolling=True)
    except TypeError:
        components.html(html, height=780, scrolling=True)

    insight_title, insight_rows, insight_note = construir_insights(strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos)
    return inferidos, elegido_por_categoria, contribuciones_xai, {"title": insight_title, "rows": insight_rows, "note": insight_note}
