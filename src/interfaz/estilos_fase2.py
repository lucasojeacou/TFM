"""Estilos visuales de la fase 2 (nodos y aristas) por estrategia; ``grafo.py`` delega aquí."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.interfaz.configuracion import (
    ACTIVE_EDGE_COLOR,
    BASE_NODE_SIZE,
    NEUTRAL_EDGE_COLOR,
    NEUTRAL_NODE_COLOR,
)
from src.interfaz.visual import (
    ACTIVE_NODE_LEGEND_COLOR,
    BRIDGE_LEGEND_COLOR,
    PRESTIGIO_COLOR_ALTO,
    PRESTIGIO_COLOR_BAJO,
    SECOND_HOP_LEGEND_COLOR,
    circulo_visible_para_vecino,
    colores_comunidades_vecinos,
    interpolar_color,
    paleta_circulos,
)
from src.interfaz.ui import construir_estilo_arista, hex_to_rgba
from src.nucleo.grafo_utils import iter_pares_segundo_salto, nodos_segundo_salto

ACTIVE_NODE_COLOR = ACTIVE_NODE_LEGEND_COLOR
SECOND_HOP_NODE_COLOR = SECOND_HOP_LEGEND_COLOR
BRIDGE_NODE_COLOR = BRIDGE_LEGEND_COLOR
ALTA_SIMILITUD_COLOR = "#facc15"
SECOND_HOP_EDGE_COLOR = "#60a5fa"

NodeRows = dict[int, dict[str, object]]
EdgeRows = dict[tuple[int, int], dict[str, object]]


def _actualizar_nodo(node_rows: NodeRows, nodo_int: int, **props) -> None:
    """Aplica un bloque de propiedades a un nodo sin repetir la ruta de actualización."""
    node_rows.setdefault(nodo_int, {}).update(props)


def _nodos_puente_desde_centralidad(centralidad: dict[int, float]) -> set[int]:
    """Nodos mejor posicionados (top ~10 %) según la centralidad de intermediación."""
    if not centralidad:
        return set()
    ordenados = sorted(
        ((int(nodo), float(peso)) for nodo, peso in centralidad.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    limite = min(len(ordenados), max(3, int(np.ceil(len(ordenados) * 0.1))))
    umbral = ordenados[limite - 1][1]
    return {nodo for nodo, peso in ordenados if peso > 0.0 and peso >= umbral}


@dataclass
class ContextoFase2:
    """Datos precalculados que consumen los manejadores de nodos de la fase 2."""

    strategy_key: str
    vecinos_set: set[int]
    circulos: dict[str, list[int]] | None
    colores_circulos: dict[str, str] = field(default_factory=dict)
    nodos_puente: set[int] = field(default_factory=set)
    centralidad: dict[int, float] = field(default_factory=dict)
    nodos_segundo_salto: set[int] = field(default_factory=set)
    comunidades_dict: dict[int, int] = field(default_factory=dict)
    colores_comunidades: dict[int, str] = field(default_factory=dict)
    nodos_alta_sim: set[int] = field(default_factory=set)
    sim_dict: dict[int, float] = field(default_factory=dict)
    pagerank: dict[int, float] = field(default_factory=dict)


def _construir_contexto(
    grafo,
    strategy_key: str,
    ego_id: int | None,
    vecinos_validos,
    circulos,
    estrategia,
    atributos_amigos,
) -> ContextoFase2:
    """Precalcula, según la estrategia, las señales estructurales necesarias para la fase 2."""
    clave = str(strategy_key).strip().lower()
    vecinos_set = {int(vecino) for vecino in vecinos_validos}
    ctx = ContextoFase2(strategy_key=clave, vecinos_set=vecinos_set, circulos=circulos)

    if clave == "circulos":
        ctx.colores_circulos = paleta_circulos(circulos)
    elif clave == "puentes":
        ctx.centralidad = (estrategia.centralidad_intermediacion() or {}) if estrategia is not None else {}
        ctx.nodos_puente = _nodos_puente_desde_centralidad(ctx.centralidad)
    elif clave == "transitividad":
        ctx.nodos_segundo_salto = nodos_segundo_salto(grafo, ego_id, vecinos_set, atributos_amigos)
    elif clave == "comunidades":
        ctx.comunidades_dict, ctx.colores_comunidades = colores_comunidades_vecinos(estrategia, ego_id, vecinos_set)
    elif clave == "prestigio":
        ctx.pagerank = (estrategia.prestigio_pagerank() or {}) if estrategia is not None else {}
    elif clave == "similitud" and estrategia is not None and ego_id is not None:
        top_similares = estrategia.similitudes_estructurales(ego_id) or []
        ctx.nodos_alta_sim = {int(v) for v, _ in top_similares}
        ctx.sim_dict = {int(v): val for v, val in top_similares}

    return ctx


def _nodo_circulos(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Colorea cada vecino con el color de su círculo social SNAP."""
    circulo_info = circulo_visible_para_vecino(nodo_int, ctx.vecinos_set, ctx.circulos, ctx.colores_circulos)
    if circulo_info is not None:
        color_circulo, circle_name = circulo_info
        _actualizar_nodo(node_rows, nodo_int, color=color_circulo, size=BASE_NODE_SIZE + 1, label="", title=f"Vecino en círculo: {circle_name}")


def _nodo_puentes(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Resalta los nodos puente y neutraliza el resto de la vecindad."""
    if nodo_int in ctx.nodos_puente:
        centralidad = float(ctx.centralidad.get(nodo_int, 0.0))
        _actualizar_nodo(node_rows, nodo_int, color=BRIDGE_NODE_COLOR, size=BASE_NODE_SIZE + 2, label="", title=f"Puente crítico | centralidad {centralidad:.3f}")
    elif nodo_int in ctx.vecinos_set:
        _actualizar_nodo(node_rows, nodo_int, color=NEUTRAL_NODE_COLOR, size=BASE_NODE_SIZE, label="", title=str(nodo_int))


def _nodo_transitividad(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Distingue vecinos directos (primer salto) de los nodos de segundo salto."""
    if nodo_int in ctx.vecinos_set:
        _actualizar_nodo(node_rows, nodo_int, color=ACTIVE_NODE_COLOR, size=BASE_NODE_SIZE + 1, label="", title=f"Vecino directo | Nodo {nodo_int}")
    elif nodo_int in ctx.nodos_segundo_salto:
        _actualizar_nodo(node_rows, nodo_int, color=SECOND_HOP_NODE_COLOR, size=BASE_NODE_SIZE + 1, label="", title=f"Segundo salto | Nodo {nodo_int}")


def _nodo_prestigio(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Escala color y tamaño de cada vecino según su PageRank relativo."""
    if nodo_int not in ctx.vecinos_set:
        return
    pr_dict = ctx.pagerank
    pagerank_val = float(pr_dict.get(nodo_int, 0.0))
    vecinos_prs = [float(pr_dict.get(v, 0.0)) for v in ctx.vecinos_set]
    min_pr = min(vecinos_prs) if vecinos_prs else 0.0
    max_pr = max(vecinos_prs) if vecinos_prs else 1.0
    intensidad = (pagerank_val - min_pr) / (max_pr - min_pr) if max_pr - min_pr > 0 else 0.5
    color = interpolar_color(PRESTIGIO_COLOR_BAJO, PRESTIGIO_COLOR_ALTO, intensidad)
    size = BASE_NODE_SIZE + 1 + int(round(intensidad * 3.0))
    _actualizar_nodo(node_rows, nodo_int, color=color, size=size, label="", title=f"Vecino directo | Prestigio {pagerank_val:.4f}")


def _nodo_comunidades(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Colorea cada vecino según la comunidad Louvain a la que pertenece."""
    if nodo_int in ctx.vecinos_set and nodo_int in ctx.comunidades_dict:
        cid = ctx.comunidades_dict[nodo_int]
        color = ctx.colores_comunidades.get(cid, ACTIVE_NODE_COLOR)
        _actualizar_nodo(node_rows, nodo_int, color=color, size=BASE_NODE_SIZE + 1, label="", title=f"Vecino (Comunidad {cid}) | Nodo {nodo_int}")


def _nodo_similitud(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Resalta los nodos de alta similitud estructural sobre los vecinos directos."""
    if nodo_int in ctx.nodos_alta_sim:
        sim_val = ctx.sim_dict.get(nodo_int, 0.0)
        _actualizar_nodo(node_rows, nodo_int, color=ALTA_SIMILITUD_COLOR, size=BASE_NODE_SIZE + 2, label="", title=f"Alta similitud estructural | Puntuación {sim_val:.3f}")
    elif nodo_int in ctx.vecinos_set:
        _actualizar_nodo(node_rows, nodo_int, color=ACTIVE_NODE_COLOR, size=BASE_NODE_SIZE + 1, label="", title=f"Vecino directo | Nodo {nodo_int}")


def _nodo_vecindad_directa(node_rows: NodeRows, nodo_int: int, ctx: ContextoFase2) -> None:
    """Comportamiento por defecto: resalta los vecinos directos del ego."""
    if nodo_int in ctx.vecinos_set:
        _actualizar_nodo(node_rows, nodo_int, color=ACTIVE_NODE_COLOR, size=BASE_NODE_SIZE + 1, label="", title=f"Vecino directo | Nodo {nodo_int}")


_HANDLERS_NODOS = {
    "circulos": _nodo_circulos,
    "puentes": _nodo_puentes,
    "transitividad": _nodo_transitividad,
    "prestigio": _nodo_prestigio,
    "comunidades": _nodo_comunidades,
    "similitud": _nodo_similitud,
}


def aplicar_estilos_nodos_fase2(
    node_rows: NodeRows,
    grafo,
    strategy_key: str,
    ego_id: int | None,
    vecinos_validos,
    circulos,
    estrategia,
    atributos_amigos,
) -> NodeRows:
    """Aplica el coloreado de fase 2 a cada nodo según la estrategia activa."""
    ctx = _construir_contexto(grafo, strategy_key, ego_id, vecinos_validos, circulos, estrategia, atributos_amigos)
    handler = _HANDLERS_NODOS.get(ctx.strategy_key, _nodo_vecindad_directa)
    for nodo in grafo.nodes():
        handler(node_rows, int(nodo), ctx)
    return node_rows


def _resaltar_aristas_ego(edge_rows: EdgeRows, ego_id: int, vecinos_validos) -> None:
    """Resalta las conexiones directas ego–vecino."""
    for vecino in vecinos_validos:
        clave = tuple(sorted((int(ego_id), int(vecino))))
        if clave in edge_rows:
            edge_rows[clave].update(construir_estilo_arista(ACTIVE_EDGE_COLOR, width=1.4, opacity=0.52))


def _resaltar_aristas_transitividad(edge_rows: EdgeRows, grafo, ego_id: int, vecinos_validos, atributos_amigos) -> None:
    """Resalta primer salto (tenue) y segundo salto (punteado) en transitividad."""
    vecinos_set = {int(vecino) for vecino in vecinos_validos}
    for vecino in vecinos_set:
        clave = tuple(sorted((int(ego_id), int(vecino))))
        if clave in edge_rows:
            edge_rows[clave].update(construir_estilo_arista(ACTIVE_EDGE_COLOR, width=0.72, opacity=0.24))
    for vecino, candidato in iter_pares_segundo_salto(grafo, ego_id, vecinos_set, atributos_amigos):
        clave = tuple(sorted((int(vecino), candidato)))
        if clave in edge_rows:
            edge_rows[clave].update({"color": hex_to_rgba(SECOND_HOP_EDGE_COLOR, 0.34), "width": 0.8, "dashes": True})


def _atenuar_aristas(edge_rows: EdgeRows) -> None:
    """Reduce el peso visual de todas las aristas para destacar los nodos."""
    for clave in edge_rows:
        edge_rows[clave].update({"color": hex_to_rgba(NEUTRAL_EDGE_COLOR, 0.08), "width": 0.28})


def _resaltar_aristas_comunidades(edge_rows: EdgeRows, grafo, estrategia, ego_id, vecinos_validos) -> None:
    """Atenúa todas las aristas y colorea las intragrupo según su comunidad."""
    _atenuar_aristas(edge_rows)
    comunidades_dict, colores_comunidades = colores_comunidades_vecinos(estrategia, ego_id, vecinos_validos)
    for u, v in grafo.edges():
        u_int, v_int = int(u), int(v)
        if u_int in comunidades_dict and v_int in comunidades_dict and comunidades_dict[u_int] == comunidades_dict[v_int]:
            clave = tuple(sorted((u_int, v_int)))
            if clave in edge_rows:
                color_c = colores_comunidades.get(comunidades_dict[u_int], ACTIVE_EDGE_COLOR)
                edge_rows[clave].update({"color": hex_to_rgba(color_c, 0.48), "width": 0.95})


def aplicar_estilos_aristas_fase2(
    edge_rows: EdgeRows,
    grafo,
    strategy_key: str,
    ego_id: int,
    vecinos_validos,
    atributos_amigos,
    estrategia,
    resaltar_aristas_ego: bool,
) -> None:
    """Aplica el estilizado de aristas de fase 2 según la estrategia activa."""
    if resaltar_aristas_ego:
        _resaltar_aristas_ego(edge_rows, ego_id, vecinos_validos)
    if strategy_key == "transitividad":
        _resaltar_aristas_transitividad(edge_rows, grafo, ego_id, vecinos_validos, atributos_amigos)
    elif strategy_key in ("puentes", "prestigio", "similitud"):
        _atenuar_aristas(edge_rows)
    elif strategy_key == "comunidades":
        _resaltar_aristas_comunidades(edge_rows, grafo, estrategia, ego_id, vecinos_validos)
