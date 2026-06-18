"""Tarjetas de leyenda, paletas y utilidades de color compartidas por grafo y plugins."""

from __future__ import annotations

from typing import Any

from src.interfaz.configuracion import ACTIVE_EDGE_COLOR, NEUTRAL_EDGE_COLOR
from src.interfaz.inferencia import generar_paleta

EGO_LEGEND_COLOR = "#6b7280"
ACTIVE_NODE_LEGEND_COLOR = "#38bdf8"
SECOND_HOP_LEGEND_COLOR = "#93c5fd"
BRIDGE_LEGEND_COLOR = "#facc15"
NEIGHBOR_LEGEND_COLOR = "#64748b"
NEUTRAL_LEGEND_COLOR = "#334155"
SECOND_HOP_EDGE_COLOR = "#60a5fa"
PRESTIGIO_COLOR_BAJO = "#38bdf8"
PRESTIGIO_COLOR_ALTO = "#8b5cf6"


def tarjeta_leyenda(color: str, texto: str) -> dict[str, str]:
    """Diccionario {texto, color} para una entrada de leyenda."""
    return {"texto": texto.strip(), "color": color}


def tarjeta_ego() -> dict[str, str]:
    """Tarjeta estándar del usuario objetivo."""
    return tarjeta_leyenda(EGO_LEGEND_COLOR, "Usuario objetivo")


def tarjetas_vecindad_directa() -> list[dict[str, str]]:
    """Leyenda de fase 2 para estrategias de vecindad directa."""
    return [
        tarjeta_ego(),
        tarjeta_leyenda(ACTIVE_NODE_LEGEND_COLOR, "Vecinos directos"),
    ]


def tarjetas_aristas_vecinos() -> list[dict[str, str]]:
    """Leyenda de aristas para enlaces ego–vecino."""
    return [
        tarjeta_leyenda(ACTIVE_EDGE_COLOR, "Enlace ego–vecino"),
        tarjeta_leyenda(NEUTRAL_EDGE_COLOR, "Demás enlaces"),
    ]


def tarjetas_aristas_transitividad() -> list[dict[str, str]]:
    """Leyenda de aristas para primer y segundo salto."""
    return [
        tarjeta_leyenda(ACTIVE_EDGE_COLOR, "Primer salto"),
        tarjeta_leyenda(SECOND_HOP_EDGE_COLOR, "Segundo salto (punteado)"),
    ]


def tarjetas_aristas_comunidades() -> list[dict[str, str]]:
    """Leyenda de aristas intragrupo en comunidades."""
    return [tarjeta_leyenda(ACTIVE_EDGE_COLOR, "Enlaces intragrupo")]


def tarjetas_puentes() -> list[dict[str, str]]:
    """Leyenda de nodos puente e intermediación alta."""
    return [
        tarjeta_ego(),
        tarjeta_leyenda(BRIDGE_LEGEND_COLOR, "Intermediación alta"),
        tarjeta_leyenda(NEIGHBOR_LEGEND_COLOR, "Vecino directo"),
    ]


def tarjetas_transitividad() -> list[dict[str, str]]:
    """Leyenda de nodos a uno y dos saltos."""
    return [
        tarjeta_ego(),
        tarjeta_leyenda(ACTIVE_NODE_LEGEND_COLOR, "Primer salto"),
        tarjeta_leyenda(SECOND_HOP_LEGEND_COLOR, "Segundo salto"),
    ]


def tarjetas_similitud() -> list[dict[str, str]]:
    """Leyenda de vecinos directos y alta similitud estructural."""
    return [
        tarjeta_ego(),
        tarjeta_leyenda(ACTIVE_NODE_LEGEND_COLOR, "Vecinos directos"),
        tarjeta_leyenda(BRIDGE_LEGEND_COLOR, "Alta similitud estructural"),
    ]


def tarjetas_prestigio() -> list[dict[str, str]]:
    """Leyenda con escala de prestigio bajo, medio y alto."""
    escala = (
        ("Bajo prestigio", 0.0),
        ("Prestigio medio", 0.5),
        ("Alto prestigio", 1.0),
    )
    return [
        tarjeta_ego(),
        *[
            tarjeta_leyenda(interpolar_color(PRESTIGIO_COLOR_BAJO, PRESTIGIO_COLOR_ALTO, t), titulo)
            for titulo, t in escala
        ],
    ]


def respuesta_leyenda(
    titulo: str,
    cards: list[dict[str, str]],
    caption: str | None = None,
    edge_cards: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Estructura unificada que consumen grafo.py y el simulador."""
    return {
        "title": titulo,
        "cards": cards,
        "caption": caption,
        "edge_cards": edge_cards or [],
    }


def interpolar_color(c1_hex: str, c2_hex: str, t: float) -> str:
    """Mezcla lineal entre dos colores hex (t ∈ [0, 1])."""
    t = max(0.0, min(1.0, float(t)))
    c1 = c1_hex.lstrip("#")
    c2 = c2_hex.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def paleta_circulos(circulos: dict[str, list[int]] | None) -> dict[str, str]:
    """Mapa nombre de círculo → color para la estrategia de círculos."""
    if not circulos:
        return {}
    nombres = list(circulos.keys())
    colores = generar_paleta(len(nombres))
    return {str(nombre): colores[pos] for pos, nombre in enumerate(nombres)}


def circulo_visible_para_vecino(
    nodo_int: int,
    vecinos_set: set[int],
    circulos: dict[str, list[int]] | None,
    colores_circulos: dict[str, str],
) -> tuple[str, str] | None:
    """Color y nombre del círculo si el vecino pertenece a alguno."""
    if nodo_int not in vecinos_set or not circulos:
        return None
    for circle_name, members in circulos.items():
        if nodo_int in {int(member) for member in members}:
            return colores_circulos.get(str(circle_name), ACTIVE_NODE_LEGEND_COLOR), str(circle_name)
    return None


def colores_comunidades_vecinos(estrategia, ego_id, vecinos_validos) -> tuple[dict[int, Any], dict[Any, str]]:
    """Asignación nodo → comunidad y comunidad → color vía Louvain."""
    if estrategia is None or ego_id is None:
        return {}, {}
    comunidades = estrategia.comunidades_locales(ego_id, list(vecinos_validos)) or {}
    if not comunidades:
        return {}, {}
    cids = sorted(set(comunidades.values()))
    palette = generar_paleta(len(cids))
    return comunidades, {cid: palette[pos] for pos, cid in enumerate(cids)}


def tarjetas_circulos_fase2(
    circulos: dict[str, list[int]],
    circle_palette: list[str],
    vecinos_validos: list[int],
) -> list[dict[str, str]]:
    """Leyenda de fase 2 con círculos que tienen vecinos visibles."""
    vecinos_set = {int(vecino) for vecino in vecinos_validos}
    cards = [tarjeta_ego()]
    for color_index, (nombre, miembros) in enumerate(circulos.items()):
        if not any(int(m) in vecinos_set for m in miembros):
            continue
        color = circle_palette[color_index % len(circle_palette)] if circle_palette else "#8b5cf6"
        cards.append(tarjeta_leyenda(color, str(nombre)))
    return cards


def tarjetas_comunidades_fase2(estrategia, ego_id, vecinos_validos) -> list[dict[str, str]]:
    """Leyenda de fase 2: ego y cada comunidad visible entre los vecinos directos."""
    comunidades_dict, colores = colores_comunidades_vecinos(estrategia, ego_id, vecinos_validos)
    cards = [tarjeta_ego()]
    if not colores:
        cards.append(tarjeta_leyenda(NEUTRAL_LEGEND_COLOR, "Sin agrupación detectada"))
        return cards

    visibles = sorted(
        {comunidades_dict[int(vecino)] for vecino in vecinos_validos if int(vecino) in comunidades_dict}
    )
    for cid in visibles:
        cards.append(tarjeta_leyenda(colores[cid], f"Comunidad {cid}"))
    return cards
