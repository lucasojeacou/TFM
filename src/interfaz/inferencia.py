"""Utilidades de atributos, paletas de color y construcción de estrategias para la UI."""

from __future__ import annotations

import colorsys
import re


def indices_rango(mapeador, category_key: str) -> list[int]:
    """Índices enteros consecutivos de una categoría semántica."""
    rango = mapeador.esquema_rangos[category_key]
    return list(range(rango["inicio"], rango["fin"] + 1))


def generar_paleta(count: int) -> list[str]:
    """Colores hex equidistantes en HLS para distinguir valores de una categoría."""
    if count <= 0:
        return []
    if count == 1:
        return ["#3b82f6"]
    palette = []
    for index in range(count):
        hue = index / count
        saturation = 0.72
        lightness = 0.52
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
        palette.append(f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}")
    return palette


def paleta_categoria(mapeador, category_key: str) -> dict[int, str]:
    """Mapa índice de atributo → color para una categoría."""
    indices = indices_rango(mapeador, category_key)
    colors = generar_paleta(len(indices))
    return {idx: colors[pos] for pos, idx in enumerate(indices)}


def etiqueta_atributo(mapeador, idx: int | str) -> str:
    """Nombre legible del atributo con separadores adaptados a la UI."""
    texto = mapeador.indices_mapeados.get(idx, str(idx)) if isinstance(idx, int) else str(idx)
    return texto.replace(";", " / ")


def valor_feat_dataset(mapeador, idx: int) -> str:
    """Texto del featname SNAP sin el prefijo de categoría."""
    raw = str(mapeador.indices_mapeados.get(int(idx), "")).strip()
    clave = "anonymized feature"
    pos = raw.lower().find(clave)
    if pos >= 0:
        return raw[pos:].strip()
    partes = [parte.strip() for parte in raw.split(";") if parte.strip()]
    return partes[-1] if partes else raw


def numero_feat_dataset(mapeador, idx: int) -> str:
    """Número del featname SNAP (p. ej. '265' en 'anonymized feature 265')."""
    texto = valor_feat_dataset(mapeador, int(idx))
    numeros = re.findall(r"\d+", texto)
    if numeros:
        return numeros[-1]
    return texto


def resolver_indice_inferido_ego(
    mapeador,
    category_key: str,
    elegido_por_categoria: dict[str, tuple[object, float]],
    inferidos: list[int] | None,
) -> int | None:
    """Atributo ganador de la categoría para colorear el ego en fase 4."""
    indices_cat = indices_rango(mapeador, category_key)
    predichos = [
        idx
        for idx in indices_cat
        if inferidos is not None and idx < len(inferidos) and int(inferidos[idx]) == 1
    ]
    if not predichos:
        return None
    winner_idx, _ = elegido_por_categoria.get(category_key, (None, 0.0))
    if winner_idx is not None:
        return int(winner_idx)
    return predichos[0]


def indices_leyenda_categoria(
    mapeador,
    category_key: str,
    atributos_amigos: dict[int, list[int]],
    vecinos_validos,
    phase: int,
    inferidos: list[int] | None = None,
    elegido_por_categoria: dict[str, tuple[object, float]] | None = None,
) -> list[int]:
    """Atributos que deben aparecer en la leyenda de fases 3-4."""
    indices = indices_coloreados_en_grafo(mapeador, category_key, atributos_amigos, vecinos_validos)
    if int(phase) >= 4:
        idx_ego = resolver_indice_inferido_ego(
            mapeador,
            category_key,
            elegido_por_categoria or {},
            inferidos,
        )
        if idx_ego is not None and idx_ego not in indices:
            orden = indices_rango(mapeador, category_key)
            indices = sorted(set(indices) | {idx_ego}, key=lambda i: orden.index(i))
    return indices


def indice_atributo_dominante_nodo(mapeador, category_key: str, values) -> int | None:
    """Primer atributo activo de la categoría en el vector de un nodo."""
    for idx in indices_rango(mapeador, category_key):
        if idx < len(values) and int(values[idx]) == 1:
            return idx
    return None


def indices_coloreados_en_grafo(mapeador, category_key: str, atributos_amigos: dict[int, list[int]], vecinos_validos) -> list[int]:
    """Atributos que determinan el color de al menos un vecino visible."""
    visibles = set()
    for vecino in vecinos_validos:
        values = atributos_amigos.get(int(vecino))
        if not values:
            continue
        dominante = indice_atributo_dominante_nodo(mapeador, category_key, values)
        if dominante is not None:
            visibles.add(dominante)
    orden = indices_rango(mapeador, category_key)
    return [idx for idx in orden if idx in visibles]


def formato_atributos_activados(mapeador, category_key: str, vector: list[int]) -> str:
    """Lista legible de featnames activos en una categoría ('Ninguno' si vacío)."""
    indices = indices_rango(mapeador, category_key)
    activos = [idx for idx in indices if idx < len(vector) and int(vector[idx]) == 1]
    if not activos:
        return "Ninguno"
    return ", ".join(numero_feat_dataset(mapeador, idx) for idx in activos)
