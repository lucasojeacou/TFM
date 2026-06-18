"""Políticas de decisión por categoría: clasificación de ruido y binarización de puntuaciones."""

from __future__ import annotations

import numpy as np
from typing import Any

from .umbrales import (
    PENALIZACION_FRECUENCIA,
    delta_minima_single_index,
    umbral_dinamico_categoria,
)


def clasificar_politica_categoria(
    categoria: str,
    rango: dict[str, int],
    mapeador: Any,
    atributos_amigos: dict[int, list[int]],
) -> dict[str, Any]:
    """Elige modo ruido, valor único o multi-valor según densidad y dispersión de la categoría."""
    cat_normalizada = str(categoria).strip().lower()
    palabras_prohibidas = {"birthday", "name"}
    if cat_normalizada in palabras_prohibidas or any(p in cat_normalizada for p in palabras_prohibidas):
        return {"modo": "ruido", "max_k": 0}

    inicio, fin = int(rango["inicio"]), int(rango["fin"])
    lista_attrs = list(atributos_amigos.values())
    if not lista_attrs:
        return {"modo": "ruido", "max_k": 0}
    try:
        matriz_base = np.asarray(lista_attrs, dtype=float)
    except ValueError:
        return {"modo": "ruido", "max_k": 0}
    if matriz_base.ndim != 2 or matriz_base.size == 0:
        return {"modo": "ruido", "max_k": 0}
    columnas = matriz_base.shape[1]
    if inicio < 0 or fin < inicio or inicio >= columnas:
        return {"modo": "ruido", "max_k": 0}
    fin_seguro = min(fin, columnas - 1)
    matriz_cat = matriz_base[:, inicio : fin_seguro + 1]
    if matriz_cat.size == 0 or np.max(np.sum(matriz_cat, axis=0)) <= 2:
        return {"modo": "ruido", "max_k": 0}

    activos = np.sum(matriz_cat, axis=1)
    usuarios_validos = activos[activos > 0]
    if usuarios_validos.size == 0:
        return {"modo": "ruido", "max_k": 0}

    tamano_rango = fin_seguro - inicio + 1
    umbral = umbral_dinamico_categoria(tamano_rango)

    media_activos = np.mean(usuarios_validos)
    if media_activos == 1.0:
        return {"modo": "single_index", "umbral": umbral, "max_k": 1}

    max_k = int(np.ceil(media_activos)) + 1
    if tamano_rango <= 3:
        max_k = 1
    else:
        max_k = min(max_k, tamano_rango - 1)
    return {"modo": "multi_k", "umbral": umbral, "max_k": max_k}


def seleccionar_por_categoria(
    puntuaciones: dict[int, float],
    mapeador: Any,
    atributos_amigos: dict[int, list[int]],
) -> tuple[list[int], dict[str, tuple[Any, float]]]:
    """Convierte puntuaciones continuas en vector binario por categoría semántica."""
    if mapeador is None or not hasattr(mapeador, "esquema_rangos"):
        return [], {}

    total_attrs = len(mapeador.indices_mapeados) if hasattr(mapeador, "indices_mapeados") else (max(puntuaciones.keys()) + 1 if puntuaciones else 0)
    inferidos = [0] * total_attrs
    elegido_por_categoria = {}

    puntuaciones_penalizadas = {}
    if atributos_amigos:
        lista_amigos = list(atributos_amigos.values())
        if lista_amigos:
            try:
                matriz_amigos = np.asarray(lista_amigos, dtype=float)
            except ValueError:
                matriz_amigos = np.empty((0, 0))
            if matriz_amigos.ndim == 2 and matriz_amigos.size > 0:
                frecuencias_globales = np.mean(matriz_amigos, axis=0)
                for idx, score in puntuaciones.items():
                    if 0 <= idx < len(frecuencias_globales):
                        frec_global = float(frecuencias_globales[idx])
                        puntuaciones_penalizadas[idx] = score * (1.0 - PENALIZACION_FRECUENCIA * frec_global)
                    else:
                        puntuaciones_penalizadas[idx] = score
            else:
                puntuaciones_penalizadas = dict(puntuaciones)
        else:
            puntuaciones_penalizadas = dict(puntuaciones)
    else:
        puntuaciones_penalizadas = dict(puntuaciones)

    for cat, rango in mapeador.esquema_rangos.items():
        politica = clasificar_politica_categoria(cat, rango, mapeador, atributos_amigos)
        inicio, fin = int(rango["inicio"]), int(rango["fin"])

        if politica["modo"] == "single_index":
            for i in range(inicio, fin + 1):
                if 0 <= i < len(inferidos):
                    inferidos[i] = 0
        if politica["modo"] == "ruido":
            continue

        indices_cat = [i for i in range(inicio, fin + 1) if 0 <= i < len(inferidos) and i in puntuaciones_penalizadas]
        if not indices_cat:
            continue

        ordenados = sorted(indices_cat, key=lambda idx: puntuaciones_penalizadas.get(idx, 0.0), reverse=True)
        mejor_idx = ordenados[0]
        mejor_score = puntuaciones_penalizadas.get(mejor_idx, 0.0)

        if politica["modo"] == "single_index":
            segundo_score = puntuaciones_penalizadas.get(ordenados[1], 0.0) if len(ordenados) > 1 else 0.0
            delta = delta_minima_single_index(len(atributos_amigos))

            if mejor_score >= politica["umbral"] and (mejor_score - segundo_score) >= delta:
                inferidos[mejor_idx] = 1
                elegido_por_categoria[cat] = (mejor_idx, mejor_score)
            else:
                elegido_por_categoria[cat] = (None, mejor_score)
            continue

        seleccionados = [i for i in ordenados if puntuaciones_penalizadas.get(i, 0.0) >= politica["umbral"]][:politica["max_k"]]
        if seleccionados:
            for i in seleccionados:
                if 0 <= i < len(inferidos):
                    inferidos[i] = 1
            elegido_por_categoria[cat] = (seleccionados[0], puntuaciones_penalizadas.get(seleccionados[0], 0.0))
        else:
            elegido_por_categoria[cat] = (None, mejor_score)

    return inferidos, elegido_por_categoria
