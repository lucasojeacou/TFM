import os
import sys
import re
import numpy as np
from typing import Any, Dict, List, Tuple

# Tokeniza un texto extrayendo términos alfanuméricos limpios en minúsculas.
def _tokenizar_semantica(texto: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", texto.lower()) if token]

# Clasifica dinámicamente una categoría analizando la distribución estadística en la matriz de amigos.
def clasificar_politica_categoria(
    categoria: str,
    rango: Dict[str, int],
    mapeador: Any,
    atributos_amigos: Dict[int, List[int]],
    umbral_bajo: float,
    umbral_alto: float,
) -> Dict[str, Any]:
    cat_normalizada = str(categoria).strip().lower()
    
    # FILTRO SEMÁNTICO CRÍTICO: Forzamos a ruido datos de identidad, locales o campos caóticos no inferibles topológicamente
    palabras_prohibidas = {"birthday", "name"}
    if cat_normalizada in palabras_prohibidas or any(p in cat_normalizada for p in palabras_prohibidas):
        return {"modo": "ruido", "umbral": umbral_alto, "max_k": 0}

    inicio, fin = rango["inicio"], rango["fin"]
    
    # Transformamos en matriz de NumPy y extraemos en bloque el rango de la categoría
    matriz_cat = np.array(list(atributos_amigos.values()))[:, inicio : fin + 1]
    
    # Filtro de Coincidencia Estructural Mínima: Si el bit más repetido lo tienen 2 o menos amigos, es ruido disperso
    if matriz_cat.size == 0 or np.max(np.sum(matriz_cat, axis=0)) <= 2:
        return {"modo": "ruido", "umbral": umbral_alto, "max_k": 0}

    # Filtro de Índice Único: Contamos cuántos atributos activa cada usuario que rellenó el campo
    activos = np.sum(matriz_cat, axis=1)
    usuarios_validos = activos[activos > 0]

    if usuarios_validos.size == 0:
        return {"modo": "ruido", "umbral": umbral_alto, "max_k": 0}

    # Calculamos la media real de opciones que rellena un usuario en esta red específica
    media_activos = np.mean(usuarios_validos)

    # Si la media exacta de bits encendidos es 1.0, la categoría es de exclusión mutua (ej: género)
    if media_activos == 1.0:
        return {"modo": "single_index", "umbral": umbral_bajo, "max_k": 1}

    # NUEVA LÓGICA MULTI-VALOR ADAPTATIVA: El límite real es la media de los amigos + 1 (redondeado hacia arriba)
    max_k = int(np.ceil(media_activos)) + 1
    
    # Control defensivo de seguridad: En rangos pequeños o densos, impedimos predecir el 100% de las opciones
    tamano_rango = fin - inicio + 1
    if tamano_rango <= 3:
        max_k = 1
    else:
        max_k = min(max_k, tamano_rango - 1)

    return {"modo": "multi_k", "umbral": umbral_bajo, "max_k": max_k}

# Aplica la política binarizadora dinámica protegiendo las exclusiones de índices únicos.
def seleccionar_por_categoria(
    puntuaciones: Dict[int, float],
    mapeador: Any,
    atributos_amigos: Dict[int, List[int]],
    umbral_bajo: float = 0.12,
    umbral_alto: float = 0.22,
) -> Tuple[List[int], Dict[str, Tuple[Any, float]]]:
    if mapeador is None or not hasattr(mapeador, "esquema_rangos"):
        return [], {}

    total_attrs = len(mapeador.indices_mapeados) if hasattr(mapeador, "indices_mapeados") else max(puntuaciones.keys()) + 1
    inferidos = [0] * total_attrs
    elegido_por_categoria = {}

    for cat, rango in mapeador.esquema_rangos.items():
        politica = clasificar_politica_categoria(cat, rango, mapeador, atributos_amigos, umbral_bajo, umbral_alto)
        inicio, fin = rango["inicio"], rango["fin"]
        
        # Reseteo estricto para categorías exclusivas
        if politica["modo"] == "single_index":
            for i in range(inicio, fin + 1): inferidos[i] = 0
        if politica["modo"] == "ruido": 
            continue

        indices_cat = [i for i in range(inicio, fin + 1) if i in puntuaciones]
        if not indices_cat: 
            continue

        ordenados = sorted(indices_cat, key=lambda idx: puntuaciones.get(idx, 0.0), reverse=True)
        mejor_idx = ordenados[0]
        mejor_score = puntuaciones.get(mejor_idx, 0.0)

        if politica["modo"] == "single_index":
            segundo_score = puntuaciones.get(ordenados[1], 0.0) if len(ordenados) > 1 else 0.0
            # Desempate competitivo por margen delta mínimo de 0.05
            if mejor_score >= politica["umbral"] and (mejor_score - segundo_score) >= 0.05:
                inferidos[mejor_idx] = 1
                elegido_por_categoria[cat] = (mejor_idx, mejor_score)
            else:
                elegido_por_categoria[cat] = (None, mejor_score)
            continue

        # Inferencia multi-valor adaptativa filtrada por el max_k basado en el vecindario
        seleccionados = [i for i in ordenados if puntuaciones.get(i, 0.0) >= politica["umbral"]][:politica["max_k"]]
        if seleccionados:
            for i in seleccionados: inferidos[i] = 1
            elegido_por_categoria[cat] = (seleccionados[0], puntuaciones.get(seleccionados[0], 0.0))
        else:
            elegido_por_categoria[cat] = (None, mejor_score)

    return inferidos, elegido_por_categoria