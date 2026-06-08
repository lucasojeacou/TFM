"""Orquestación de inferencia para el dashboard: snapshot, resultado y tabla de auditoría."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.nucleo.evaluacion import calcular_metricas_binarias
from src.nucleo.politicas_semanticas import clasificar_politica_categoria
from src.interfaz.configuracion import STRATEGY_REGISTRY
from src.interfaz.datos import cargar_snapshot
from src.interfaz.inferencia import (
    construir_estrategia,
    formato_atributos_activados,
    indices_rango,
    numero_feat_dataset,
)


@dataclass(frozen=True)
class InferenceSnapshot:
    """Datos inmutables de una ego-red cargada desde SNAP."""

    grafo: object
    atributos_amigos: Dict[int, List[int]]
    perfil_ego: List[int]
    circulos: Dict[str, List[int]]
    mapeador: object
    posiciones: Dict[int, Tuple[float, float]]


@dataclass(frozen=True)
class InferenceResult:
    """Salida de una inferencia: vector binario, ganadores por categoría y explicabilidad."""

    inferidos: List[int]
    elegido_por_categoria: Dict[str, Tuple[object, float]]
    contribuciones_xai: Dict[int, Dict[int, float]]
    metricas: Dict[str, float]


def cargar_instantanea(ruta_datos: Path, ego_id: int) -> InferenceSnapshot:
    """Carga grafo, perfiles y layout en un contenedor inmutable."""
    grafo, atributos, perfil, circulos, mapeador, posiciones = cargar_snapshot(str(ruta_datos), int(ego_id))
    return InferenceSnapshot(grafo, atributos, perfil, circulos, mapeador, posiciones)


def ejecutar_inferencia(
    snapshot: InferenceSnapshot,
    ego_id: int,
    strategy_key: str,
) -> InferenceResult:
    """Ejecuta la estrategia indicada y calcula métricas frente al perfil real."""
    estrategia = construir_estrategia(STRATEGY_REGISTRY, strategy_key, snapshot.grafo, snapshot.atributos_amigos, snapshot.mapeador)
    inferidos, elegido_por_categoria, xai = estrategia.inferir(int(ego_id), circulos=snapshot.circulos)
    metricas = calcular_metricas_binarias(snapshot.perfil_ego, inferidos)
    return InferenceResult(list(inferidos), dict(elegido_por_categoria), dict(xai or {}), metricas)


def categorias_evaluables(
    mapeador,
    atributos_amigos: Dict[int, List[int]],
) -> List[str]:
    """Categorías con política distinta de ruido (aptas para auditoría)."""
    categorias = []
    for categoria, rango in sorted(mapeador.esquema_rangos.items(), key=lambda item: item[1]["inicio"]):
        politica = clasificar_politica_categoria(categoria, rango, mapeador, atributos_amigos)
        if politica.get("modo") != "ruido":
            categorias.append(categoria)
    return categorias


def activos_en_rango(vector: Iterable[int], indices: Iterable[int]) -> List[int]:
    """Índices con valor 1 dentro del rango dado."""
    activos = []
    for idx in indices:
        try:
            if int(vector[idx]) == 1:
                activos.append(idx)
        except IndexError:
            continue
    return activos


def estado_categoria(reales: List[int], predichos: List[int]) -> str:
    """Etiqueta de acierto: Correcto, Parcial, Fallido o Sin datos."""
    reales_set = set(reales)
    predichos_set = set(predichos)
    if not reales_set and not predichos_set:
        return "Sin datos"
    if reales_set == predichos_set:
        return "Correcto"
    if not reales_set.isdisjoint(predichos_set):
        return "Parcial"
    return "Fallido"


def construir_auditoria(snapshot: InferenceSnapshot, resultado: InferenceResult) -> pd.DataFrame:
    """Tabla comparativa real vs inferido por categoría evaluable."""
    filas = []
    for categoria in categorias_evaluables(snapshot.mapeador, snapshot.atributos_amigos):
        indices = indices_rango(snapshot.mapeador, categoria)
        reales = activos_en_rango(snapshot.perfil_ego, indices)
        predichos = activos_en_rango(resultado.inferidos, indices)
        ganador, confianza = resultado.elegido_por_categoria.get(categoria, (None, 0.0))
        filas.append(
            {
                "categoria": categoria,
                "Categoria": categoria.replace(";", " / "),
                "Real": formato_atributos_activados(snapshot.mapeador, categoria, snapshot.perfil_ego),
                "Inferido": formato_atributos_activados(snapshot.mapeador, categoria, resultado.inferidos),
                "Estado": estado_categoria(reales, predichos),
                "Ganador": numero_feat_dataset(snapshot.mapeador, int(ganador)) if ganador is not None else "Sin ganador",
                "ganador_idx": int(ganador) if ganador is not None else None,
                "Confianza": float(confianza),
                "real_indices": reales,
                "pred_indices": predichos,
            }
        )
    return pd.DataFrame(filas)


def top_contribuyentes(resultado: InferenceResult, atributo_id: int | None, limite: int | None = None) -> pd.DataFrame:
    """Vecinos ordenados por peso de explicabilidad para un atributo inferido."""
    if atributo_id is None:
        return pd.DataFrame(columns=["Nodo", "Peso", "Porcentaje"])
    contribuciones = resultado.contribuciones_xai.get(int(atributo_id), {})
    filas = sorted(((int(nodo), float(peso)) for nodo, peso in contribuciones.items()), key=lambda item: item[1], reverse=True)
    total = sum(peso for _, peso in filas) or 1.0
    if limite is not None:
        filas = filas[:limite]
    return pd.DataFrame(
        [
            {"Nodo": f"Nodo {nodo}", "Peso": round(peso, 6), "Porcentaje": round((100.0 * peso) / total, 2)}
            for nodo, peso in filas
        ]
    )


def leer_csv_seguro(ruta: Path) -> pd.DataFrame | None:
    """Lee un CSV si existe; devuelve None en caso contrario."""
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)
