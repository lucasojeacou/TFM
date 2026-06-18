"""Utilidades de test: ejecución unificada de estrategias con el registro oficial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

from src.estrategias.base import EstrategiaInferencia
from src.registro import STRATEGY_REGISTRY

ESTRATEGIAS = [item["class"] for item in STRATEGY_REGISTRY]


@dataclass(slots=True)
class ResultadoInferencia:
    """Resultado de la ejecución de una estrategia de inferencia."""

    inferidos: list[int]
    elegido: dict[str, tuple[Any, float]]
    xai: dict[int, dict[int, float]]
    mapeador: Any


def ejecutar_estrategia(
    clase_estrategia: Type[EstrategiaInferencia],
    grafo: Any,
    atributos: Any,
    circulos: Any,
    mapeador: Any,
    nodo_id: int = 0,
) -> ResultadoInferencia:
    """Ejecuta una estrategia y normaliza la salida de inferencia y XAI."""
    estrategia = clase_estrategia(grafo, atributos, mapeador)

    inferidos, elegido_por_categoria, xai = estrategia.inferir(
        nodo_id,
        circulos=circulos,
    )

    xai_raw = xai or {}

    xai_normalizado = {
        int(nodo): {
            int(origen): float(peso)
            for origen, peso in contribuciones.items()
        }
        for nodo, contribuciones in xai_raw.items()
    }

    return ResultadoInferencia(
        inferidos=list(inferidos),
        elegido=elegido_por_categoria,
        xai=xai_normalizado,
        mapeador=mapeador,
    )
