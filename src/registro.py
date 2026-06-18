"""Registro único de estrategias (clave, etiqueta y clase), fuera de la interfaz para que el batch no dependa de la UI."""

from __future__ import annotations

from typing import Iterator

from src.estrategias.base import EstrategiaInferencia
from src.estrategias.circulos import EstrategiaCirculos
from src.estrategias.comunidades import EstrategiaComunidades
from src.estrategias.correlacion import EstrategiaCorrelacion
from src.estrategias.puentes import EstrategiaPuentes
from src.estrategias.similitud import EstrategiaSimilitud
from src.estrategias.transitividad import EstrategiaTransitividad
from src.estrategias.vecinos import EstrategiaVecinos
from src.estrategias.prestigio import EstrategiaPrestigio

CLAVES_EXCLUIDAS_BATCH: frozenset[str] = frozenset({"base"})

STRATEGY_REGISTRY: list[dict[str, object]] = [
    {"key": "base", "label": "Homofilia plana", "class": EstrategiaInferencia},
    {"key": "vecinos", "label": "Vecinos directos", "class": EstrategiaVecinos},
    {"key": "circulos", "label": "Círculos sociales", "class": EstrategiaCirculos},
    {"key": "correlacion", "label": "Correlación de atributos", "class": EstrategiaCorrelacion},
    {"key": "puentes", "label": "Puentes de la red", "class": EstrategiaPuentes},
    {"key": "transitividad", "label": "Transitividad", "class": EstrategiaTransitividad},
    {"key": "similitud", "label": "Similitud estructural", "class": EstrategiaSimilitud},
    {"key": "comunidades", "label": "Comunidades locales", "class": EstrategiaComunidades},
    {"key": "prestigio", "label": "Prestigio social", "class": EstrategiaPrestigio},
]


def estrategias_para_batch() -> Iterator[tuple[str, type[EstrategiaInferencia]]]:
    """Estrategias evaluadas por el batch científico (excluye ``base``)."""
    for item in STRATEGY_REGISTRY:
        if item["key"] not in CLAVES_EXCLUIDAS_BATCH:
            yield str(item["key"]), item["class"]


def estrategias_para_dashboard(incluir_base: bool = False) -> list[dict[str, object]]:
    """Lista de estrategias expuesta en el selector del dashboard."""
    if incluir_base:
        return list(STRATEGY_REGISTRY)
    return [item for item in STRATEGY_REGISTRY if item["key"] not in CLAVES_EXCLUIDAS_BATCH]


def construir_estrategia(
    strategy_key: str,
    grafo,
    atributos_amigos,
    mapeador,
) -> EstrategiaInferencia:
    """Instancia la clase de estrategia registrada o la base como respaldo."""
    mapping = {item["key"]: item["class"] for item in STRATEGY_REGISTRY}
    strategy_cls = mapping.get(strategy_key, EstrategiaInferencia)
    return strategy_cls(grafo, atributos_amigos, mapeador)
