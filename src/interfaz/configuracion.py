"""Registro único de estrategias y constantes visuales compartidas por grafo y dashboard.

La estrategia ``base`` se excluye de la ejecución comparativa porque duplica la homofilia de ``vecinos``.
"""

from __future__ import annotations

from src.estrategias.base import EstrategiaInferencia
from src.estrategias.circulos import EstrategiaCirculos
from src.estrategias.comunidades import EstrategiaComunidades
from src.estrategias.correlacion import EstrategiaCorrelacion
from src.estrategias.puentes import EstrategiaPuentes
from src.estrategias.similitud import EstrategiaSimilitud
from src.estrategias.transitividad import EstrategiaTransitividad
from src.estrategias.vecinos import EstrategiaVecinos
from src.estrategias.prestigio import EstrategiaPrestigio

NEUTRAL_NODE_COLOR = "#64748b"
NEUTRAL_OTHER_COLOR = "#334155"
NEUTRAL_EDGE_COLOR = "#475569"
ACTIVE_EDGE_COLOR = "#facc15"
EGO_UNKNOWN_COLOR = "#94a3b8"
EGO_WARNING_COLOR = "#6b7280"
BASE_NODE_SIZE = 3
EGO_SIZE = 7

CLAVES_EXCLUIDAS_BATCH = frozenset({"base"})

STRATEGY_REGISTRY = [
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


def estrategias_para_batch():
    """Estrategias evaluadas por main.py (sin base)."""
    for item in STRATEGY_REGISTRY:
        if item["key"] not in CLAVES_EXCLUIDAS_BATCH:
            yield str(item["key"]), item["class"]


def estrategias_para_dashboard(incluir_base: bool = False):
    """Lista de estrategias expuesta en el selector del dashboard."""
    if incluir_base:
        return list(STRATEGY_REGISTRY)
    return [item for item in STRATEGY_REGISTRY if item["key"] not in CLAVES_EXCLUIDAS_BATCH]
