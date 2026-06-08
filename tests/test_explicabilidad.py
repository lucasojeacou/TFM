"""Tests de explicabilidad (XAI) para transitividad y similitud estructural."""

from __future__ import annotations

from src.estrategias.similitud import EstrategiaSimilitud
from src.estrategias.transitividad import EstrategiaTransitividad


def test_transitividad_xai_incluye_vecinos_de_segundo_salto(datos_prueba):
    """La XAI de transitividad incluye nodos a dos saltos con contribución positiva."""
    grafo, atributos, circulos, mapeador = datos_prueba
    estrategia = EstrategiaTransitividad(grafo, atributos, mapeador)

    inferidos, _, xai = estrategia.inferir(0, circulos=circulos)

    assert inferidos[3] == 1
    assert 5 in xai[3]
    assert 5 not in estrategia.obtener_vecinos_validos(0)


def test_similitud_xai_conserva_contribuciones_positivas(datos_prueba):
    """La XAI de similitud conserva solo contribuciones estrictamente positivas."""
    grafo, atributos, circulos, mapeador = datos_prueba
    estrategia = EstrategiaSimilitud(grafo, atributos, mapeador)

    inferidos, _, xai = estrategia.inferir(0, circulos=circulos)

    assert inferidos[3] == 1
    assert xai[3]
    assert all(peso > 0 for peso in xai[3].values())
