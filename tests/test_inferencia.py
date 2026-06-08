"""Tests del pipeline de inferencia: vecinos válidos y las 9 estrategias del registro."""

from __future__ import annotations

import pytest

from src.estrategias.base import EstrategiaInferencia
from tests.utilidades.inferencia import ESTRATEGIAS, ejecutar_estrategia


def test_obtener_vecinos_universal(datos_prueba):
    """Excluye vecinos sin perfil de atributos (p. ej. nodo 14)."""
    G, atributos, circulos, mapeador = datos_prueba

    estrategia_base = EstrategiaInferencia(
        G,
        atributos,
        mapeador,
    )

    vecinos_validos = estrategia_base.obtener_vecinos_validos(
        nodo_id=0
    )

    assert 14 not in vecinos_validos
    assert vecinos_validos == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "clase_estrategia",
    ESTRATEGIAS,
    ids=[clase.__name__ for clase in ESTRATEGIAS],
)
def test_ejecutar_pipeline_inferencia(datos_prueba, clase_estrategia):
    """Cada estrategia del registro infiere colegio (índice 3) y edad (índice 4)."""
    G, atributos, circulos, mapeador = datos_prueba

    resultado = ejecutar_estrategia(
        clase_estrategia,
        G,
        atributos,
        circulos,
        mapeador,
    )

    assert resultado.inferidos[3] == 1
    assert resultado.inferidos[4] == 1
    assert resultado.xai
