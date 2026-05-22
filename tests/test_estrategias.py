import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from estrategias.vecinos import EstrategiaVecinos
from estrategias.circulos import EstrategiaCirculos
from estrategias.transitividad import EstrategiaTransitividad
from estrategias.similitud import EstrategiaSimilitud
from estrategias.comunidades import EstrategiaComunidades
from estrategias.correlacion import EstrategiaCorrelacion
from estrategias.puentes import EstrategiaPuentes


@pytest.fixture
def fixture_grafo():
    grafo = nx.Graph()
    grafo.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (3, 4),
        ]
    )
    grafo.add_node(99)
    return grafo


@pytest.fixture
def fixture_mapeador():
    return SimpleNamespace(
        raw_names=["gender_0", "gender_1", "school_0", "school_1", "school_2"],
        indices_mapeados={
            0: "gender_0",
            1: "gender_1",
            2: "school_0",
            3: "school_1",
            4: "school_2",
        },
        esquema_rangos={
            "gender": {"inicio": 0, "fin": 1},
            "school": {"inicio": 2, "fin": 4},
        },
    )


@pytest.fixture
def fixture_atributos_amigos():
    return {
        1: [1, 0, 1, 0, 0],
        2: [1, 0, 1, 1, 0],
        3: [1, 1, 0, 1, 1],
        4: [0, 1, 1, 1, 1],
    }


@pytest.fixture
def fixture_circulos():
    return {
        "circle_a": [1, 2],
        "circle_b": [3, 4],
        "circle_c": [1, 4],
    }


def _assert_contrato_y_xai(resultado, mapeador):
    # comprobamos el contrato de salida y la selectividad del xai
    assert isinstance(resultado, tuple)
    assert len(resultado) == 3

    inferidos, elegido_por_cat, nodos_influyentes_atributos = resultado

    assert isinstance(inferidos, list)
    assert len(inferidos) == len(mapeador.indices_mapeados)
    assert set(inferidos).issubset({0, 1})

    assert isinstance(elegido_por_cat, dict)
    assert isinstance(nodos_influyentes_atributos, dict)

    indices_activos = {idx for idx, valor in enumerate(inferidos) if valor == 1}
    assert set(nodos_influyentes_atributos.keys()).issubset(indices_activos)

    for idx_atributo, pesos_amigos in nodos_influyentes_atributos.items():
        assert inferidos[idx_atributo] == 1
        assert isinstance(pesos_amigos, dict)
        assert all(peso >= 0 for peso in pesos_amigos.values())


def _assert_nodo_aislado_devuelve_ceros(resultado, mapeador):
    # el nodo aislado debe responder sin romper el flujo
    assert isinstance(resultado, tuple)
    assert len(resultado) == 3

    inferidos, elegido_por_cat, nodos_influyentes_atributos = resultado

    assert inferidos == [0] * len(mapeador.indices_mapeados)
    assert elegido_por_cat == {}
    assert nodos_influyentes_atributos == {}


def test_vecinos_inferencia_completa(fixture_grafo, fixture_mapeador, fixture_atributos_amigos):
    estrategia = EstrategiaVecinos(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_circulos_inferencia_completa(
    fixture_grafo,
    fixture_mapeador,
    fixture_atributos_amigos,
    fixture_circulos,
):
    estrategia = EstrategiaCirculos(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0, circulos=fixture_circulos)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99, circulos=fixture_circulos)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_transitividad_inferencia_completa(fixture_grafo, fixture_mapeador, fixture_atributos_amigos):
    estrategia = EstrategiaTransitividad(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_similitud_inferencia_completa(fixture_grafo, fixture_mapeador, fixture_atributos_amigos):
    estrategia = EstrategiaSimilitud(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_comunidades_inferencia_completa(
    fixture_grafo,
    fixture_mapeador,
    fixture_atributos_amigos,
    monkeypatch,
):
    from estrategias import comunidades as mod_comunidades

    def _particion_determinista(subgrafo):
        return {nodo: 0 if nodo in {1, 2, 3} else 1 for nodo in subgrafo.nodes()}

    monkeypatch.setattr(mod_comunidades.community, "best_partition", _particion_determinista)

    estrategia = EstrategiaComunidades(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_correlacion_inferencia_completa(fixture_grafo, fixture_mapeador, fixture_atributos_amigos):
    estrategia = EstrategiaCorrelacion(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)


def test_puentes_inferencia_completa(
    fixture_grafo,
    fixture_mapeador,
    fixture_atributos_amigos,
    fixture_circulos,
):
    estrategia = EstrategiaPuentes(fixture_grafo, fixture_atributos_amigos, fixture_mapeador)
    resultado = estrategia.inferir(0, circulos=fixture_circulos)

    _assert_contrato_y_xai(resultado, fixture_mapeador)

    resultado_aislado = estrategia.inferir(99, circulos=fixture_circulos)
    _assert_nodo_aislado_devuelve_ceros(resultado_aislado, fixture_mapeador)
