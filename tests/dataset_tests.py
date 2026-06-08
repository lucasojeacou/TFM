"""Dataset sintético tipo SNAP para pruebas unitarias de inferencia."""

from __future__ import annotations

import networkx as nx


class MapeadorFalso:
    """Mapeador de atributos ficticio para pruebas."""

    def __init__(self):
        """Inicializa índices y rangos de categorías del escenario de prueba."""
        self.indices_mapeados = {
            0: "heroe",
            1: "civil",
            2: "instituto_midtown",
            3: "universidad_esu",
            4: "joven",
            5: "adulto",
        }
        self.esquema_rangos = {
            "facciones": {"inicio": 0, "fin": 0},
            "colegio": {"inicio": 1, "fin": 3},
            "edad": {"inicio": 4, "fin": 5},
        }


def obtener_dataset_test():
    """Grafo, atributos, círculos y mapeador del escenario de prueba."""
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 14),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (2, 5), (3, 5),
    ])

    atributos = {
        1: [0, 1, 0, 0, 1, 0],
        2: [0, 0, 0, 1, 1, 0],
        3: [0, 0, 0, 1, 1, 0],
        4: [0, 0, 1, 1, 1, 0],
        5: [0, 0, 0, 1, 1, 0],
    }

    circulos = {
        "amigos_cercanos": [1, 2, 3],
        "companeros": [3, 4, 5],
    }

    mapeador = MapeadorFalso()
    return G, atributos, circulos, mapeador
