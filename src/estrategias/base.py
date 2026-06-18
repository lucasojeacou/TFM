"""Clase base de estrategias: puntuaciones continuas y delegación al núcleo para binarizar."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..nucleo.politicas_semanticas import seleccionar_por_categoria

# (vector binario inferido, ganador por categoría, contribuciones XAI por atributo y nodo)
ResultadoInferencia = tuple[
    list[int],
    dict[str, tuple[object, float]],
    dict[int, dict[int, float]],
]


def obtener_vecinos_validos(grafo, atributos_amigos, nodo_id: int) -> list[int]:
    """Vecinos del nodo que tienen vector de atributos en la ego-red."""
    return [vecino for vecino in grafo.neighbors(nodo_id) if vecino in atributos_amigos]


class EstrategiaInferencia:
    """Homofilia plana sobre vecinos directos y contrato base, con API pública opcional que devuelve ``None`` salvo en las subclases que la implementan."""

    def __init__(self, grafo, atributos_amigos: dict[int, list[int]], mapeador) -> None:
        """Almacena grafo, perfiles de vecinos y mapeador semántico."""
        self.grafo = grafo
        self.atributos_amigos = {nodo: np.array(attrs, dtype=float) for nodo, attrs in atributos_amigos.items()}
        self.mapeador = mapeador

    def obtener_vecinos_validos(self, nodo_id: int) -> list[int]:
        """Vecinos con perfil de atributos disponible."""
        return obtener_vecinos_validos(self.grafo, self.atributos_amigos, nodo_id)

    def centralidad_intermediacion(self) -> dict[int, float] | None:
        """Centralidad de intermediación por nodo, o ``None`` si la estrategia no la usa."""
        return None

    def prestigio_pagerank(self) -> dict[int, float] | None:
        """PageRank por nodo, o ``None`` si la estrategia no lo usa."""
        return None

    def similitudes_estructurales(self, nodo_id: int) -> list[tuple[int, float]] | None:
        """Nodos más similares al ego (id, puntuación), o ``None`` si no aplica."""
        return None

    def comunidades_locales(self, nodo_id: int, vecinos: list[int]) -> dict[int, int] | None:
        """Partición nodo → comunidad de la vecindad, o ``None`` si no aplica."""
        return None

    def finalizar_inferencia(
        self,
        vecinos: list[int],
        puntuaciones: dict[int, float],
        contribuciones_xai: np.ndarray | None,
    ) -> ResultadoInferencia:
        """Binariza por categoría y filtra contribuciones de explicabilidad a atributos activados."""
        num_attrs = len(self.mapeador.indices_mapeados)

        inferidos, elegido_por_cat = seleccionar_por_categoria(
            puntuaciones=puntuaciones,
            mapeador=self.mapeador,
            atributos_amigos=self.atributos_amigos,
        )

        if len(inferidos) != num_attrs:
            if inferidos:
                inferidos = list(inferidos[:num_attrs]) + [0] * max(0, num_attrs - len(inferidos))
            else:
                inferidos = [0] * num_attrs

        nodos_influyentes_atributos: dict[int, dict[int, float]] = defaultdict(dict)

        if vecinos and contribuciones_xai is not None:
            for idx_attr in range(num_attrs):
                if inferidos[idx_attr] == 1:
                    for j, nodo_v in enumerate(vecinos):
                        peso = float(contribuciones_xai[idx_attr, j])
                        if peso > 0.0:
                            nodos_influyentes_atributos[idx_attr][int(nodo_v)] = peso

        return inferidos, elegido_por_cat, nodos_influyentes_atributos

    def inferir(self, nodo_id: int, circulos: dict[str, list[int]] | None = None) -> ResultadoInferencia:
        """Media aritmética de atributos entre vecinos directos."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)

        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.mean(matriz_attrs, axis=0)
        puntuaciones = {i: float(v) for i, v in enumerate(puntuaciones_vector)}

        contrib_xai = (matriz_attrs.T / len(vecinos)) if len(vecinos) > 0 else np.zeros_like(matriz_attrs.T)

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
