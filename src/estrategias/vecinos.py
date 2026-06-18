"""Homofilia local con caché de frecuencias por conjunto de vecinos."""

from __future__ import annotations

import numpy as np

from .base import EstrategiaInferencia, ResultadoInferencia


class EstrategiaVecinos(EstrategiaInferencia):
    """Media de atributos de vecinos directos; idéntica a base pero con caché interna."""

    def __init__(self, grafo, atributos_amigos: dict[int, list[int]], mapeador) -> None:
        """Inicializa la estrategia con caché interna de frecuencias por vecindad."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_frecuencias: dict = {}

    def _obtener_frecuencias_vecindad(self, nodo_id: int):
        """Devuelve vecinos ordenados y vector de frecuencias, reutilizando caché."""
        vecinos = tuple(sorted(self.obtener_vecinos_validos(nodo_id)))
        if vecinos in self._cache_frecuencias:
            return vecinos, self._cache_frecuencias[vecinos]

        if not vecinos:
            puntuaciones_vector = np.zeros(len(self.mapeador.indices_mapeados))
        else:
            matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
            puntuaciones_vector = np.mean(matriz_attrs, axis=0)

        self._cache_frecuencias[vecinos] = puntuaciones_vector
        return vecinos, puntuaciones_vector

    def inferir(self, nodo_id: int, circulos: dict[str, list[int]] | None = None) -> ResultadoInferencia:
        """Media de atributos de vecinos directos con explicabilidad proporcional."""
        vecinos, puntuaciones_vector = self._obtener_frecuencias_vecindad(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)

        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}

        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        contrib_xai = matriz_attrs.T / len(vecinos)

        return self.finalizar_inferencia(list(vecinos), puntuaciones, contrib_xai)
