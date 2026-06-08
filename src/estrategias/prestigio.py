"""Ponderación de vecinos según PageRank local en la ego-red."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .base import EstrategiaInferencia


class EstrategiaPrestigio(EstrategiaInferencia):
    """Media de atributos ponderada por PageRank de cada vecino."""

    def __init__(self, grafo, atributos_amigos, mapeador):
        """Inicializa la estrategia y reserva caché interna de prestigio."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_pagerank = None

    def _obtener_pagerank(self):
        """Prestigio (PageRank) de la ego-red; se calcula una sola vez por instancia."""
        if self._cache_pagerank is not None:
            return self._cache_pagerank
        self._cache_pagerank = nx.pagerank(self.grafo)
        return self._cache_pagerank

    def inferir(self, nodo_id, circulos=None):
        """Media de atributos ponderada por PageRank de cada vecino."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)

        pagerank = self._obtener_pagerank()
        pesos_nodos = np.array([float(pagerank.get(v, 0.0)) for v in vecinos])

        suma_pesos = np.sum(pesos_nodos)
        vector_pesos = pesos_nodos / (suma_pesos if suma_pesos > 0 else 1.0)

        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.dot(matriz_attrs.T, vector_pesos)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}

        contrib_xai = matriz_attrs.T * vector_pesos

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
