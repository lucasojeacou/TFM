"""Ponderación por centralidad de intermediación y círculos compartidos."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .base import EstrategiaInferencia


class EstrategiaPuentes(EstrategiaInferencia):
    """Media ponderada de atributos según centralidad de intermediación y círculos compartidos."""

    def __init__(self, grafo, atributos_amigos, mapeador):
        """Inicializa la estrategia y reserva caché interna de centralidad."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_centralidad = None

    def _obtener_centralidad(self):
        """Centralidad de intermediación de la ego-red; se calcula una sola vez por instancia."""
        if self._cache_centralidad is not None:
            return self._cache_centralidad
        self._cache_centralidad = nx.betweenness_centrality(self.grafo)
        return self._cache_centralidad

    def inferir(self, nodo_id, circulos=None):
        """Pondera atributos por intermediación y solapamiento de círculos sociales."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)
        circulos = circulos or {}
        centralidad = self._obtener_centralidad()
        vector_pesos_lista = []
        for v in vecinos:
            num_circulos_compartidos = sum(1 for miembros in circulos.values() if nodo_id in miembros and v in miembros)
            peso_nodo = float(centralidad.get(v, 0.0)) * (1.0 + num_circulos_compartidos)
            vector_pesos_lista.append(peso_nodo)
        pesos_array = np.array(vector_pesos_lista)
        suma_pesos = np.sum(pesos_array)
        vector_pesos = pesos_array / (suma_pesos if suma_pesos > 0 else 1.0)
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.dot(matriz_attrs.T, vector_pesos)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        contrib_xai = matriz_attrs.T * vector_pesos
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
