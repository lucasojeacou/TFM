"""Detección Louvain en subgrafo local para ponderar la comunidad dominante."""

from __future__ import annotations

import numpy as np
import networkx as nx
from community import community_louvain as community

from .base import EstrategiaInferencia


class EstrategiaComunidades(EstrategiaInferencia):
    """60 % media global de vecinos + 40 % media de la comunidad Louvain dominante."""

    def __init__(self, grafo, atributos_amigos, mapeador):
        """Inicializa la estrategia con caché interna de particiones por subgrafo."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_comunidades: dict = {}

    def _obtener_comunidades_subgrafo(self, nodo_id, vecinos):
        """Partición Louvain del subgrafo inducido por los vecinos del ego."""
        firma = frozenset(vecinos)
        if firma in self._cache_comunidades:
            return self._cache_comunidades[firma]
        subgrafo = self.grafo.subgraph(vecinos)
        comunidades = community.best_partition(subgrafo, random_state=42) if len(subgrafo) > 0 else {}
        self._cache_comunidades[firma] = comunidades
        return comunidades

    def inferir(self, nodo_id, circulos=None):
        """Mezcla media global de vecinos con la comunidad Louvain dominante."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)
        comunidades = self._obtener_comunidades_subgrafo(nodo_id, vecinos)
        conteo_comunidades = {}
        for n, c in comunidades.items():
            conteo_comunidades[c] = conteo_comunidades.get(c, 0) + 1
        comunidad_dominante = max(conteo_comunidades, key=conteo_comunidades.get) if conteo_comunidades else None
        nodos_comunidad = [n for n, c in comunidades.items() if c == comunidad_dominante]
        matriz_global = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_global = np.mean(matriz_global, axis=0)
        matriz_comunidad = np.array([self.atributos_amigos[c] for c in nodos_comunidad]) if nodos_comunidad else np.zeros_like(puntuaciones_global)
        puntuaciones_comunidad = np.mean(matriz_comunidad, axis=0) if nodos_comunidad else np.zeros_like(puntuaciones_global)
        puntuaciones_vector = (0.6 * puntuaciones_global) + (0.4 * puntuaciones_comunidad)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        contrib_global_xai = 0.6 * (matriz_global.T / len(vecinos))
        contrib_comunidad_xai = np.zeros_like(contrib_global_xai)
        if nodos_comunidad:
            for j, v in enumerate(vecinos):
                if v in nodos_comunidad:
                    contrib_comunidad_xai[:, j] = 0.4 * (matriz_global[j, :] / len(nodos_comunidad))
        contrib_xai_total = contrib_global_xai + contrib_comunidad_xai
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai_total)
