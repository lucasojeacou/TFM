"""Similitud estructural Adamic-Adar combinada con homofilia directa."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .base import EstrategiaInferencia


class EstrategiaSimilitud(EstrategiaInferencia):
    """70 % vecinos directos + 30 % de los 5 nodos con mayor similitud estructural respecto al ego."""

    def __init__(self, grafo, atributos_amigos, mapeador):
        """Inicializa la estrategia con caché interna de similitudes por ego."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_similitudes: dict = {}

    def _obtener_similitudes_estructurales(self, nodo_id):
        """Los 5 nodos con mayor similitud estructural respecto al ego."""
        if nodo_id in self._cache_similitudes:
            return self._cache_similitudes[nodo_id]
        pares_nodos = [(nodo_id, u) for u in self.atributos_amigos if u != nodo_id and u in self.grafo]
        preds = nx.adamic_adar_index(self.grafo, pares_nodos)
        similitudes = []
        for u, v, p in preds:
            if p > 0.0:
                similitudes.append((v, float(p)))
        top_similares = sorted(similitudes, key=lambda x: x[1], reverse=True)[:5]
        self._cache_similitudes[nodo_id] = top_similares
        return top_similares

    def inferir(self, nodo_id, circulos=None):
        """Combina homofilia directa (70 %) con nodos de alta similitud estructural (30 %)."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        top_similares = self._obtener_similitudes_estructurales(nodo_id)
        if not vecinos and not top_similares:
            return self.finalizar_inferencia([], {}, None)
        contribuciones_por_nodo = {}

        if vecinos:
            matriz_vecinos = np.array([self.atributos_amigos[v] for v in vecinos])
            freqs_base = np.mean(matriz_vecinos, axis=0)
            contrib_vecinos = matriz_vecinos.T / len(vecinos)
            for j, nodo in enumerate(vecinos):
                contribuciones_por_nodo[int(nodo)] = 0.7 * contrib_vecinos[:, j]
        else:
            freqs_base = np.zeros(len(self.mapeador.indices_mapeados))
        if top_similares:
            nodos_sim, pesos_sim = zip(*top_similares)
            matriz_sim = np.array([self.atributos_amigos[ns] for ns in nodos_sim])
            suma_sim = np.sum(pesos_sim)
            pesos_sim_norm = np.array(pesos_sim) / (suma_sim if suma_sim > 0 else 1.0)
            freqs_sim = np.dot(matriz_sim.T, pesos_sim_norm)
            contrib_sim = matriz_sim.T * pesos_sim_norm
            for j, nodo in enumerate(nodos_sim):
                nodo_int = int(nodo)
                contribuciones_por_nodo[nodo_int] = contribuciones_por_nodo.get(
                    nodo_int,
                    np.zeros(len(self.mapeador.indices_mapeados)),
                ) + (0.3 * contrib_sim[:, j])
        else:
            freqs_sim = np.zeros_like(freqs_base)
        puntuaciones_vector = (0.7 * freqs_base) + (0.3 * freqs_sim)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        nodos_xai = list(contribuciones_por_nodo.keys())
        contrib_xai_final = np.column_stack([contribuciones_por_nodo[n] for n in nodos_xai]) if nodos_xai else None
        return self.finalizar_inferencia(nodos_xai, puntuaciones, contrib_xai_final)
