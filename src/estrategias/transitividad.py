"""Combinación ponderada de señales a 1 y 2 saltos desde el ego."""

from __future__ import annotations

import numpy as np

from .base import EstrategiaInferencia


class EstrategiaTransitividad(EstrategiaInferencia):
    """Mezcla atributos de vecinos directos y nodos a dos saltos (α=1.0, β=0.6)."""

    def inferir(self, nodo_id, circulos=None):
        """Combina señales a 1 y 2 saltos con pesos α=1.0 y β=0.6."""
        alpha = 1.0
        beta = 0.6
        vecinos = self.obtener_vecinos_validos(nodo_id)
        vecinos_set = set(vecinos)
        num_attrs = len(self.mapeador.indices_mapeados)

        if not vecinos:
            return self.finalizar_inferencia([], {}, None)

        matriz_1hop = np.array([self.atributos_amigos[v] for v in vecinos])
        grados_1hop = np.array([self.grafo.degree(v) for v in vecinos])

        with np.errstate(divide="ignore", invalid="ignore"):
            pesos_1hop = 1.0 / np.sqrt(grados_1hop)
        pesos_1hop[~np.isfinite(pesos_1hop)] = 0.0

        suma_1hop = np.sum(pesos_1hop)
        pesos_1hop_norm = pesos_1hop / (suma_1hop if suma_1hop > 0 else 1.0)

        puntuaciones_1hop = np.dot(matriz_1hop.T, pesos_1hop_norm)
        contrib_1hop = matriz_1hop.T * pesos_1hop_norm

        vecinos_2hop = []
        pesos_2hop_lista = []
        nodos_2hop_vistos = set()

        for v in vecinos:
            gr_v = self.grafo.degree(v)
            for v2 in self.grafo.neighbors(v):
                if v2 != nodo_id and v2 not in vecinos_set and v2 in self.atributos_amigos:
                    if v2 not in nodos_2hop_vistos:
                        nodos_2hop_vistos.add(v2)
                        vecinos_2hop.append(v2)
                        peso_v2 = 1.0 / (np.sqrt(gr_v) * np.sqrt(self.grafo.degree(v2)))
                        pesos_2hop_lista.append(peso_v2)

        if vecinos_2hop:
            matriz_2hop = np.array([self.atributos_amigos[v2] for v2 in vecinos_2hop])
            pesos_2hop = np.array(pesos_2hop_lista)
            suma_2hop = np.sum(pesos_2hop)
            pesos_2hop_norm = pesos_2hop / (suma_2hop if suma_2hop > 0 else 1.0)
            puntuaciones_2hop = np.dot(matriz_2hop.T, pesos_2hop_norm)
            contrib_2hop = matriz_2hop.T * pesos_2hop_norm
        else:
            puntuaciones_2hop = np.zeros(num_attrs)
            contrib_2hop = np.zeros((num_attrs, 0))

        puntuaciones_vector = (alpha * puntuaciones_1hop + beta * puntuaciones_2hop) / (alpha + beta)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        contrib_xai_final = np.concatenate(
            [
                contrib_1hop * (alpha / (alpha + beta)),
                contrib_2hop * (beta / (alpha + beta)),
            ],
            axis=1,
        )
        nodos_xai = list(vecinos) + list(vecinos_2hop)

        return self.finalizar_inferencia(nodos_xai, puntuaciones, contrib_xai_final)
