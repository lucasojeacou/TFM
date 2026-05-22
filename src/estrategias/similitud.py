# Estrategia de inferencia basada en la similitud estructural de Adamic-Adar.
# 
# Lógica de inferencia:
# - Ejecuta la métrica de Adamic-Adar para penalizar a los intermediarios hiperconectados.
# - Extrae los nodos estructuralmente más análogos de la red completa.
# - Fusiona el conocimiento democrático de primer lazo con la matriz de perfiles similares.
# - Enruta el vector compuesto hacia la base para purgar el mapa XAI de influencia.

import numpy as np
import networkx as nx
from .base import EstrategiaInferencia

class EstrategiaSimilitud(EstrategiaInferencia):

    def __init__(self, grafo, atributos_amigos, mapeador, cache_similitudes=None):
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_similitudes = cache_similitudes if cache_similitudes is not None else {}

    def _obtener_similitudes_estructurales(self, nodo_id):
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

    def inferir(self, nodo_id, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos = self.obtener_vecinos_validos(nodo_id)
        top_similares = self._obtener_similitudes_estructurales(nodo_id)

        if not vecinos and not top_similares:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)

        if vecinos:
            matriz_vecinos = np.array([self.atributos_amigos[v] for v in vecinos])
            freqs_base = np.mean(matriz_vecinos, axis=0)
            contrib_vecinos = matriz_vecinos.T / len(vecinos)
        else:
            freqs_base = np.zeros(len(self.mapeador.indices_mapeados))
            contrib_vecinos = None

        if top_similares:
            nodos_sim, pesos_sim = zip(*top_similares)
            matriz_sim = np.array([self.atributos_amigos[ns] for ns in nodos_sim])
            suma_sim = np.sum(pesos_sim)
            pesos_sim_norm = np.array(pesos_sim) / (suma_sim if suma_sim > 0 else 1.0)
            freqs_sim = np.dot(matriz_sim.T, pesos_sim_norm)
        else:
            freqs_sim = np.zeros_like(freqs_base)

        puntuaciones_vector = (0.7 * freqs_base) + (0.3 * freqs_sim)
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        contrib_xai_final = (contrib_vecinos * 0.7) if contrib_vecinos is not None else None

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai_final, umbral_bajo, umbral_alto)