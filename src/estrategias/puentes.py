# Estrategia de inferencia basada en nodos puente y centralidad de intermediación.
# 
# Lógica de inferencia:
# - Filtra los amigos directos que disponen de un perfil de atributos analizable.
# - Calcula la centralidad de intermediación global del grafo.
# - Cruza la centralidad con la afinidad íntima del recuento de círculos compartidos.
# - Otorga pesos dominantes a los conectores estructurales estratégicos de la red social.
# - Deriva el reporte de pesos a la base para la destilación final del XAI selectivo.

import numpy as np
import networkx as nx
from .base import EstrategiaInferencia

class EstrategiaPuentes(EstrategiaInferencia):

    def __init__(self, grafo, atributos_amigos, mapeador, cache_centralidad=None):
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_centralidad = cache_centralidad

    def _obtener_centralidad(self):
        if self._cache_centralidad is not None:
            return self._cache_centralidad
        self._cache_centralidad = nx.betweenness_centrality(self.grafo)
        return self._cache_centralidad

    def inferir(self, nodo_id, circulos, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)

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

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai, umbral_bajo, umbral_alto)