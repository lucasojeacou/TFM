# Estrategia de inferencia basada en homofilia local y votación mayoritaria simple.
# 
# Lógica de inferencia:
# - Filtra y recupera los amigos directos que disponen de perfiles válidos en la red.
# - Asigna un peso democrático equivalente a cada miembro de la vecindad inmediata.
# - Calcula la frecuencia de aparición de cada atributo mediante promediado matricial directo.
# - Despacha las puntuaciones a la base para binarización semántica inteligente.

import numpy as np
from .base import EstrategiaInferencia

class EstrategiaVecinos(EstrategiaInferencia):

    def __init__(self, grafo, atributos_amigos, mapeador, cache_frecuencias=None):
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_frecuencias = cache_frecuencias if cache_frecuencias is not None else {}

    def _obtener_frecuencias_vecindad(self, nodo_id):
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

    def inferir(self, nodo_id, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos, puntuaciones_vector = self._obtener_frecuencias_vecindad(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)
            
        puntuaciones = {i: float(val) for i, val in enumerate(puntuaciones_vector)}
        
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        contrib_xai = matriz_attrs.T / len(vecinos)
        
        return self.finalizar_inferencia(list(vecinos), puntuaciones, contrib_xai, umbral_bajo, umbral_alto)