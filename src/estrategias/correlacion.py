# Estrategia de inferencia basada en la coocurrencia y correlación estadística de perfiles.
# 
# Lógica de inferencia:
# - Compila los perfiles de atributos de toda la red en una matriz binaria global en memoria.
# - Calculates una matriz de coocurrencia cruzada normalizada para modelar dependencias.
# - Extrae las frecuencias de atributos base procedentes de la vecindad directa del nodo.
# - Aplica un empuje multiplicativo sobre propiedades que suelen aparecer juntas.
# - Remite las puntuaciones extendidas a la base para binarización adaptativa.

import numpy as np
from .base import EstrategiaInferencia

class EstrategiaCorrelacion(EstrategiaInferencia):

    def __init__(self, grafo, atributos_amigos, mapeador, cache_correlacion=None):
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_correlacion = cache_correlacion
        self._cache_matriz_atributos = None
        self._cache_nodos_atributos = None

    def _calcular_matriz_atributos_binarios(self):
        if self._cache_matriz_atributos is not None:
            return self._cache_matriz_atributos, self._cache_nodos_atributos
        nodos = list(self.atributos_amigos.keys())
        matriz = np.array([self.atributos_amigos[n] for n in nodos])
        self._cache_matriz_atributos = matriz
        self._cache_nodos_atributos = nodos
        return matriz, nodos

    def _obtener_matriz_correlacion(self):
        if self._cache_correlacion is not None:
            return self._cache_correlacion
        matriz_binaria, _ = self._calcular_matriz_atributos_binarios()
        coocurrencia = np.dot(matriz_binaria.T, matriz_binaria)
        diagonal = np.diag(coocurrencia)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            diagonal_inv = np.where(diagonal > 0, 1.0 / diagonal, 0.0)
        norm_matriz = coocurrencia * diagonal_inv[:, None]
        np.nan_to_num(norm_matriz, copy=False, nan=0.0, posinf=0.0)
        
        self._cache_correlacion = norm_matriz
        return norm_matriz

    def inferir(self, nodo_id, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)

        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        freq_vec = np.mean(matriz_attrs, axis=0)
        scores_vec = freq_vec.copy()

        mask = freq_vec > 0.5
        if np.any(mask):
            coocurrencia = self._obtener_matriz_correlacion()
            boost_vec = np.dot(coocurrencia[mask].T, freq_vec[mask])
            scores_vec = np.clip(freq_vec + 0.3 * boost_vec, 0.0, 1.0)

        puntuaciones = {i: float(val) for i, val in enumerate(scores_vec)}
        contrib_xai = matriz_attrs.T / len(vecinos)

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai, umbral_bajo, umbral_alto)