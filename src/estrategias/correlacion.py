"""Refuerzo de atributos frecuentes mediante coocurrencia en la ego-red."""

from __future__ import annotations

import numpy as np

from .base import EstrategiaInferencia, ResultadoInferencia
from .parametros import CORRELACION_BOOST, CORRELACION_UMBRAL_ACTIVACION


class EstrategiaCorrelacion(EstrategiaInferencia):
    """Homofilia directa con refuerzo por correlación cruzada entre atributos activos."""

    def __init__(self, grafo, atributos_amigos: dict[int, list[int]], mapeador) -> None:
        """Inicializa la estrategia con caché interna de matrices de correlación."""
        super().__init__(grafo, atributos_amigos, mapeador)
        self._cache_correlacion = None
        self._cache_matriz_atributos = None
        self._cache_nodos_atributos = None

    def _calcular_matriz_atributos_binarios(self):
        """Matriz nodo × atributo de toda la ego-red."""
        if self._cache_matriz_atributos is not None:
            return self._cache_matriz_atributos, self._cache_nodos_atributos
        nodos = list(self.atributos_amigos.keys())
        matriz = np.array([self.atributos_amigos[n] for n in nodos])
        self._cache_matriz_atributos = matriz
        self._cache_nodos_atributos = nodos
        return matriz, nodos

    def _obtener_matriz_correlacion(self):
        """Coocurrencia normalizada por frecuencia marginal de cada atributo."""
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

    def inferir(self, nodo_id: int, circulos: dict[str, list[int]] | None = None) -> ResultadoInferencia:
        """Homofilia directa con refuerzo por coocurrencia cruzada de atributos."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        freq_vec = np.mean(matriz_attrs, axis=0)
        scores_vec = freq_vec.copy()
        mask = freq_vec > CORRELACION_UMBRAL_ACTIVACION
        if np.any(mask):
            coocurrencia = self._obtener_matriz_correlacion()
            boost_vec = np.dot(coocurrencia[mask].T, freq_vec[mask])
            scores_vec = np.clip(freq_vec + CORRELACION_BOOST * boost_vec, 0.0, 1.0)
        puntuaciones = {i: float(val) for i, val in enumerate(scores_vec)}
        contrib_xai = matriz_attrs.T / len(vecinos)
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
