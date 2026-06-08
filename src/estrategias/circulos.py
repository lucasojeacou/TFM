"""Ponderación de vecinos según pertenencia a círculos sociales SNAP."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from .base import EstrategiaInferencia


class EstrategiaCirculos(EstrategiaInferencia):
    """Atributos agregados con peso inverso a la raíz del tamaño de cada círculo."""

    def inferir(self, nodo_id, circulos=None):
        """Agrega atributos ponderando vecinos según pertenencia a círculos SNAP."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)
        circulos = circulos or {}
        pesos = defaultdict(float)
        for nombre_circulo, miembros in circulos.items():
            miembros_validos = [m for m in miembros if m in self.atributos_amigos]
            if len(miembros_validos) > 0:
                peso_circulo = 1.0 / np.sqrt(len(miembros_validos))
                for m in miembros_validos:
                    if m != nodo_id:
                        pesos[m] += peso_circulo
        suma_pesos = sum(pesos.values())
        vector_pesos = np.array([pesos[v] / (suma_pesos if suma_pesos > 0 else 1.0) for v in vecinos])
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.dot(matriz_attrs.T, vector_pesos)
        puntuaciones = {i: float(v) for i, v in enumerate(puntuaciones_vector)}
        contrib_xai = matriz_attrs.T * vector_pesos
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
