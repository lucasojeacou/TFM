"""Clase base de estrategias: puntuaciones continuas y delegación al núcleo para binarizar."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from ..nucleo.politicas_semanticas import seleccionar_por_categoria


def obtener_vecinos_validos(grafo, atributos_amigos, nodo_id):
    """Vecinos del nodo que tienen vector de atributos en la ego-red."""
    return [vecino for vecino in grafo.neighbors(nodo_id) if vecino in atributos_amigos]


class EstrategiaInferencia:
    """Referencia didáctica: homofilia plana sobre vecinos directos."""

    def __init__(self, grafo, atributos_amigos, mapeador):
        """Almacena grafo, perfiles de vecinos y mapeador semántico."""
        self.grafo = grafo
        self.atributos_amigos = {nodo: np.array(attrs, dtype=float) for nodo, attrs in atributos_amigos.items()}
        self.mapeador = mapeador

    def obtener_vecinos_validos(self, nodo_id):
        """Vecinos con perfil de atributos disponible."""
        return obtener_vecinos_validos(self.grafo, self.atributos_amigos, nodo_id)

    def finalizar_inferencia(self, vecinos, puntuaciones, contribuciones_xai):
        """Binariza por categoría y filtra contribuciones de explicabilidad a atributos activados."""
        num_attrs = len(self.mapeador.indices_mapeados)

        inferidos, elegido_por_cat = seleccionar_por_categoria(
            puntuaciones=puntuaciones,
            mapeador=self.mapeador,
            atributos_amigos=self.atributos_amigos,
        )

        if len(inferidos) != num_attrs:
            if inferidos:
                inferidos = list(inferidos[:num_attrs]) + [0] * max(0, num_attrs - len(inferidos))
            else:
                inferidos = [0] * num_attrs

        nodos_influyentes_atributos = defaultdict(dict)

        if vecinos and contribuciones_xai is not None:
            for idx_attr in range(num_attrs):
                if inferidos[idx_attr] == 1:
                    for j, nodo_v in enumerate(vecinos):
                        peso = float(contribuciones_xai[idx_attr, j])
                        if peso > 0.0:
                            nodos_influyentes_atributos[idx_attr][int(nodo_v)] = peso

        return inferidos, elegido_por_cat, nodos_influyentes_atributos

    def inferir(self, nodo_id, circulos=None):
        """Media aritmética de atributos entre vecinos directos."""
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None)

        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.mean(matriz_attrs, axis=0)
        puntuaciones = {i: float(v) for i, v in enumerate(puntuaciones_vector)}

        contrib_xai = (matriz_attrs.T / len(vecinos)) if len(vecinos) > 0 else np.zeros_like(matriz_attrs.T)

        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai)
