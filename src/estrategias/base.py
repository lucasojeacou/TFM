# Clase base abstracta y motor de cálculo vectorial para estrategias de inferencia.
# 
# Propósito y arquitectura:
# - Define el contrato estructural e inicializa los grafos y metadatos para las clases hijas.
# - Centraliza la optimización matemática mediante la conversión de perfiles a matrices NumPy.
# - Provee un motor de binarización adaptativa coordinado con las políticas semánticas.
# - Implementa XAI selectivo: filtra y almacena la influencia sobre atributos ganadores usando defaultdict.

from collections import defaultdict
import numpy as np
from nucleo.politicas_semanticas import seleccionar_por_categoria

class EstrategiaInferencia:

    def __init__(self, grafo, atributos_amigos, mapeador):
        self.grafo = grafo
        self.atributos_amigos = {nodo: np.array(attrs, dtype=float) for nodo, attrs in atributos_amigos.items()}
        self.mapeador = mapeador

    # Obtiene vecinos del nodo que tienen atributos disponibles.
    def obtener_vecinos_validos(self, nodo_id):
        return [v for v in self.grafo.neighbors(nodo_id) if v in self.atributos_amigos]

    # Centralización adaptativa de inferencia y XAI selectivo protegido contra mutaciones de tamaño
    def finalizar_inferencia(self, vecinos, puntuaciones, contribuciones_xai, umbral_bajo=0.12, umbral_alto=0.22):
        num_attrs = len(self.mapeador.indices_mapeados)
        
        # Ejecutamos el motor semántico (apaga ruidos o excesos y activa desempates excluyentes)
        inferidos, elegido_por_cat = seleccionar_por_categoria(
            puntuaciones=puntuaciones,
            mapeador=self.mapeador,
            atributos_amigos=self.atributos_amigos, # Matriz NumPy requerida para el Max_K dinámico
            umbral_bajo=umbral_bajo,
            umbral_alto=umbral_alto
        )

        # Usamos defaultdict de la librería collections para blindar la memoria contra cambios de tamaño
        nodos_influyentes_atributos = defaultdict(dict)
        
        if vecinos and contribuciones_xai is not None:
            for idx_attr in range(num_attrs):
                # Condición clave: ¿el motor semántico aprobó la inferencia de este atributo?
                if inferidos[idx_attr] == 1:
                    for j, nodo_v in enumerate(vecinos):
                        peso = float(contribuciones_xai[idx_attr, j])
                        if peso > 0.0:
                            nodos_influyentes_atributos[idx_attr][int(nodo_v)] = peso

        return inferidos, elegido_por_cat, nodos_influyentes_atributos

    # Estrategia base: homofilia local plana (promedio directo ponderado uniforme)
    def inferir(self, nodo_id, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)
            
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.mean(matriz_attrs, axis=0)
        puntuaciones = {i: float(v) for i, v in enumerate(puntuaciones_vector)}
        
        # El XAI de la homofilia plana distribuye la importancia uniformemente entre amigos activos
        contrib_xai = (matriz_attrs.T / len(vecinos)) if len(vecinos) > 0 else np.zeros_like(matriz_attrs.T)
        
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai, umbral_bajo, umbral_alto)