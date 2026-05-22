# Estrategia de inferencia basada en la confianza y exclusividad de círculos sociales.
# 
# Lógica de inferencia:
# - Filtra los amigos directos (vecinos) que disponen de un perfil de atributos válido.
# - Analiza los círculos y calcula un peso de influencia inverso a su tamaño.
# - Ejecuta promediado ponderado en bloque mediante multiplicación matricial NumPy.
# - Pasa la matriz XAI de círculos a la base para binarizar con políticas semánticas.

from collections import defaultdict
import numpy as np
from .base import EstrategiaInferencia

class EstrategiaCirculos(EstrategiaInferencia):

    def inferir(self, nodo_id, circulos, umbral_bajo=0.12, umbral_alto=0.22):
        vecinos = self.obtener_vecinos_validos(nodo_id)
        if not vecinos:
            return self.finalizar_inferencia([], {}, None, umbral_bajo, umbral_alto)
            
        pesos = defaultdict(float)
        # Penalización inversa de círculos: grupos compartidos más pequeños influyen más
        for nombre_circulo, miembros in circulos.items():
            miembros_validos = [m for m in miembros if m in self.atributos_amigos]
            if len(miembros_validos) > 0:
                peso_circulo = 1.0 / np.sqrt(len(miembros_validos))
                for m in miembros_validos:
                    if m != nodo_id:
                        pesos[m] += peso_circulo

        suma_pesos = sum(pesos.values())
        vector_pesos = np.array([pesos[v] / (suma_pesos if suma_pesos > 0 else 1.0) for v in vecinos])
        
        # Álgebra lineal en bloque para puntuaciones continuas
        matriz_attrs = np.array([self.atributos_amigos[v] for v in vecinos])
        puntuaciones_vector = np.dot(matriz_attrs.T, vector_pesos)
        puntuaciones = {i: float(v) for i, v in enumerate(puntuaciones_vector)}
        
        # El XAI de círculos escala la presencia del atributo por la importancia relativa del amigo
        contrib_xai = matriz_attrs.T * vector_pesos
        
        # Enviamos los datos para procesar el embudo semántico adaptativo y el XAI selectivo
        return self.finalizar_inferencia(vecinos, puntuaciones, contrib_xai, umbral_bajo, umbral_alto)