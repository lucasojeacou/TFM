"""Umbrales adaptativos y penalizaciones para la binarización por categoría."""

from __future__ import annotations

import math

# Puntuación mínima para activar un atributo inferido (por categoría).
# Fórmula: clip(BASE / sqrt(n_atributos_en_categoria), MIN, MAX)
UMBRAL_DINAMICO_BASE = 0.24
UMBRAL_DINAMICO_MIN = 0.04
UMBRAL_DINAMICO_MAX = 0.35

# En categorías de un solo valor (p. ej. género): separación mínima entre 1.º y 2.º candidato.
DELTA_SINGLE_Z = 1.96
DELTA_SINGLE_MIN = 0.02
DELTA_SINGLE_MAX = 0.12
DELTA_SINGLE_SIN_AMIGOS = 0.05

# Penalización de candidatos muy frecuentes en toda la ego-red.
PENALIZACION_FRECUENCIA = 0.5


def umbral_dinamico_categoria(tamano_rango: int) -> float:
    """Umbral adaptativo según cuántos atributos tiene la categoría."""
    if tamano_rango <= 0:
        return UMBRAL_DINAMICO_MAX
    valor = UMBRAL_DINAMICO_BASE / math.sqrt(tamano_rango)
    return float(max(UMBRAL_DINAMICO_MIN, min(UMBRAL_DINAMICO_MAX, valor)))


def delta_minima_single_index(num_amigos: int) -> float:
    """Margen exigido entre el mejor y el segundo candidato en categorías de un solo valor."""
    if num_amigos <= 0:
        return DELTA_SINGLE_SIN_AMIGOS
    valor = DELTA_SINGLE_Z / (2.0 * math.sqrt(num_amigos))
    return float(max(DELTA_SINGLE_MIN, min(DELTA_SINGLE_MAX, valor)))
