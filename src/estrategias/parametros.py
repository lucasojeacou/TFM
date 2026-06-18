"""Pesos de ponderación de las estrategias, centralizados para su trazabilidad experimental."""

from __future__ import annotations

# Transitividad: combinación de señales a 1 salto (alpha) y 2 saltos (beta).
TRANSITIVIDAD_ALPHA: float = 1.0
TRANSITIVIDAD_BETA: float = 0.6

# Similitud estructural: mezcla homofilia directa con nodos Adamic-Adar más similares.
SIMILITUD_PESO_DIRECTO: float = 0.7
SIMILITUD_PESO_ESTRUCTURAL: float = 0.3
SIMILITUD_TOP_K: int = 5

# Comunidades: mezcla media global de vecinos con la comunidad Louvain dominante.
COMUNIDADES_PESO_GLOBAL: float = 0.6
COMUNIDADES_PESO_LOCAL: float = 0.4

# Correlación: refuerzo por coocurrencia y umbral de frecuencia para activarlo.
CORRELACION_BOOST: float = 0.3
CORRELACION_UMBRAL_ACTIVACION: float = 0.5
