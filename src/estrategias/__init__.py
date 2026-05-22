"""Estrategias de inferencia estructural.

Módulos independientes para cada estrategia de inferencia:
- Homofilia (vecinos)
- Círculos contextual
- Transitividad (dos saltos)
- Correlación (atributos cruzados)
- Puentes (centralidad)
- Similitud estructural (Adamic-Adar)
- Comunidades (Louvain)
"""

from .base import EstrategiaInferencia
from .vecinos import EstrategiaVecinos
from .circulos import EstrategiaCirculos
from .transitividad import EstrategiaTransitividad
from .correlacion import EstrategiaCorrelacion
from .puentes import EstrategiaPuentes
from .similitud import EstrategiaSimilitud
from .comunidades import EstrategiaComunidades

__all__ = [
    "EstrategiaInferencia",
    "EstrategiaVecinos",
    "EstrategiaCirculos",
    "EstrategiaTransitividad",
    "EstrategiaCorrelacion",
    "EstrategiaPuentes",
    "EstrategiaSimilitud",
    "EstrategiaComunidades",
]