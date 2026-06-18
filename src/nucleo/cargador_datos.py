"""Lectura de ficheros SNAP (.edges, .feat, .egofeat, .circles, .featnames) y descubrimiento de egos."""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def descubrir_ego_ids(ruta_datos: Path) -> list[int]:
    """Ids de ego con ficheros ``.edges`` y ``.egofeat`` disponibles en ``ruta_datos``."""
    def _ids_por_glob(patron: str) -> set[int]:
        """Extrae ids numéricos de los ficheros que coinciden con el patrón glob."""
        encontrados: set[int] = set()
        for path in Path(ruta_datos).glob(patron):
            try:
                encontrados.add(int(path.stem))
            except ValueError:
                logger.warning("Archivo ignorado por nombre no numérico: %s", path.name)
        return encontrados

    por_edges = _ids_por_glob("*.edges")
    por_egofeat = _ids_por_glob("*.egofeat")
    if por_edges and por_egofeat:
        return sorted(por_edges & por_egofeat)
    return sorted(por_edges | por_egofeat)


def _ruta_archivo(ruta_datos: str | Path, ego_id: int, extension: str) -> Path:
    """Ruta al fichero SNAP ``{ego_id}.{extension}``."""
    return Path(ruta_datos) / f"{ego_id}.{extension}"


def cargar_grafo_base(ruta_datos: str | Path, ego_id: int) -> nx.Graph:
    """Grafo de amigos sin el nodo ego (solo aristas del fichero .edges)."""
    ruta_edges = _ruta_archivo(ruta_datos, ego_id, "edges")
    return nx.read_edgelist(str(ruta_edges), nodetype=int)


def reconstruir_ego(grafo: nx.Graph, ego_id: int) -> nx.Graph:
    """Conecta el ego con todos los nodos existentes para formar la ego-red."""
    G = grafo.copy()
    nodo_ego = int(ego_id)
    nodos_existentes = list(G.nodes())
    G.add_edges_from((nodo_ego, nodo) for nodo in nodos_existentes)
    return G


def cargar_atributos_amigos(ruta_datos: str | Path, ego_id: int) -> dict[int, list[int]]:
    """Diccionario nodo → vector binario de atributos (.feat)."""
    atributos: dict[int, list[int]] = {}
    with _ruta_archivo(ruta_datos, ego_id, "feat").open("r", encoding="utf-8") as f:
        for linea in f:
            partes = list(map(int, linea.split()))
            atributos[partes[0]] = partes[1:]
    return atributos


def cargar_atributos_ego(ruta_datos: str | Path, ego_id: int) -> list[int]:
    """Vector binario del perfil real del ego (.egofeat)."""
    with _ruta_archivo(ruta_datos, ego_id, "egofeat").open("r", encoding="utf-8") as f:
        return list(map(int, f.read().split()))


def cargar_nombres_atributos(ruta_datos: str | Path, ego_id: int) -> list[str]:
    """Lista ordenada de nombres descriptivos (.featnames)."""
    nombres: list[str] = []
    with _ruta_archivo(ruta_datos, ego_id, "featnames").open("r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            nombres.append(" ".join(partes[1:]))
    return nombres


def cargar_circulos(ruta_datos: str | Path, ego_id: int) -> dict[str, list[int]]:
    """Círculos sociales SNAP: nombre → lista de ids de miembros."""
    ruta_circles = _ruta_archivo(ruta_datos, ego_id, "circles")
    circulos: dict[str, list[int]] = {}
    if ruta_circles.exists():
        with ruta_circles.open("r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split()
                if partes:
                    circulos[partes[0]] = list(map(int, partes[1:]))
    return circulos
