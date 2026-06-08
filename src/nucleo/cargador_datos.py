"""Lectura de ficheros SNAP (.edges, .feat, .egofeat, .circles, .featnames)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def _ruta_archivo(ruta_datos, ego_id: int, extension: str) -> Path:
    """Ruta al fichero SNAP ``{ego_id}.{extension}``."""
    return Path(ruta_datos) / f"{ego_id}.{extension}"


def cargar_grafo_base(ruta_datos, ego_id):
    """Grafo de amigos sin el nodo ego (solo aristas del fichero .edges)."""
    ruta_edges = _ruta_archivo(ruta_datos, ego_id, "edges")
    return nx.read_edgelist(str(ruta_edges), nodetype=int)


def reconstruir_ego(grafo, ego_id):
    """Conecta el ego con todos los nodos existentes para formar la ego-red."""
    G = grafo.copy()
    nodo_ego = int(ego_id)
    nodos_existentes = list(G.nodes())
    G.add_edges_from((nodo_ego, nodo) for nodo in nodos_existentes)
    return G


def cargar_atributos_amigos(ruta_datos, ego_id):
    """Diccionario nodo → vector binario de atributos (.feat)."""
    atributos = {}
    with _ruta_archivo(ruta_datos, ego_id, "feat").open("r", encoding="utf-8") as f:
        for linea in f:
            partes = list(map(int, linea.split()))
            atributos[partes[0]] = partes[1:]
    return atributos


def cargar_atributos_ego(ruta_datos, ego_id):
    """Vector binario del perfil real del ego (.egofeat)."""
    with _ruta_archivo(ruta_datos, ego_id, "egofeat").open("r", encoding="utf-8") as f:
        return list(map(int, f.read().split()))


def cargar_nombres_atributos(ruta_datos, ego_id):
    """Lista ordenada de nombres descriptivos (.featnames)."""
    nombres = []
    with _ruta_archivo(ruta_datos, ego_id, "featnames").open("r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            nombres.append(" ".join(partes[1:]))
    return nombres


def cargar_circulos(ruta_datos, ego_id):
    """Círculos sociales SNAP: nombre → lista de ids de miembros."""
    ruta_circles = _ruta_archivo(ruta_datos, ego_id, "circles")
    circulos = {}
    if ruta_circles.exists():
        with ruta_circles.open("r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split()
                if partes:
                    circulos[partes[0]] = list(map(int, partes[1:]))
    return circulos
