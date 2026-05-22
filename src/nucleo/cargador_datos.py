import os
import networkx as nx


# Carga el grafo original tal y como aparece en el archivo .edges
def cargar_grafo_base(ruta_datos, ego_id):
    ruta_edges = os.path.join(ruta_datos, f"{ego_id}.edges")
    return nx.read_edgelist(ruta_edges, nodetype=int)


# Reconstruye la ego-network añadiendo el nodo ego y conectándolo con todos los nodos existentes
def reconstruir_ego(grafo, ego_id):
    G = grafo.copy()
    nodo_ego = int(ego_id)
    nodos_existentes = list(G.nodes())
    G.add_edges_from((nodo_ego, nodo) for nodo in nodos_existentes)
    return G


# Carga los atributos de los amigos desde el archivo .feat
def cargar_atributos_amigos(ruta_datos, ego_id):
    ruta_feat = os.path.join(ruta_datos, f"{ego_id}.feat")
    atributos = {}
    with open(ruta_feat, "r") as f:
        for linea in f:
            partes = list(map(int, linea.split()))
            atributos[partes[0]] = partes[1:]
    return atributos


# Carga los atributos reales del ego desde .egofeat
def cargar_atributos_ego(ruta_datos, ego_id):
    ruta_egofeat = os.path.join(ruta_datos, f"{ego_id}.egofeat")
    with open(ruta_egofeat, "r") as f:
        return list(map(int, f.read().split()))


# Carga el nombre de cada atributo desde .featnames
def cargar_nombres_atributos(ruta_datos, ego_id):
    ruta_featnames = os.path.join(ruta_datos, f"{ego_id}.featnames")
    nombres = []
    with open(ruta_featnames, "r") as f:
        for linea in f:
            partes = linea.strip().split()
            nombres.append(" ".join(partes[1:]))
    return nombres


# Carga los círculos sociales definidos por el usuario
def cargar_circulos(ruta_datos, ego_id):
    ruta_circles = os.path.join(ruta_datos, f"{ego_id}.circles")
    circulos = {}
    if os.path.exists(ruta_circles):
        with open(ruta_circles, "r") as f:
            for linea in f:
                partes = linea.strip().split()
                if partes:
                    circulos[partes[0]] = list(map(int, partes[1:]))
    return circulos