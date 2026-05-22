import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import random


# Calcula las métricas fundamentales del grafo.
def obtener_metricas_basicas(grafo):
    return {
        "nodos": grafo.number_of_nodes(),
        "aristas": grafo.number_of_edges(),
        "densidad": nx.density(grafo),
        "transitividad": nx.transitivity(grafo),
    }


# Identifica los nodos con más conexiones dentro del grafo.
def nodos_mas_influyentes(grafo, top_n=5):
    grados = dict(grafo.degree())
    return sorted(grados.items(), key=lambda item: item[1], reverse=True)[:top_n]


# Mantenemos esta función intacta para usarla en tus estrategias de inferencia más adelante
def detectar_comunidades(grafo):
    return community.louvain_communities(grafo)


# # Visualizar el grafo con los círculos sociales destacados
# def visualizar_grafo(grafo, circulos, ego_id=0, nombre_archivo="grafo_ego_0.png"):
#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(grafo, seed=42)
    
#     # Cada círculo tendrá un color único 
#     colores_circulos = {nombre: f"#{random.randint(0, 0xFFFFFF):06x}" for nombre in circulos.keys()}
#     colores_nodos = []
#     nodo_ego = int(ego_id)
    
#     for nodo in grafo.nodes():
#         if nodo == nodo_ego:
#             colores_nodos.append("red")
#             continue
            
#         color_asignado = "skyblue"
#         for nombre_circulo, miembros in circulos.items():
#             if nodo in miembros:
#                 color_asignado = colores_circulos[nombre_circulo]
#                 break
#         colores_nodos.append(color_asignado)

#     nx.draw_networkx_nodes(grafo, pos, node_size=50, node_color=colores_nodos, alpha=0.8)
#     nx.draw_networkx_edges(grafo, pos, alpha=0.1)

#     for nombre_circulo, color in colores_circulos.items():
#         plt.scatter([], [], c=[color], label=nombre_circulo)
    
#     plt.legend(scatterpoints=1, labelspacing=1, title="Círculos Sociales (Dataset)", loc="upper right")
#     plt.axis("off")
#     plt.savefig(nombre_archivo, bbox_inches="tight")
#     plt.close()