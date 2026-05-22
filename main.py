import os
import sys
import csv
import copy
import numpy as np

# Configuración directa de rutas del proyecto
RAIZ_PROYECTO = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(RAIZ_PROYECTO, "src"))

from nucleo.cargador_datos import (
    cargar_atributos_amigos, cargar_atributos_ego, cargar_circulos,
    cargar_grafo_base, cargar_nombres_atributos, reconstruir_ego
)
from nucleo.mapeo_atributos import analizar_red
from estrategias.vecinos import EstrategiaVecinos
from estrategias.circulos import EstrategiaCirculos
from estrategias.transitividad import EstrategiaTransitividad
from estrategias.similitud import EstrategiaSimilitud
from estrategias.comunidades import EstrategiaComunidades
from estrategias.correlacion import EstrategiaCorrelacion
from estrategias.puentes import EstrategiaPuentes
from nucleo.evaluacion import calcular_metricas_binarias, exportar_resultados_csv

# Rutas de almacenamiento consolidadas
RUTA_RESULTADOS_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_estrategias.csv")
RUTA_EXPLICABILIDAD_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_explicabilidad.csv")
RUTA_PREDICCIONES_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_predicciones.csv")

# Función única simplificada para volcar de forma incremental cualquier matriz analítica a disco
def exportar_datos_incrementales(filas, ruta_csv, campos):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    archivo_existe = os.path.exists(ruta_csv)
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        if not archivo_existe:
            escritor.writeheader()
        escritor.writerows(filas)

# Ejecuta el conjunto completo de estrategias de inferencia sobre un nodo ego específico.
def ejecutar_experimento(ego_id, ruta_base, umbral_bajo=0.12, umbral_alto=0.22):
    g_base = cargar_grafo_base(ruta_base, ego_id)
    grafo = reconstruir_ego(g_base, ego_id)
    attrs_amigos = cargar_atributos_amigos(ruta_base, ego_id)
    perfil_real = cargar_atributos_ego(ruta_base, ego_id)
    circulos = cargar_circulos(ruta_base, ego_id)
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    mapeador = analizar_red(ruta_base, ego_id)
    sys.stdout = original_stdout

    ego_reconstruido = perfil_real

    resultados_ego = []
    explicabilidad_ego = []
    predicciones_ego = []

    # Generamos los nombres de las estrategias y sus constructores bajo demanda
    instancias_lista = [
        ("vecinos", EstrategiaVecinos(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("circulos", EstrategiaCirculos(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("transitividad", EstrategiaTransitividad(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("similitud", EstrategiaSimilitud(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("comunidades", EstrategiaComunidades(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("correlacion", EstrategiaCorrelacion(grafo, copy.deepcopy(attrs_amigos), mapeador)),
        ("puentes", EstrategiaPuentes(grafo, copy.deepcopy(attrs_amigos), mapeador)),
    ]

    for nom, estrategia in instancias_lista:
        if nom in {"circulos", "puentes"}:
            inferidos, elegido_por_cat, xai_dict = estrategia.inferir(
                ego_id, circulos=circulos, umbral_bajo=umbral_bajo, umbral_alto=umbral_alto
            )
        else:
            inferidos, elegido_por_cat, xai_dict = estrategia.inferir(
                ego_id, umbral_bajo=umbral_bajo, umbral_alto=umbral_alto
            )

        metricas = calcular_metricas_binarias(ego_reconstruido, inferidos)
        
        resultados_ego.append({
            "ego_id": ego_id,
            "estrategia": nom,
            "precision": metricas["precision"],
            "sensibilidad": metricas["sensibilidad"],
            "f1": metricas["f1"],
            "exactitud": metricas["exactitud"]
        })

        for idx_attr, valor in enumerate(list(inferidos)):
            if valor == 1:
                predicciones_ego.append({
                    "ego_id": ego_id, 
                    "estrategia": nom, 
                    "id_atributo": int(idx_attr)
                })

        # Clonamos de forma segura el diccionario XAI retornado para desvincularlo de defaultdict
        if xai_dict is not None:
            copia_segura_xai = dict(xai_dict)
            for id_attr in list(copia_segura_xai.keys()):
                dict_amigos = copia_segura_xai[id_attr]
                if isinstance(dict_amigos, dict):
                    for id_amigo, peso in list(dict_amigos.items()):
                        explicabilidad_ego.append({
                            "ego_id": ego_id, 
                            "estrategia": nom, 
                            "id_atributo": int(id_attr), 
                            "id_amigo": int(id_amigo), 
                            "contribucion": float(peso)
                        })

    if explicabilidad_ego:
        exportar_datos_incrementales(explicabilidad_ego, RUTA_EXPLICABILIDAD_CSV, ["ego_id", "estrategia", "id_atributo", "id_amigo", "contribucion"])
    if predicciones_ego:
        exportar_datos_incrementales(predicciones_ego, RUTA_PREDICCIONES_CSV, ["ego_id", "estrategia", "id_atributo"])

    return resultados_ego

if __name__ == "__main__":
    ruta_datos = os.path.join(RAIZ_PROYECTO, "datos", "facebook")
    all_ego_ids = sorted([int(n.split(".", 1)[0]) for n in os.listdir(ruta_datos) if n.endswith(".egofeat")])
    
    for ruta in [RUTA_EXPLICABILIDAD_CSV, RUTA_PREDICCIONES_CSV]:
        if os.path.exists(ruta): 
            os.remove(ruta)
        
    print(f"Iniciando pipeline masivo. Detectados {len(all_ego_ids)} Ego IDs para procesamiento.")
    print("-" * 60)

    todos_resultados = []
    for current_id in all_ego_ids:
        print(f"Procesando Ego ID: {current_id}...")
        try:
            resultados_nodo = ejecutar_experimento(current_id, ruta_datos)
            todos_resultados.extend(resultados_nodo)
        except Exception as e:
            print(f"Error procesando Ego {current_id}: {e}")

    if todos_resultados:
        exportar_resultados_csv(todos_resultados, RUTA_RESULTADOS_CSV)
        print("-" * 60)
        print("Pipeline finalizado. Tablas analíticas consolidadas con éxito en la carpeta /resultados.")