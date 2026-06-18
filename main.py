"""Batch comparativo: ejecuta todas las estrategias sobre cada ego-red y exporta CSV."""

from __future__ import annotations

import logging
from pathlib import Path

from src.nucleo.cargador_datos import (
    cargar_atributos_amigos,
    cargar_atributos_ego,
    cargar_circulos,
    cargar_grafo_base,
    descubrir_ego_ids,
    reconstruir_ego,
)
from src.nucleo.mapeo_atributos import analizar_red
from src.nucleo.evaluacion import (
    calcular_metricas_binarias,
    exportar_datos_incrementales,
    exportar_resultados_csv,
)
from src.registro import estrategias_para_batch

logger = logging.getLogger(__name__)

RAIZ_PROYECTO = Path(__file__).resolve().parent

RUTA_RESULTADOS_CSV = RAIZ_PROYECTO / "resultados" / "resultados_estrategias.csv"
RUTA_EXPLICABILIDAD_CSV = RAIZ_PROYECTO / "resultados" / "resultados_explicabilidad.csv"
RUTA_PREDICCIONES_CSV = RAIZ_PROYECTO / "resultados" / "resultados_predicciones.csv"


def ejecutar_experimento(ego_id, ruta_base):
    """Evalúa cada estrategia del registro sobre un ego y acumula métricas y XAI."""
    g_base = cargar_grafo_base(ruta_base, ego_id)
    grafo = reconstruir_ego(g_base, ego_id)
    attrs_amigos = cargar_atributos_amigos(ruta_base, ego_id)
    perfil_real = cargar_atributos_ego(ruta_base, ego_id)
    circulos = cargar_circulos(ruta_base, ego_id)
    mapeador = analizar_red(ruta_base, ego_id)

    resultados_ego = []
    explicabilidad_ego = []
    predicciones_ego = []

    for nom, estrategia_cls in estrategias_para_batch():
        estrategia = estrategia_cls(grafo, attrs_amigos, mapeador)
        inferidos, _, xai_dict = estrategia.inferir(ego_id, circulos=circulos)
        metricas = calcular_metricas_binarias(perfil_real, inferidos)

        resultados_ego.append({
            "ego_id": ego_id,
            "estrategia": nom,
            "precision": metricas["precision"],
            "sensibilidad": metricas["sensibilidad"],
            "f1": metricas["f1"],
            "exactitud": metricas["exactitud"],
        })

        for idx_attr, valor in enumerate(inferidos):
            if int(valor) == 1:
                predicciones_ego.append({
                    "ego_id": ego_id,
                    "estrategia": nom,
                    "id_atributo": idx_attr,
                })

        if xai_dict is not None:
            for id_attr, dict_amigos in xai_dict.items():
                if isinstance(dict_amigos, dict):
                    for id_amigo, peso in dict_amigos.items():
                        explicabilidad_ego.append({
                            "ego_id": ego_id,
                            "estrategia": nom,
                            "id_atributo": id_attr,
                            "id_amigo": id_amigo,
                            "contribucion": peso,
                        })

    if explicabilidad_ego:
        exportar_datos_incrementales(
            explicabilidad_ego,
            RUTA_EXPLICABILIDAD_CSV,
            ["ego_id", "estrategia", "id_atributo", "id_amigo", "contribucion"],
        )
    if predicciones_ego:
        exportar_datos_incrementales(
            predicciones_ego,
            RUTA_PREDICCIONES_CSV,
            ["ego_id", "estrategia", "id_atributo"],
        )

    return resultados_ego


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ruta_datos = RAIZ_PROYECTO / "datos" / "facebook"
    all_ego_ids = descubrir_ego_ids(ruta_datos)

    for ruta_csv in [RUTA_EXPLICABILIDAD_CSV, RUTA_PREDICCIONES_CSV]:
        if ruta_csv.exists():
            ruta_csv.unlink()

    todos_resultados = []
    for current_id in all_ego_ids:
        try:
            resultados_nodo = ejecutar_experimento(current_id, ruta_datos)
            todos_resultados.extend(resultados_nodo)
        except Exception:
            logger.exception("Fallo al procesar el ego %s; se omite.", current_id)
            continue

    if todos_resultados:
        exportar_resultados_csv(todos_resultados, RUTA_RESULTADOS_CSV)
        logger.info("Procesados %s egos. Resultados en %s", len(all_ego_ids), RUTA_RESULTADOS_CSV)
    else:
        logger.warning("No se generaron resultados: ningún ego se procesó correctamente.")
