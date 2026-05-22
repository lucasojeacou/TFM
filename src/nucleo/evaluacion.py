import os
import csv
import numpy as np
from nucleo import clasificar_politica_categoria


# Calcula métricas binarias mediante máscaras lógicas vectoriales de NumPy.
def calcular_metricas_binarias(real, predicho, indices_evaluables=None):
    arr_real, arr_pred = np.array(real), np.array(predicho)
    if indices_evaluables is not None:
        idx = list(indices_evaluables)
        arr_real, arr_pred = arr_real[idx], arr_pred[idx]

    vp = np.sum((arr_real == 1) & (arr_pred == 1))
    fp = np.sum((arr_real == 0) & (arr_pred == 1))
    fn = np.sum((arr_real == 1) & (arr_pred == 0))
    vn = np.sum((arr_real == 0) & (arr_pred == 0))

    total = vp + fp + fn + vn
    exactitud = (vp + vn) / total if total > 0 else 0.0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0.0

    return {
        "exactitud": exactitud, "precision": precision, "sensibilidad": sensibilidad, "f1": f1,
        "vp": float(vp), "fp": float(fp), "fn": float(fn), "vn": float(vn)
    }


# Muestra en consola un resumen de rendimiento comparativo por cada estrategia.
def mostrar_reporte_consolidado(resultados, titulo="REPORTE GLOBAL DE RENDIMIENTO"):
    if not resultados:
        print("\nNo hay resultados para mostrar.")
        return

    agregados = {}
    for res in resultados:
        nom = res["estrategia"]
        if nom not in agregados:
            agregados[nom] = {"f1": 0.0, "precision": 0.0, "sensibilidad": 0.0, "ok": 0.0, "parcial": 0.0, "fallidas": 0.0, "total": 0.0, "n": 0}
        
        agregado = agregados[nom]
        agregado["n"] += 1
        agregado["f1"] += float(res["f1"])
        agregado["precision"] += float(res["precision"])
        agregado["sensibilidad"] += float(res["sensibilidad"])
        
        for campo in ["categorias_ok", "categorias_parciales", "categorias_fallidas", "categorias_evaluadas"]:
            clave_corta = campo.replace("categorias_", "") if "categorias_" in campo else campo
            agregado[clave_corta] += float(res[campo])

    print(f"\n" + "=" * 90 + f"\n{titulo}\n" + "=" * 90)
    cabecera = f"{'ESTRATEGIA':<18} | {'F1 medio':>8} | {'Precision':>9} | {'Sensib.':>8} | {'OK':>5} | {'PARCIAL':>8} | {'X':>5} | {'Eval.'}"
    print(cabecera + "\n" + "-" * len(cabecera))

    for nom, agreg in sorted(agregados.items(), key=lambda x: x[1]["f1"] / max(x[1]["n"], 1.0), reverse=True):
        n = max(agreg["n"], 1.0)
        print(f"{nom:<18} | {agreg['f1']/n:>8.4f} | {agreg['precision']/n:>9.4f} | {agreg['sensibilidad']/n:>8.4f} | {agreg['ok']/n:>5.1f} | {agreg['parcial']/n:>8.1f} | {agreg['fallidas']/n:>5.1f} | {agreg['total']/n:>5.1f}")


# Guarda los datos recolectados en un archivo CSV limpio.
def exportar_resultados_csv(resultados, ruta_csv):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    if not resultados: 
        return
    campos = list(resultados[0].keys())
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)