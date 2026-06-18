"""Métricas de clasificación binaria y exportación de resultados a CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def calcular_metricas_binarias(real, predicho):
    """Precision, sensibilidad, F1 y exactitud sobre vectores de atributos binarios."""
    arr_real = np.asarray(real, dtype=int).reshape(-1)
    arr_pred = np.asarray(predicho, dtype=int).reshape(-1)
    if arr_real.size != arr_pred.size:
        longitud = max(arr_real.size, arr_pred.size)
        arr_real = np.pad(arr_real, (0, longitud - arr_real.size), constant_values=0)
        arr_pred = np.pad(arr_pred, (0, longitud - arr_pred.size), constant_values=0)
    arr_real = np.where(arr_real == 1, 1, 0)
    arr_pred = np.where(arr_pred == 1, 1, 0)

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
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "f1": f1,
        "vp": float(vp),
        "fp": float(fp),
        "fn": float(fn),
        "vn": float(vn),
    }


def exportar_datos_incrementales(filas, ruta_csv, campos) -> None:
    """Añade filas a un CSV, creando cabecera si el fichero no existe."""
    ruta = Path(ruta_csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    archivo_existe = ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        if not archivo_existe:
            escritor.writeheader()
        escritor.writerows(filas)


def exportar_resultados_csv(resultados, ruta_csv) -> None:
    """Sobrescribe un CSV con el lote completo de resultados."""
    ruta = Path(ruta_csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not resultados:
        return
    campos = list(resultados[0].keys())
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


def estado_categoria(reales: list[int], predichos: list[int]) -> str:
    """Etiqueta de acierto de una categoría: Correcto, Parcial, Fallido o Sin datos."""
    reales_set = set(reales)
    predichos_set = set(predichos)
    if not reales_set and not predichos_set:
        return "Sin datos"
    if reales_set == predichos_set:
        return "Correcto"
    if not reales_set.isdisjoint(predichos_set):
        return "Parcial"
    return "Fallido"
