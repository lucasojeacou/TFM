"""Tests del núcleo: métricas, umbrales, estados de auditoría y exportación CSV."""

from __future__ import annotations

from src.nucleo.evaluacion import (
    calcular_metricas_binarias,
    estado_categoria,
    exportar_datos_incrementales,
)
from src.nucleo.umbrales import umbral_dinamico_categoria


def test_estado_categoria_coincidencia_exacta():
    """Marca Correcto cuando real e inferido coinciden."""
    assert estado_categoria([1, 2], [1, 2]) == "Correcto"


def test_estado_categoria_parcial():
    """Marca Parcial cuando hay intersección sin coincidencia total."""
    assert estado_categoria([1], [1, 2]) == "Parcial"


def test_calcular_metricas_binarias_basico():
    """Calcula VP, VN y F1 sobre vectores binarios simples."""
    metricas = calcular_metricas_binarias([1, 0, 1, 0], [1, 1, 0, 0])
    assert metricas["vp"] == 1.0
    assert metricas["vn"] == 1.0
    assert 0.0 <= metricas["f1"] <= 1.0


def test_umbral_dinamico_categoria_escala_con_tamano():
    """El umbral decrece al aumentar el número de atributos en la categoría."""
    assert umbral_dinamico_categoria(1) > umbral_dinamico_categoria(16)


def test_exportar_datos_incrementales(tmp_path):
    """Crea cabecera y filas en un CSV incremental nuevo."""
    ruta = tmp_path / "salida" / "prueba.csv"
    exportar_datos_incrementales(
        [{"a": 1, "b": 2}],
        ruta,
        ["a", "b"],
    )
    contenido = ruta.read_text(encoding="utf-8")
    assert "a,b" in contenido
    assert "1,2" in contenido
