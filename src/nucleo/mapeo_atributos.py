"""Mapeo de featnames SNAP a categorías semánticas y rangos de índices."""

from __future__ import annotations

from types import SimpleNamespace


def generar_indices_mapeados(featnames: list[str]) -> dict[int, str]:
    """Índice de vector → nombre descriptivo en minúsculas."""
    return {i: nombre.lower() for i, nombre in enumerate(featnames)}


def generar_esquema(indices_mapeados: dict[int, str]) -> dict[str, dict[str, int]]:
    """Agrupa índices consecutivos por prefijo semántico (antes de 'anonymized')."""
    esquema: dict[str, dict[str, int]] = {}
    for idx, nombre in indices_mapeados.items():
        keyword = "anonymized"
        if keyword in nombre:
            categoria = nombre.split(keyword)[0].strip(" ;")
        else:
            categoria = "otros"

        if categoria not in esquema:
            esquema[categoria] = {"inicio": idx, "fin": idx}
        else:
            esquema[categoria]["fin"] = idx
    return esquema


def _construir_mapeador(featnames: list[str]) -> SimpleNamespace:
    """Ensambla índices y esquema de rangos a partir de featnames."""
    indices_mapeados = generar_indices_mapeados(featnames)
    esquema_rangos = generar_esquema(indices_mapeados)
    return SimpleNamespace(
        raw_names=featnames,
        indices_mapeados=indices_mapeados,
        esquema_rangos=esquema_rangos,
    )


def analizar_red(ruta_datos, ego_id) -> SimpleNamespace:
    """Carga featnames del ego y devuelve el mapeador usado por estrategias y políticas."""
    from .cargador_datos import cargar_nombres_atributos

    featnames = cargar_nombres_atributos(ruta_datos, ego_id)
    return _construir_mapeador(featnames)
