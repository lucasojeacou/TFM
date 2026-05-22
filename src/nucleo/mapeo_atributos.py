import os
from types import SimpleNamespace
from typing import Dict, List

# Genera el diccionario de nombres indexados a partir de featnames.
def generar_indices_mapeados(featnames: List[str]) -> Dict[int, str]:
    return {i: nombre.lower() for i, nombre in enumerate(featnames)}


# Detecta bloques contiguos de atributos mediante análisis de prefijos.
def generar_esquema(indices_mapeados: Dict[int, str]) -> Dict[str, Dict[str, int]]:
    esquema: Dict[str, Dict[str, int]] = {}

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


# Construye el objeto de mapeo que usan las estrategias y la explicabilidad.
def crear_mapeador(featnames: List[str]):
    indices_mapeados = generar_indices_mapeados(featnames)
    esquema_rangos = generar_esquema(indices_mapeados)

    return SimpleNamespace(
        raw_names=featnames,
        indices_mapeados=indices_mapeados,
        esquema_rangos=esquema_rangos,
    )


# Devuelve todos los índices pertenecientes a una categoría.
def obtener_indices_por_categoria(mapeador, categoria: str) -> List[int]:
    cat = categoria.lower()
    return [i for i, nombre in mapeador.indices_mapeados.items() if cat in nombre]


# Muestra un reporte de las categorías y sus rangos.
def mostrar_reporte(esquema_rangos: Dict[str, Dict[str, int]]):
    header = f"{'CATEGORIA SEMANTICA':<45} | {'RANGO':<12} | {'TOTAL'}"
    print(f"\n{header}")
    print("-" * len(header))

    for categoria, rango in esquema_rangos.items():
        count = (rango["fin"] - rango["inicio"]) + 1
        texto_rango = f"[{rango['inicio']}-{rango['fin']}]"
        print(f"{categoria.upper():<45} | {texto_rango:<12} | {count}")


# Carga featnames desde disco y construye el mapeador.
def analizar_red(ruta_datos, ego_id):
    from .cargador_datos import cargar_nombres_atributos
    featnames = cargar_nombres_atributos(ruta_datos, ego_id)
    indices_mapeados = generar_indices_mapeados(featnames)
    esquema_rangos = generar_esquema(indices_mapeados)

    return SimpleNamespace(
        raw_names=featnames,
        indices_mapeados=indices_mapeados,
        esquema_rangos=esquema_rangos,
    )