"""Punto de entrada Streamlit: pestañas de resultados, simulador y auditoría XAI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.interfaz.analisis import leer_csv_seguro
from src.nucleo.cargador_datos import descubrir_ego_ids
from src.registro import estrategias_para_dashboard
from src.interfaz.dashboard.auditoria import renderizar_pestana_auditoria
from src.interfaz.dashboard.estilos import cargar_estilos
from src.interfaz.dashboard.resultados import renderizar_pestana_resultados
from src.interfaz.dashboard.simulador import renderizar_pestana_simulador

RAIZ_PROYECTO = Path(__file__).resolve().parent
RUTA_DATOS = RAIZ_PROYECTO / "datos" / "facebook"
RUTA_RESULTADOS_CSV = RAIZ_PROYECTO / "resultados" / "resultados_estrategias.csv"


def main() -> None:
    """Configura la app y renderiza las tres pestañas principales."""
    st.set_page_config(page_title="TFM", layout="wide")
    cargar_estilos()

    df_historico = leer_csv_seguro(RUTA_RESULTADOS_CSV)
    ego_ids = descubrir_ego_ids(RUTA_DATOS)

    if not ego_ids:
        st.error("No se encontraron ego-redes SNAP en datos/facebook.")
        st.stop()

    estrategias = estrategias_para_dashboard(incluir_base=False)
    estrategia_keys = [str(item["key"]) for item in estrategias]
    estrategia_labels = {str(item["key"]): str(item["label"]) for item in estrategias}

    st.title("Inferencia privada en redes sociales")
    tab_resultados, tab_simulador, tab_grafo = st.tabs(
        ["Resultados", "Simulador", "Auditoría y grafo XAI"]
    )

    with tab_resultados:
        renderizar_pestana_resultados(df_historico, ego_ids, estrategia_keys, estrategia_labels)

    with tab_simulador:
        renderizar_pestana_simulador(RUTA_DATOS, ego_ids, estrategia_keys, estrategia_labels)

    with tab_grafo:
        renderizar_pestana_auditoria(RUTA_DATOS, ego_ids, estrategia_keys, estrategia_labels)


main()
