"""Pestaña Simulador: flujo en 4 fases con inferencia única en la fase final."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.estrategias.base import obtener_vecinos_validos
from src.interfaz.analisis import cargar_instantanea, construir_auditoria, ejecutar_inferencia
from src.registro import construir_estrategia
from src.interfaz.dashboard.comun import renderizar_leyenda_grafo
from src.interfaz.grafo import renderizar_grafo
from src.interfaz.ui import (
    renderizar_categoria_seleccionada,
    renderizar_encabezado_fase,
    renderizar_tarjetas_fase,
)

FASES_CONFIG = {
    1: {
        "titulo": "Estado inicial / reposo",
        "desc": "Grafo neutro en reposo con el ego en el centro.",
        "siguiente": "Detección de la vecindad del ego.",
    },
    2: {
        "titulo": "Escáner y filtrado",
        "desc": "Selección de amigos y aristas según el algoritmo de inferencia.",
        "siguiente": "Mapeo de categorías y colores de vecinos.",
    },
    3: {
        "titulo": "Categorías y colores",
        "desc": "Vecindad localizada y coloreada por categoría correspondiente.",
        "siguiente": "Ejecución de inferencia y cálculo del veredicto final.",
    },
    4: {
        "titulo": "Veredicto final",
        "desc": "Inferencia ejecutada. La red queda plana y estática para leer el valor real.",
        "siguiente": "Simulación finalizada. Reinicia para probar otra combinación.",
    },
}


def _texto_fase(phase: int) -> tuple[str, str]:
    """Título y descripción de la fase actual."""
    fase = FASES_CONFIG.get(phase, FASES_CONFIG[1])
    return fase["titulo"], fase["desc"]


def _texto_siguiente_paso(phase: int) -> str:
    """Texto orientativo del paso siguiente."""
    return FASES_CONFIG.get(phase, FASES_CONFIG[1])["siguiente"]


def _etiqueta_boton(phase: int) -> str:
    """Etiqueta del botón de avance o reinicio."""
    return "Siguiente paso" if phase < 4 else "Reiniciar simulación"


def _avanzar_fase_callback() -> None:
    """Avanza una fase o reinicia al llegar a la 4."""
    actual = st.session_state.get("sim_phase", 1)
    st.session_state.sim_phase = 1 if actual >= 4 else actual + 1


def _reiniciar_simulador_si_cambio_config() -> None:
    """Reinicia a fase 1 si cambian ego, estrategia o categoría."""
    categoria = st.session_state.get("sim_category")
    clave_actual = (
        int(st.session_state.get("sim_ego", 0)),
        str(st.session_state.get("sim_strategy", "")),
        str(categoria) if categoria is not None else "",
    )
    clave_previa = st.session_state.get("sim_config_clave")
    if clave_previa is not None and clave_previa != clave_actual:
        st.session_state.sim_phase = 1
    st.session_state.sim_config_clave = clave_actual


def renderizar_pestana_simulador(
    ruta_datos: Path,
    ego_ids: list[int],
    estrategia_keys: list[str],
    estrategia_labels: dict[str, str],
) -> None:
    """UI del simulador: selección, fases 1–4 y grafo con inferencia única."""
    st.subheader("Simulador paso a paso")
    st.markdown(
        "<p class='small-note'>Ejecución visual de una estrategia sobre un ego y una categoría.</p>",
        unsafe_allow_html=True,
    )

    sim_c0, sim_c1, sim_c2 = st.columns([1, 1, 1.25])
    sim_ego_id = sim_c0.selectbox(
        "Ego simulado",
        ego_ids,
        index=0,
        format_func=lambda value: f"Ego {value}",
        key="sim_ego",
    )
    sim_strategy_key = sim_c1.selectbox(
        "Estrategia simulada",
        estrategia_keys,
        format_func=lambda key: estrategia_labels[key],
        key="sim_strategy",
    )

    sim_snapshot = cargar_instantanea(ruta_datos, int(sim_ego_id))
    sim_resultado = ejecutar_inferencia(sim_snapshot, int(sim_ego_id), str(sim_strategy_key))
    sim_auditoria = construir_auditoria(sim_snapshot, sim_resultado)

    categorias_lista = sim_auditoria["categoria"].tolist() if not sim_auditoria.empty else []

    if categorias_lista:
        categoria_default = categorias_lista.index("gender") if "gender" in categorias_lista else 0
        sim_category = sim_c2.selectbox(
            "Categoría simulada",
            categorias_lista,
            index=categoria_default,
            format_func=lambda value: value.replace(";", " / "),
            key="sim_category",
        )
    else:
        sim_category = None
        sim_c2.warning("No hay categorías evaluables.")

    _reiniciar_simulador_si_cambio_config()

    if "sim_phase" not in st.session_state:
        st.session_state.sim_phase = 1

    phase = int(st.session_state.sim_phase)
    titulo, subtitulo = _texto_fase(phase)
    phase_title_col, phase_button_col = st.columns([4, 1])

    with phase_title_col:
        renderizar_encabezado_fase(phase, titulo)
        st.markdown(
            f"<p style='color: #64748b; font-size: 0.95rem; margin-top: -0.65rem; margin-bottom: 0.8rem;'>{subtitulo}</p>",
            unsafe_allow_html=True,
        )
    with phase_button_col:
        st.write("")
        st.button(_etiqueta_boton(phase), width="stretch", on_click=_avanzar_fase_callback)

    renderizar_tarjetas_fase(phase)
    st.progress(phase / 4.0)
    st.markdown(
        f"<p class='small-note'><b>Paso siguiente:</b> {_texto_siguiente_paso(phase)}</p>",
        unsafe_allow_html=True,
    )

    if not sim_category:
        st.warning("Selecciona una categoría evaluable para iniciar la simulación.")
        return

    seleccion_sim = sim_auditoria[sim_auditoria["categoria"] == sim_category].iloc[0]
    vecinos = obtener_vecinos_validos(sim_snapshot.grafo, sim_snapshot.atributos_amigos, int(sim_ego_id))

    estrategia = construir_estrategia(
        str(sim_strategy_key),
        sim_snapshot.grafo,
        sim_snapshot.atributos_amigos,
        sim_snapshot.mapeador,
    )

    renderizar_leyenda_grafo(
        str(sim_strategy_key),
        sim_snapshot.mapeador,
        str(sim_category),
        sim_snapshot.circulos,
        sim_snapshot.atributos_amigos,
        vecinos,
        phase,
        inferidos=sim_resultado.inferidos,
        elegido_por_categoria=sim_resultado.elegido_por_categoria,
        estrategia=estrategia,
        ego_id=int(sim_ego_id),
    )
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    if phase >= 4:
        st.markdown("#### Resultado de la categoría simulada")
        renderizar_categoria_seleccionada(seleccion_sim)
    renderizar_grafo(
        grafo=sim_snapshot.grafo,
        posiciones=sim_snapshot.posiciones,
        ego_id=int(sim_ego_id),
        atributos_amigos=sim_snapshot.atributos_amigos,
        perfil_ego=sim_snapshot.perfil_ego,
        category_key=str(sim_category),
        mapeador=sim_snapshot.mapeador,
        phase=phase,
        estrategia=estrategia,
        strategy_key=str(sim_strategy_key),
        circulos=sim_snapshot.circulos,
        resultado_inferencia=sim_resultado,
    )
