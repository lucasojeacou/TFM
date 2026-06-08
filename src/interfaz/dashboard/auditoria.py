"""Pestaña Auditoría: tabla real vs inferido, grafo XAI y contribuciones por atributo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.estrategias.base import obtener_vecinos_validos
from src.interfaz.analisis import cargar_instantanea, construir_auditoria, ejecutar_inferencia
from src.interfaz.configuracion import STRATEGY_REGISTRY
from src.interfaz.dashboard.comun import renderizar_influencias, renderizar_leyenda_grafo
from src.interfaz.grafo import renderizar_grafo
from src.interfaz.inferencia import construir_estrategia
from src.interfaz.ui import (
    categoria_clave_por_nombre,
    categoria_desde_tabla,
    renderizar_categoria_seleccionada,
    tabla_auditoria_coloreada,
)


def renderizar_pestana_auditoria(
    ruta_datos: Path,
    ego_ids: list[int],
    estrategia_keys: list[str],
    estrategia_labels: dict[str, str],
) -> None:
    """UI de auditoría: tabla, grafo XAI y panel de contribuciones."""
    st.subheader("Auditoría por categoría y grafo XAI")
    st.markdown(
        "<p class='small-note'>Comparación real vs inferido por categoría. "
        "Al seleccionar una fila, el grafo XAI aplica escala elástica y color por aportación dominante.</p>",
        unsafe_allow_html=True,
    )

    audit_c1, audit_c2 = st.columns([1, 1.4])
    audit_ego_id = audit_c1.selectbox(
        "Ego auditado",
        ego_ids,
        index=0,
        format_func=lambda value: f"Ego {value}",
        key="audit_ego",
    )
    audit_strategy_key = audit_c2.selectbox(
        "Estrategia auditada",
        estrategia_keys,
        format_func=lambda key: estrategia_labels[key],
        key="audit_strategy",
    )

    audit_snapshot = cargar_instantanea(ruta_datos, int(audit_ego_id))
    audit_resultado = ejecutar_inferencia(audit_snapshot, int(audit_ego_id), str(audit_strategy_key))
    auditoria = construir_auditoria(audit_snapshot, audit_resultado)

    if auditoria.empty:
        st.warning("No hay categorias evaluables para este ego.")
        return

    st.markdown("#### Tabla de categorias")
    tabla_evento = st.dataframe(
        tabla_auditoria_coloreada(auditoria),
        width="stretch",
        hide_index=True,
        height=340,
        on_select="rerun",
        selection_mode="single-row",
    )

    seleccion_rows = tabla_evento.get("selection", {}).get("rows", []) if tabla_evento else []

    if seleccion_rows:
        nombre_categoria = categoria_desde_tabla(auditoria, tabla_evento)
        categoria_key = categoria_clave_por_nombre(auditoria, nombre_categoria)
        seleccion = auditoria[auditoria["categoria"] == categoria_key].iloc[0]
        hay_categoria_seleccionada = True
    else:
        hay_categoria_seleccionada = False

    st.markdown("#### Categoría seleccionada")
    if hay_categoria_seleccionada:
        renderizar_categoria_seleccionada(seleccion)
        atributos_enfocados = [int(idx) for idx in list(seleccion["pred_indices"])]
        if not atributos_enfocados:
            st.caption("Esta categoria no tiene ningun atributo inferido que explicar en el grafo.")
    else:
        atributos_enfocados = []
        st.caption("Ninguna categoria seleccionada. Selecciona una fila de la tabla para ver detalles.")

    if not hay_categoria_seleccionada:
        return

    categorias_enfocadas = {
        str(audit_snapshot.mapeador.indices_mapeados.get(int(atributo), "")).split(";")[0].strip()
        for atributo in atributos_enfocados
    }
    mostrar_leyenda_valores = len(categorias_enfocadas) > 1
    vecinos = obtener_vecinos_validos(audit_snapshot.grafo, audit_snapshot.atributos_amigos, int(audit_ego_id))
    estrategia = construir_estrategia(
        STRATEGY_REGISTRY,
        str(audit_strategy_key),
        audit_snapshot.grafo,
        audit_snapshot.atributos_amigos,
        audit_snapshot.mapeador,
    )

    renderizar_leyenda_grafo(
        str(audit_strategy_key),
        audit_snapshot.mapeador,
        str(categoria_key),
        audit_snapshot.circulos,
        audit_snapshot.atributos_amigos,
        vecinos,
        4,
        atributos_enfocados=atributos_enfocados,
        mostrar_leyenda_valores=mostrar_leyenda_valores,
        inferidos=audit_resultado.inferidos,
        elegido_por_categoria=audit_resultado.elegido_por_categoria,
        estrategia=estrategia,
        ego_id=int(audit_ego_id),
    )
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Grafo XAI")
    if atributos_enfocados:
        st.caption("El color identifica el atributo inferido dominante de cada nodo y el tamaño su fuerza.")
    else:
        st.caption("Sin atributo inferido enfocado. Los nodos quedan neutros.")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    renderizar_grafo(
        grafo=audit_snapshot.grafo,
        posiciones=audit_snapshot.posiciones,
        ego_id=int(audit_ego_id),
        atributos_amigos=audit_snapshot.atributos_amigos,
        perfil_ego=audit_snapshot.perfil_ego,
        category_key=str(categoria_key),
        mapeador=audit_snapshot.mapeador,
        phase=4,
        estrategia=estrategia,
        strategy_key=str(audit_strategy_key),
        circulos=audit_snapshot.circulos,
        atributo_enfocado=[int(atributo) for atributo in atributos_enfocados] if atributos_enfocados else None,
        resultado_inferencia=audit_resultado,
    )

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Nodos influyentes")
    st.caption("La fuerza es la aportacion relativa a cada atributo inferido.")
    renderizar_influencias(
        audit_resultado,
        seleccion,
        audit_snapshot,
        atributo_enfocado=[int(atributo) for atributo in atributos_enfocados] if atributos_enfocados else None,
    )
