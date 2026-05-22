import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Configuración directa de rutas del sistema
RAIZ_PROYECTO = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(RAIZ_PROYECTO, "src"))

from nucleo import clasificar_politica_categoria
from nucleo.cargador_datos import cargar_atributos_amigos, cargar_atributos_ego, cargar_grafo_base, reconstruir_ego
from nucleo.mapeo_atributos import analizar_red

# Configuración de página de Streamlit
st.set_page_config(page_title="TFM", layout="wide")
st.title("Herramienta para la Inferencia de Información Privada en Redes Sociales")

st.markdown("""
    <style>
        button[data-baseweb="tab"] { flex: 1; text-align: center; font-size: 16px !important; font-weight: bold !important; }
        iframe { border: none !important; background-color: #141414 !important; }
    </style>
""", unsafe_allow_html=True)

RUTA_RESULTADOS_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_estrategias.csv")
RUTA_EXPLICABILIDAD_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_explicabilidad.csv")
RUTA_PREDICCIONES_CSV = os.path.join(RAIZ_PROYECTO, "resultados", "resultados_predicciones.csv")
RUTA_DATOS = os.path.join(RAIZ_PROYECTO, "datos", "facebook")

@st.cache_data
def cargar_todo_experimento(ruta_base: str, ego_id: int):
    g_base = cargar_grafo_base(ruta_base, ego_id)
    grafo = reconstruir_ego(g_base, ego_id)
    attrs_amigos = cargar_atributos_amigos(ruta_base, ego_id)
    perfil_real = cargar_atributos_ego(ruta_base, ego_id)
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    mapeador = analizar_red(ruta_base, ego_id)
    sys.stdout = original_stdout
    
    pos = nx.spring_layout(grafo, seed=42, k=2.0 / np.sqrt(len(grafo.nodes())))
    posiciones = {int(n): (float(coords[0] * 1500), float(coords[1] * 1500)) for n, coords in pos.items()}
    
    return grafo, attrs_amigos, perfil_real, mapeador, posiciones

df_historico = pd.read_csv(RUTA_RESULTADOS_CSV) if os.path.exists(RUTA_RESULTADOS_CSV) else None
df_xai = pd.read_csv(RUTA_EXPLICABILIDAD_CSV) if os.path.exists(RUTA_EXPLICABILIDAD_CSV) else None
df_preds = pd.read_csv(RUTA_PREDICCIONES_CSV) if os.path.exists(RUTA_PREDICCIONES_CSV) else None

tab_macro, tab_micro = st.tabs(["Vista General Macroscópica", "Inspección Interactiva de Redes"])

# ==========================================
# PESTAÑA 1: VISTA GENERAL MACROSCÓPICA
# ==========================================
with tab_macro:
    if df_historico is None:
        st.error("No se encontró el archivo histórico de resultados globales.")
    else:
        st.subheader("Métricas Globales Consolidadas")
        resumen = df_historico.groupby("estrategia")[["f1", "precision", "sensibilidad"]].mean().reset_index()
        mejor_est = resumen.sort_values(by="f1", ascending=False).iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor Estrategia General (F1)", mejor_est["estrategia"].upper(), f"{mejor_est['f1']:.4f}")
        c2.metric("Precisión Media Líder", f"{resumen['precision'].max():.4f}")
        c3.metric("Total de Redes Evaluadas", len(df_historico["ego_id"].unique()))
        
        st.markdown("---")
        df_melted = resumen.melt(id_vars="estrategia", value_vars=["f1", "precision", "sensibilidad"], var_name="Métrica", value_name="Valor")
        fig = px.bar(df_melted, x="estrategia", y="Valor", color="Métrica", barmode="group", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
        
        filtro = st.multiselect("Filtrar por Estrategia:", df_historico["estrategia"].unique(), default=df_historico["estrategia"].unique())
        st.dataframe(df_historico[df_historico["estrategia"].isin(filtro)], use_container_width=True)

# ==========================================
# PESTAÑA 2: INSPECCIÓN INTERACTIVA DE REDES
# ==========================================
with tab_micro:
    if df_xai is None or df_preds is None or not os.path.exists(RUTA_DATOS):
        st.error("Faltan archivos de datos consolidados. Ejecuta primero 'main.py'.")
    else:
        ego_ids = sorted([int(n.split(".", 1)[0]) for n in os.listdir(RUTA_DATOS) if n.endswith(".egofeat")])
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: ego_sel = st.selectbox("Selecciona el Ego ID:", ego_ids)
        with col_sel2: est_sel = st.selectbox("Selecciona la Estrategia:", df_historico["estrategia"].unique() if df_historico is not None else [])
            
        grafo, attrs_amigos, perfil_real, mapeador, posiciones_geometricas = cargar_todo_experimento(RUTA_DATOS, ego_sel)
        
        # Filtramos los datos precalculados estrictamente para este Ego y Estrategia
        df_ego_est_xai = df_xai[(df_xai["ego_id"] == float(ego_sel)) & (df_xai["estrategia"] == est_sel)]
        df_ego_est_preds = df_preds[(df_preds["ego_id"] == float(ego_sel)) & (df_preds["estrategia"] == est_sel)]

        # Reconstrucción de la tabla leyendo las predicciones reales calculadas por el pipeline
        filas_tabla = []
        for cat, rango in mapeador.esquema_rangos.items():
            if clasificar_politica_categoria(cat, rango, mapeador, attrs_amigos, 0.12, 0.22)["modo"] == "ruido": continue
            inicio, fin = rango["inicio"], rango["fin"]
            
            reales = [str(i + inicio) for i, val in enumerate(perfil_real[inicio : fin + 1]) if val == 1]
            
            # LEEMOS DIRECTAMENTE QUÉ ATRIBUTOS PASARON EL CORTE DEL UMBRAL DESDE EL NUEVO CSV
            predichos = [str(a) for a in df_ego_est_preds[df_ego_est_preds["id_atributo"].between(inicio, fin)]["id_atributo"].tolist()]
            
            estado = "Sin Datos" if not reales else "OK" if set(reales).issubset(set(predichos)) and predichos else "PARCIAL" if set(reales).intersection(set(predichos)) else "FALLIDO"
            filas_tabla.append({"Categoría Semántica": cat, "Real (Índices)": ", ".join(reales) if reales else "Ninguno", "Predicción (Índices)": ", ".join(predichos) if predichos else "Ninguno", "Estado": estado})
            
        df_auditoria = pd.DataFrame(filas_tabla)

        col_grafo, col_tabla = st.columns([11, 8])
        
        with col_tabla:
            st.write("##### Perfil Real vs Predicción Semántica")
            st.caption("💡 Haz clic en cualquier fila de la tabla para proyectar la influencia de esa categoría semántica en el mapa de la izquierda.")
            
            # Selección reglamentaria con guion medio
            tabla_seleccion = st.dataframe(
                df_auditoria.style.map(lambda v: f"background-color: {'#2E7D32' if v == 'OK' else '#EF6C00' if v == 'PARCIAL' else '#C62828' if v == 'FALLIDO' else '#333333'}; color: white; font-weight: bold;", subset=["Estado"]),
                use_container_width=True, height=700, on_select="rerun", selection_mode="single-row"
            )
            
        filas_clicadas = tabla_seleccion.get("selection", {}).get("rows", [])
        
        if filas_clicadas:
            cat_a_resaltar = df_auditoria.iloc[filas_clicadas[0]]["Categoría Semántica"]
            rango_cat = mapeador.esquema_rangos[cat_a_resaltar]
            df_filtrado_xai = df_ego_est_xai[df_ego_est_xai["id_atributo"].between(rango_cat["inicio"], rango_cat["fin"])]
        else:
            cat_a_resaltar = "General (Toda la estrategia)"
            df_filtrado_xai = df_ego_est_xai

        # Mapeo y normalización de intensidades XAI para PyVis
        df_amigos = df_filtrado_xai.groupby("id_amigo")["contribucion"].sum().reset_index()
        influencias = dict(zip(df_amigos["id_amigo"].astype(int), df_amigos["contribucion"].astype(float)))
        valores_inf = list(influencias.values())
        max_inf, min_inf = max(valores_inf) if valores_inf else 0.0, min(valores_inf) if valores_inf else 0.0
        rango_inf = max_inf - min_inf

        with col_grafo:
            st.write(f"##### Enfoque de Explicabilidad Topológica: **{cat_a_resaltar}**")
            pv_net = Network(height="700px", width="100%", bgcolor="#141414", font_color="white")
            
            for nodo in grafo.nodes():
                coords = posiciones_geometricas.get(nodo, (0.0, 0.0))
                
                if nodo == ego_sel:
                    color, size = "#F4D03F", 26
                else:
                    val_inf = influencias.get(nodo, 0.0)
                    intensidad = ((val_inf - min_inf) / rango_inf) if rango_inf > 0 and val_inf > 0 else 1.0 if val_inf > 0 else 0.0
                    if intensidad > 0:
                        color = f"rgba(46, 204, 113, {0.25 + 0.75 * intensidad})"
                        size = int(12 + 14 * intensidad)
                    else:
                        color, size = "#2C3E50", 6
                        
                pv_net.add_node(nodo, label=str(nodo), color=color, size=size, x=coords[0], y=coords[1])
                
            for u, v in grafo.edges(): pv_net.add_edge(u, v, color="#252525")
                
            pv_net.toggle_physics(False)
            pv_net.write_html("grafo_temporal.html")
            
            with open("grafo_temporal.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=710)
            st.caption("Leyenda: Amarillo = Ego Central | Verde Resplandeciente = Nodos con impacto real | Gris = Amigos neutrales.")