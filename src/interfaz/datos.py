"""Carga cacheada de ego-redes SNAP y cálculo de layout del grafo para el dashboard."""

from __future__ import annotations

import logging
import math
import random

import networkx as nx
import streamlit as st

from src.nucleo.cargador_datos import (
    cargar_atributos_amigos,
    cargar_atributos_ego,
    cargar_circulos,
    cargar_grafo_base,
    reconstruir_ego,
)
from src.nucleo.mapeo_atributos import analizar_red

logger = logging.getLogger(__name__)

_ESCALA_LAYOUT = 150.0
_UMBRAL_KAMADA = 60
_UMBRAL_RADIAL = 100


def _centrar_en_ego(posiciones: dict, ego_id) -> dict:
    """Traslada el layout para dejar el ego en el origen."""
    if ego_id not in posiciones:
        return dict(posiciones)
    ex, ey = posiciones[ego_id]
    return {nodo: (float(x - ex), float(y - ey)) for nodo, (x, y) in posiciones.items()}


def _normalizar_posiciones(posiciones: dict, ego_id, escala_objetivo: float = _ESCALA_LAYOUT) -> dict:
    """Escala coordenadas a un radio máximo fijo."""
    if not posiciones or ego_id not in posiciones:
        return dict(posiciones)
    centradas = _centrar_en_ego(posiciones, ego_id)
    max_dist = max(
        (math.hypot(x, y) for nodo, (x, y) in centradas.items() if nodo != ego_id),
        default=0.0,
    )
    if max_dist < 1e-6:
        return {ego_id: (0.0, 0.0)}
    factor = escala_objetivo / max_dist
    return {
        nodo: (0.0, 0.0) if nodo == ego_id else (float(x * factor), float(y * factor))
        for nodo, (x, y) in centradas.items()
    }


def _posiciones_iniciales_aleatorias(grafo, ego_id, semilla: int = 42) -> dict:
    """Semilla en disco para spring_layout sin simetría circular rígida."""
    rng = random.Random(semilla)
    n = max(1, grafo.number_of_nodes())
    radio_min = max(24.0, min(42.0, 260.0 / math.sqrt(n)))
    radio_max = radio_min + max(28.0, min(52.0, 360.0 / math.sqrt(n)))
    posiciones = {ego_id: (0.0, 0.0)}
    for nodo in grafo.nodes():
        if nodo == ego_id:
            continue
        radio = rng.uniform(radio_min, radio_max)
        angulo = rng.uniform(0.0, 2.0 * math.pi)
        posiciones[nodo] = (radio * math.cos(angulo), radio * math.sin(angulo))
    return posiciones


def posiciones_radiales(grafo, ego_id):
    """Layout por anillos según distancia en saltos al ego."""
    if ego_id not in grafo:
        return {}
    distancias = nx.single_source_shortest_path_length(grafo, ego_id)
    niveles = {}
    for nodo, distancia in distancias.items():
        niveles.setdefault(distancia, []).append(nodo)
    n = max(1, grafo.number_of_nodes())
    radio_base = max(38.0, min(72.0, 520.0 / math.sqrt(n)))
    posiciones = {ego_id: (0.0, 0.0)}
    for nivel, nodos in niveles.items():
        if nivel == 0:
            continue
        radio = radio_base * nivel
        for i, nodo in enumerate(nodos):
            angulo = (2 * math.pi * i) / len(nodos)
            posiciones[nodo] = (radio * math.cos(angulo), radio * math.sin(angulo))
    return posiciones


def _perturbar_posiciones(posiciones_radiales, ego_id, semilla=42):
    """Ruido proporcional al radio para romper la simetría circular."""
    rng = random.Random(semilla)
    perturbadas = {}
    for nodo, (x, y) in posiciones_radiales.items():
        if nodo == ego_id:
            perturbadas[nodo] = (x, y)
        else:
            radio = math.hypot(x, y)
            escala = max(4.0, radio * 0.35)
            perturbadas[nodo] = (x + rng.uniform(-escala, escala), y + rng.uniform(-escala, escala))
    return perturbadas


def calcular_posiciones(grafo: nx.Graph, ego_id) -> dict:
    """Kamada-Kawai en grafos pequeños; spring_layout con respaldo radial."""
    cantidad_nodos = grafo.number_of_nodes()
    if ego_id not in grafo or cantidad_nodos == 0:
        return {}
    if cantidad_nodos == 1:
        return {ego_id: (0.0, 0.0)}

    separacion = max(44.0, min(105.0, 210.0 / math.sqrt(cantidad_nodos)))
    iteraciones = max(200, min(600, 40 * cantidad_nodos))

    if cantidad_nodos <= _UMBRAL_KAMADA:
        try:
            pos = nx.kamada_kawai_layout(grafo, scale=_ESCALA_LAYOUT)
            return _normalizar_posiciones(pos, ego_id, _ESCALA_LAYOUT)
        except Exception:
            logger.debug("Kamada-Kawai no aplicable; se usa spring_layout.", exc_info=True)

    if cantidad_nodos > _UMBRAL_RADIAL:
        pos_inicial = _perturbar_posiciones(posiciones_radiales(grafo, ego_id), ego_id)
    else:
        pos_inicial = _posiciones_iniciales_aleatorias(grafo, ego_id)

    try:
        pos = nx.spring_layout(
            grafo,
            pos=pos_inicial,
            fixed=[ego_id],
            iterations=iteraciones,
            seed=42,
            scale=_ESCALA_LAYOUT,
            k=separacion,
            threshold=1e-4,
        )
        return _normalizar_posiciones(pos, ego_id, _ESCALA_LAYOUT)
    except Exception:
        logger.exception("Fallo al calcular posiciones; layout radial de respaldo.")
        return _normalizar_posiciones(posiciones_radiales(grafo, ego_id), ego_id, _ESCALA_LAYOUT)


@st.cache_data(show_spinner=False)
def cargar_datos_ego(ruta_datos: str, ego_id: int):
    """Grafo, atributos, perfil, círculos y mapeador de un ego (sin layout)."""
    try:
        grafo_base = cargar_grafo_base(ruta_datos, ego_id)
        grafo = reconstruir_ego(grafo_base, ego_id)
        atributos_amigos = cargar_atributos_amigos(ruta_datos, ego_id)
        perfil_ego = cargar_atributos_ego(ruta_datos, ego_id)
        circulos = cargar_circulos(ruta_datos, ego_id)
        mapeador = analizar_red(ruta_datos, ego_id)
    except Exception:
        logger.exception("Fallo al cargar la instantánea SNAP para el ego %s.", ego_id)
        raise
    return grafo, atributos_amigos, perfil_ego, circulos, mapeador


@st.cache_data(show_spinner=False)
def cargar_snapshot(ruta_datos: str, ego_id: int):
    """Instantánea completa incluyendo posiciones precalculadas del grafo."""
    grafo, atributos_amigos, perfil_ego, circulos, mapeador = cargar_datos_ego(ruta_datos, ego_id)
    posiciones = calcular_posiciones(grafo, ego_id)
    return grafo, atributos_amigos, perfil_ego, circulos, mapeador, posiciones
