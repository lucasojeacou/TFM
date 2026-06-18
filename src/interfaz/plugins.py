"""Plugins de UI por estrategia: leyendas, insights y estilos de fase 2."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.interfaz.inferencia import (
    etiqueta_atributo,
    indices_leyenda_categoria,
    paleta_categoria,
)
from src.interfaz.visual import (
    respuesta_leyenda,
    tarjetas_aristas_transitividad,
    tarjetas_aristas_vecinos,
    tarjetas_circulos_fase2,
    tarjetas_comunidades_fase2,
    tarjetas_prestigio,
    tarjetas_puentes,
    tarjetas_similitud,
    tarjetas_transitividad,
    tarjetas_vecindad_directa,
    tarjeta_leyenda,
)

STRATEGY_UI_REGISTRY: dict[str, type["StrategyUI"]] = {}


def register_strategy_ui(*strategy_keys: str):
    """Registra una clase de plugin UI para una o más claves de estrategia."""
    def decorator(cls: type["StrategyUI"]):
        """Asocia la clase UI a cada clave de estrategia indicada."""
        for strategy_key in strategy_keys:
            STRATEGY_UI_REGISTRY[strategy_key] = cls
        return cls
    return decorator


class StrategyUI:
    """Comportamiento visual por defecto; cada estrategia puede sobreescribir leyenda e insights."""

    strategy_key = "base"

    def highlight_phase2_edges(self) -> bool:
        """Indica si la fase 2 resalta aristas ego–vecino."""
        return self.strategy_key in ("base", "vecinos", "correlacion")

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Vecindad directa"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_vecindad_directa()

    def _aristas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de aristas para la leyenda de fase 2."""
        return tarjetas_aristas_vecinos() if self.highlight_phase2_edges() else []

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return None

    def render_legend(
        self,
        mapeador,
        category_key: str,
        strategy_key: str,
        circulos: dict[str, list[int]],
        circle_palette: list[str],
        atributos_amigos: dict[int, list[int]],
        vecinos_validos: list[int],
        phase: int,
        inferidos=None,
        elegido_por_categoria=None,
        **kwargs,
    ) -> dict[str, Any]:
        """Construye la leyenda del grafo según fase, categoría y estrategia."""
        if phase < 2:
            return respuesta_leyenda("Sin leyenda", [])

        if phase == 2:
            ctx = {
                "mapeador": mapeador,
                "circulos": circulos,
                "circle_palette": circle_palette,
                "vecinos_validos": vecinos_validos,
                "estrategia": kwargs.get("estrategia"),
                "ego_id": kwargs.get("ego_id"),
            }
            return respuesta_leyenda(
                self._titulo_fase2(),
                self._tarjetas_fase2(**ctx),
                self._caption_fase2(**ctx),
                self._aristas_fase2(**ctx),
            )

        palette = paleta_categoria(mapeador, category_key)
        indices = indices_leyenda_categoria(
            mapeador,
            category_key,
            atributos_amigos,
            vecinos_validos,
            phase,
            inferidos=inferidos,
            elegido_por_categoria=elegido_por_categoria,
        )
        categoria = category_key.replace(";", " / ")
        cards = [tarjeta_leyenda(palette[idx], etiqueta_atributo(mapeador, idx)) for idx in indices]
        caption = None
        if cards and int(phase) >= 3:
            caption = (
                "Si un vecino tiene varios valores activos en la categoría, "
                "el grafo muestra el primero del rango."
            )
        return respuesta_leyenda(f"Valores de {categoria}", cards, caption)

    def render_insights(
        self,
        strategy_key,
        phase,
        grafo,
        ego_id,
        category_key,
        vecinos_validos,
        atributos_amigos,
        mapeador,
        estrategia,
        circulos,
    ) -> dict[str, Any]:
        """Panel de frecuencias de atributos en la vecindad (estrategia por defecto)."""
        from src.interfaz.indicadores import filas_frecuencia_categoria

        rows = filas_frecuencia_categoria(vecinos_validos, atributos_amigos, mapeador, category_key)
        return {"title": "Distribución de atributos", "rows": rows, "note": None}


class CirculosUI(StrategyUI):
    """Leyenda e insights basados en círculos sociales SNAP."""

    strategy_key = "circulos"

    def highlight_phase2_edges(self) -> bool:
        """Los círculos colorean nodos, no aristas."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Círculos en la vecindad"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_circulos_fase2(
            ctx.get("circulos") or {},
            ctx.get("circle_palette") or [],
            ctx.get("vecinos_validos") or [],
        )

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return "Solo se colorean vecinos directos pertenecientes a un círculo."

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Penetración de vecinos en cada círculo social."""
        from src.interfaz.indicadores import filas_resumen_circulos

        return {"title": "Círculos sociales", "rows": filas_resumen_circulos(circulos, vecinos_validos), "note": None}


class ComunidadesUI(StrategyUI):
    """Leyenda e insights basados en comunidades Louvain."""

    strategy_key = "comunidades"

    def highlight_phase2_edges(self) -> bool:
        """Las comunidades resaltan aristas intragrupo en fase 2."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Comunidades locales"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_comunidades_fase2(
            ctx.get("estrategia"),
            ctx.get("ego_id"),
            ctx.get("vecinos_validos") or [],
        )

    def _aristas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Sin leyenda de aristas: el color del nodo ya identifica la comunidad."""
        return []

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return "Cada color corresponde a una comunidad detectada entre los vecinos directos."

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Distribución de vecinos entre comunidades detectadas."""
        from src.interfaz.indicadores import filas_resumen_comunidades

        return {"title": "Comunidades", "rows": filas_resumen_comunidades(estrategia, ego_id, vecinos_validos), "note": None}


class PuentesUI(StrategyUI):
    """Leyenda e insights basados en centralidad de intermediación."""

    strategy_key = "puentes"

    def highlight_phase2_edges(self) -> bool:
        """Los puentes destacan nodos, no aristas ego–vecino."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Puentes de la red"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_puentes()

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return "Los nodos destacados pueden quedar fuera de la vecindad si concentran caminos entre grupos."

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Vecinos ordenados por centralidad de intermediación."""
        from src.interfaz.indicadores import filas_resumen_puentes

        return {"title": "Nodos puente", "rows": filas_resumen_puentes(estrategia, vecinos_validos), "note": None}


class TransitividadUI(StrategyUI):
    """Leyenda e insights para expansión a uno y dos saltos."""

    strategy_key = "transitividad"

    def highlight_phase2_edges(self) -> bool:
        """La transitividad usa aristas de primer y segundo salto."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Expansión por transitividad"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_transitividad()

    def _aristas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de aristas para la leyenda de fase 2."""
        return tarjetas_aristas_transitividad()

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Nodos a dos saltos más conectados con la vecindad."""
        from src.interfaz.indicadores import filas_resumen_transitividad

        return {
            "title": "Relaciones extendidas",
            "rows": filas_resumen_transitividad(grafo, ego_id, vecinos_validos, atributos_amigos),
            "note": None,
        }


class SimilitudUI(StrategyUI):
    """Leyenda e insights basados en similitud estructural Adamic-Adar."""

    strategy_key = "similitud"

    def highlight_phase2_edges(self) -> bool:
        """La similitud destaca nodos estructuralmente cercanos."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Similitud estructural"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_similitud()

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return "Los nodos amarillos destacan por cercanía estructural global, no solo por ser vecinos directos."

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Top nodos por coeficiente Adamic-Adar."""
        from src.interfaz.indicadores import filas_resumen_similitud

        return {"title": "Relaciones extendidas", "rows": filas_resumen_similitud(estrategia, ego_id), "note": None}


class PrestigioUI(StrategyUI):
    """Leyenda e insights basados en PageRank local."""

    strategy_key = "prestigio"

    def highlight_phase2_edges(self) -> bool:
        """El prestigio escala color y tamaño de vecinos."""
        return False

    def _titulo_fase2(self) -> str:
        """Título de la leyenda en fase 2."""
        return "Prestigio en la vecindad"

    def _tarjetas_fase2(self, **ctx) -> list[dict[str, Any]]:
        """Tarjetas de nodos para la leyenda de fase 2."""
        return tarjetas_prestigio()

    def _caption_fase2(self, **ctx) -> str | None:
        """Texto auxiliar bajo la leyenda de fase 2."""
        return "El color y el tamaño de cada vecino escalan con su prestigio relativo en la red."

    def render_insights(self, strategy_key, phase, grafo, ego_id, category_key, vecinos_validos, atributos_amigos, mapeador, estrategia, circulos):
        """Vecinos ordenados por PageRank en la ego-red."""
        pr_dict = estrategia.prestigio_pagerank() if estrategia is not None else None
        if not vecinos_validos or not pr_dict:
            return {"title": "Prestigio", "rows": [], "note": None}
        rows = sorted(
            (
                {
                    "label": f"Nodo {vecino}",
                    "score": float(pr_dict.get(vecino, 0.0)),
                    "detail": f"Prestigio {float(pr_dict.get(vecino, 0.0)):.4f}",
                }
                for vecino in vecinos_validos
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return {"title": "Prestigio", "rows": rows, "note": None}


STRATEGY_UI_REGISTRY["base"] = StrategyUI
register_strategy_ui("vecinos", "correlacion")(StrategyUI)
register_strategy_ui("circulos")(CirculosUI)
register_strategy_ui("comunidades")(ComunidadesUI)
register_strategy_ui("puentes")(PuentesUI)
register_strategy_ui("transitividad")(TransitividadUI)
register_strategy_ui("similitud")(SimilitudUI)
register_strategy_ui("prestigio")(PrestigioUI)


@lru_cache(maxsize=32)
def get_strategy_ui(strategy_key: str) -> StrategyUI:
    """Devuelve la instancia de plugin UI asociada a una estrategia."""
    plugin_cls = STRATEGY_UI_REGISTRY.get(strategy_key, STRATEGY_UI_REGISTRY["base"])
    return plugin_cls()
