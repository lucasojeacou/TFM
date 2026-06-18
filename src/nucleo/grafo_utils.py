"""Utilidades de recorrido de grafos compartidas: cálculo de nodos de "segundo salto"."""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping


def iter_pares_segundo_salto(
    grafo,
    ego_id: int,
    vecinos_validos: Iterable[int],
    atributos_amigos: Mapping[int, object] | None = None,
) -> Iterator[tuple[int, int]]:
    """Itera pares ``(vecino, candidato)`` del segundo salto en orden (excluye ego, vecinos directos y, si se pasa, nodos sin atributos)."""
    vecinos_lista = [int(vecino) for vecino in vecinos_validos]
    vecinos_set = set(vecinos_lista)
    ego = int(ego_id)
    for vecino in vecinos_lista:
        if not grafo.has_node(vecino):
            continue
        for candidato in grafo.neighbors(vecino):
            candidato_int = int(candidato)
            if candidato_int == ego or candidato_int in vecinos_set:
                continue
            if atributos_amigos is not None and candidato_int not in atributos_amigos:
                continue
            yield vecino, candidato_int


def nodos_segundo_salto(
    grafo,
    ego_id: int | None,
    vecinos_validos: Iterable[int],
    atributos_amigos: Mapping[int, object] | None = None,
) -> set[int]:
    """Conjunto de nodos alcanzables a dos saltos del ego a través de sus vecinos."""
    if ego_id is None:
        return set()
    return {candidato for _, candidato in iter_pares_segundo_salto(grafo, ego_id, vecinos_validos, atributos_amigos)}
