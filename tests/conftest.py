"""Fixtures compartidas de pytest para el suite de tests."""

from __future__ import annotations

import pytest

from tests.dataset_tests import obtener_dataset_test


@pytest.fixture
def datos_prueba():
    """Dataset sintético de ego-red para pruebas de inferencia."""
    return obtener_dataset_test()
