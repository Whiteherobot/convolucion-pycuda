"""Comparación numérica entre versiones. Bloque 2 - Jordi Romero.

Referencia: C/C++ secuencial de la práctica 01. Errores ANTES del recorte a 0-255.
"""
import numpy as np

TOLERANCIA = 1e-3  # justificar en el informe según float32 y el orden de las sumas


def errores(referencia: np.ndarray, otra: np.ndarray) -> tuple[float, float]:
    """Devuelve (error absoluto máximo, error absoluto medio)."""
    raise NotImplementedError
