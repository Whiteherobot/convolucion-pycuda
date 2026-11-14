"""Comparación numérica entre versiones. Bloque 2 - Jordi Romero.

Referencia: C/C++ secuencial de la práctica 01. Errores ANTES del recorte a 0-255.
"""
import numpy as np

# Medido con 400x300 y valores 0-255: el error máximo contra C fue 7.6e-5 (Python, 9x9) y 4.6e-5
# (CUDA C y PyCUDA, 9x9). Viene de float32 (unos 7 dígitos) y del orden de las sumas; 1e-3 deja
# un margen de más de 10 veces sin aceptar un error de lógica, que sería de al menos 1 por píxel.
TOLERANCIA = 1e-3


def errores(referencia: np.ndarray, otra: np.ndarray) -> tuple[float, float]:
    """Devuelve (error absoluto máximo, error absoluto medio)."""
    if referencia.shape != otra.shape:
        raise ValueError(f"formas distintas: {referencia.shape} y {otra.shape}")
    # En float64 para que la resta no sume su propio error de redondeo.
    diferencia = np.abs(referencia.astype(np.float64) - otra.astype(np.float64))
    return float(diferencia.max()), float(diferencia.mean())
