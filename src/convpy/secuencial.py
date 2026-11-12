"""Convolución secuencial en Python puro. Bloque 1 - Michael Lata.

Reglas de la guía: recorridos explícitos de píxeles y coeficientes. NumPy solo para guardar
datos; prohibido scipy/cv2.filter2D, np.convolve, numba/JIT, multiprocessing o llamar a C.
Misma operación que convolucion_cpu de la práctica 01: correlación, ceros fuera de la imagen,
acumulación en float32, entrada y salida separadas.
"""
import numpy as np


def convolucion_secuencial(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Devuelve un float32 (alto, ancho) nuevo, sin recortar."""
    raise NotImplementedError
