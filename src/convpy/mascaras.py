"""Máscaras de la práctica 01. Bloque 1 - Michael Lata.

Mismos coeficientes que practica01/src/modelo/mascara.cpp, como float32 (k, k) C-contiguo.
"""
import numpy as np


def media(k: int) -> np.ndarray:
    """Todos los coeficientes valen 1/(k*k). Suaviza la imagen."""
    if k < 1 or k % 2 == 0:
        raise ValueError(f"k debe ser impar y positivo, se recibió {k}")
    return np.full((k, k), np.float32(1.0) / np.float32(k * k), dtype=np.float32)


def nitidez() -> np.ndarray:
    """Refuerza el píxel central frente a sus vecinos."""
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def laplaciano() -> np.ndarray:
    """Responde a los cambios bruscos de intensidad."""
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def crear(nombre: str, k: int = 3) -> np.ndarray:
    """'media' (k x k) | 'nitidez' | 'laplaciano' (solo 3x3)."""
    if nombre == "media":
        return media(k)
    if k != 3:
        raise ValueError(f"{nombre} solo existe en 3x3")
    if nombre == "nitidez":
        return nitidez()
    if nombre == "laplaciano":
        return laplaciano()
    raise ValueError(f"filtro desconocido: {nombre}")
