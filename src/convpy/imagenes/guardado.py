"""SECCIÓN 1 · GUARDADO DE IMÁGENES. Bloque 1 - Michael Lata.

La convolución devuelve float32 SIN recortar (la nitidez puede pasar de 255 y el laplaciano da
negativos). El recorte a 0-255 solo se hace al guardar para ver, igual que en la práctica 01.
"""
import numpy as np
from PIL import Image


def a_rango_visual(imagen: np.ndarray) -> np.ndarray:
    """Igual que a_rango_visual de la P01: recorta a [0, 255] y redondea sumando 0.5."""
    return (np.clip(imagen, 0.0, 255.0) + np.float32(0.5)).astype(np.uint8)


def guardar_visual(ruta: str, imagen: np.ndarray) -> None:
    """Guarda como uint8 (PGM o PNG según la extensión).

    compress_level=1: PNG comprime menos pero guarda ~5 veces más rápido; los píxeles son los mismos.
    """
    opciones = {"compress_level": 1} if str(ruta).lower().endswith(".png") else {}
    Image.fromarray(a_rango_visual(imagen), mode="L").save(ruta, **opciones)


def guardar_crudo(ruta: str, imagen: np.ndarray) -> None:
    """Guarda el float32 sin recortar (.npy) para la comparación numérica."""
    np.save(ruta, np.ascontiguousarray(imagen, dtype=np.float32))


def cargar_crudo(ruta: str, ancho: int = 0, alto: int = 0) -> np.ndarray:
    """Lee un .npy (Python) o un .f32 (volcado binario de C/CUDA C, requiere ancho y alto)."""
    if ruta.endswith(".npy"):
        return np.load(ruta)
    return np.fromfile(ruta, dtype=np.float32).reshape(alto, ancho)
