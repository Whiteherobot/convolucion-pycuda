"""Lectura, conversión y escritura de imágenes. Bloque 1 - Michael Lata.

Contrato con los demás bloques: toda imagen viaja como ndarray float32, 2D (alto, ancho),
C-contiguo (np.ascontiguousarray). El recorte a 0-255 solo se hace al guardar para ver.
"""
import numpy as np


def cargar_gris(ruta: str) -> np.ndarray:
    """Lee la imagen, la pasa a escala de grises y devuelve float32 (alto, ancho) contiguo."""
    raise NotImplementedError


def guardar_visual(ruta: str, imagen: np.ndarray) -> None:
    """Redondea, recorta a 0-255 y guarda como uint8 (PGM o PNG según la extensión)."""
    raise NotImplementedError


def guardar_crudo(ruta: str, imagen: np.ndarray) -> None:
    """Guarda el float32 sin recortar (.npy) para la comparación numérica."""
    raise NotImplementedError


def cargar_crudo(ruta: str) -> np.ndarray:
    """Lee un .npy (Python) o un .f32 (volcado de C/CUDA C) como float32 (alto, ancho)."""
    raise NotImplementedError
