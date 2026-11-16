"""Lectura, conversión y escritura de imágenes. Bloque 1 - Michael Lata.

Contrato con los demás bloques: toda imagen viaja como ndarray float32, 2D (alto, ancho),
C-contiguo. El recorte a 0-255 solo se hace al guardar para ver, igual que en la práctica 01.
"""
import numpy as np
from PIL import Image


def cargar_gris(ruta: str) -> np.ndarray:
    """Lee la imagen, la pasa a escala de grises ('L') y devuelve float32 (alto, ancho) contiguo."""
    with Image.open(ruta) as img:
        gris = img.convert("L")
        return np.ascontiguousarray(np.asarray(gris, dtype=np.float32))


def a_rango_visual(imagen: np.ndarray) -> np.ndarray:
    """Igual que a_rango_visual de la P01: recorta a [0, 255] y redondea sumando 0.5."""
    return (np.clip(imagen, 0.0, 255.0) + np.float32(0.5)).astype(np.uint8)


def guardar_visual(ruta: str, imagen: np.ndarray) -> None:
    """Guarda como uint8 (PGM o PNG según la extensión)."""
    Image.fromarray(a_rango_visual(imagen), mode="L").save(ruta)


def guardar_crudo(ruta: str, imagen: np.ndarray) -> None:
    """Guarda el float32 sin recortar (.npy) para la comparación numérica."""
    np.save(ruta, np.ascontiguousarray(imagen, dtype=np.float32))


def cargar_crudo(ruta: str, ancho: int = 0, alto: int = 0) -> np.ndarray:
    """Lee un .npy (Python) o un .f32 (volcado binario de C/CUDA C, requiere ancho y alto)."""
    if ruta.endswith(".npy"):
        return np.load(ruta)
    return np.fromfile(ruta, dtype=np.float32).reshape(alto, ancho)
