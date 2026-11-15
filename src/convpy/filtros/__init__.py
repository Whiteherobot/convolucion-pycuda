"""SECCIÓN 2 · LOS TRES FILTROS. Bloque 1 - Michael Lata.

Un archivo por filtro (media.py, nitidez.py, laplaciano.py), cada uno con su máscara y la
explicación de qué hace. Mismos coeficientes que practica01/src/modelo/mascara.cpp, como
float32 (k, k) C-contiguo. Un filtro solo es su máscara: la operación (convolución) es la misma
para los tres y está en convpy/convolucion/.
"""
import numpy as np

from . import laplaciano, media, nitidez

CATALOGO = {m.NOMBRE: m for m in (media, nitidez, laplaciano)}


def crear(nombre: str, k: int = 3) -> np.ndarray:
    """'media' (k x k impar) | 'nitidez' | 'laplaciano' (solo 3x3)."""
    if nombre not in CATALOGO:
        raise ValueError(f"filtro desconocido: {nombre} (usar {', '.join(CATALOGO)})")
    return CATALOGO[nombre].mascara(k)
