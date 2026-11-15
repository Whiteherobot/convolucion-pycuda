"""FILTRO 2 · NITIDEZ (sharpen). Resalta bordes y detalles.

Máscara fija 3x3:

     0 -1  0
    -1  5 -1
     0 -1  0

Equivale a: salida = píxel + (4*píxel - suma de los 4 vecinos) = original + "bordes".
En una zona plana los vecinos se cancelan y queda el mismo valor (suma de coeficientes = 1);
en un borde la diferencia se amplifica. Puede dar valores < 0 o > 255: se recortan solo al
guardar la imagen.
"""
import numpy as np

NOMBRE = "nitidez"
QUE_HACE = "resalta bordes (original + diferencia con los vecinos)"


def mascara(k: int = 3) -> np.ndarray:
    if k != 3:
        raise ValueError("nitidez solo existe en 3x3")
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
