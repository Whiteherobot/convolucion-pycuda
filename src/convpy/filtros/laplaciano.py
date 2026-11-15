"""FILTRO 3 · LAPLACIANO. Detecta bordes (segunda derivada).

Máscara fija 3x3:

     0  1  0
     1 -4  1
     0  1  0

Suma de coeficientes = 0: en una zona plana la salida es 0 (negro). Solo responde donde la
intensidad cambia bruscamente, y lo hace con valores positivos y negativos a cada lado del
borde. Al guardar, los negativos se recortan a 0, por eso la imagen se ve casi toda negra con
bordes finos; el valor real (sin recortar) es el que se compara entre versiones.
Es la "parte de bordes" de la nitidez: nitidez = identidad - laplaciano.
"""
import numpy as np

NOMBRE = "laplaciano"
QUE_HACE = "detecta bordes (zonas planas -> 0)"


def mascara(k: int = 3) -> np.ndarray:
    if k != 3:
        raise ValueError("laplaciano solo existe en 3x3")
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
