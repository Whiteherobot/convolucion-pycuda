"""FILTRO 1 · MEDIA (promedio). Suaviza / desenfoca.

Máscara k x k con todos los coeficientes iguales a 1/(k*k). Ejemplo 3x3:

    1/9 1/9 1/9
    1/9 1/9 1/9
    1/9 1/9 1/9

Cada píxel de salida es el promedio de su vecindad, así que el ruido y los detalles finos se
diluyen. La suma de coeficientes es 1: una zona plana conserva su brillo (por eso la salida
nunca sale de 0-255). Es el único filtro con tamaño variable: el exp. 3 usa 3x3, 9x9 y 21x21.
Coste: k*k multiplicaciones por píxel (9 en 3x3, 441 en 21x21: 49 veces más trabajo).
"""
import numpy as np

NOMBRE = "media"
QUE_HACE = "suaviza (promedia la vecindad)"


def mascara(k: int = 3) -> np.ndarray:
    if k < 1 or k % 2 == 0:
        raise ValueError(f"k debe ser impar y positivo, se recibió {k}")
    return np.full((k, k), np.float32(1.0) / np.float32(k * k), dtype=np.float32)
