"""Convolución secuencial en Python, hecha a mano. Bloque 1 - Michael Lata.

Misma operación que convolucion_cpu de la práctica 01: recorre cada píxel de salida y suma la
vecindad k x k multiplicada por la máscara; fuera de la imagen se toma cero (no se suma).
El cálculo no usa NumPy: la imagen y la máscara se pasan a listas de Python y se recorren con
bucles. NumPy solo se usa en el borde, para recibir y devolver los datos (como en el resto del programa).
"""
import numpy as np


def convolucion(imagen: list[list[float]], mascara: list[list[float]]) -> list[list[float]]:
    """Convolución con listas de Python puro. Entrada y salida separadas."""
    alto = len(imagen)
    ancho = len(imagen[0])
    k = len(mascara)
    radio = k // 2
    salida = [[0.0] * ancho for _ in range(alto)]

    for fila in range(alto):
        for columna in range(ancho):
            suma = 0.0
            for i in range(k):
                y = fila + i - radio
                if y < 0 or y >= alto:
                    continue
                for j in range(k):
                    x = columna + j - radio
                    if 0 <= x < ancho:
                        suma += imagen[y][x] * mascara[i][j]
            salida[fila][columna] = suma
    return salida


def convolucion_secuencial(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Pasa los datos a listas, convoluciona a mano y devuelve float32 (alto, ancho) sin recortar."""
    resultado = convolucion(imagen.tolist(), mascara.tolist())
    return np.array(resultado, dtype=np.float32)
