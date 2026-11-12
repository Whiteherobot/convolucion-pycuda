"""Convolución secuencial en Python. Bloque 1 - Michael Lata.

Misma operación que convolucion_cpu de la práctica 01: recorre cada píxel de salida y suma la
vecindad k x k multiplicada por la máscara; fuera de la imagen se toma cero (no se suma).
Recorridos explícitos: NumPy solo guarda los datos. La acumulación es np.float32, como el
float de C, para que la precisión sea la misma en las cuatro versiones.
"""
import numpy as np


def convolucion_secuencial(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Devuelve un float32 (alto, ancho) nuevo, sin recortar. Entrada y salida separadas."""
    alto, ancho = imagen.shape
    k = mascara.shape[0]
    radio = k // 2
    salida = np.zeros((alto, ancho), dtype=np.float32)

    for fila in range(alto):
        for columna in range(ancho):
            suma = np.float32(0.0)
            for i in range(k):
                y = fila + i - radio
                if y < 0 or y >= alto:
                    continue
                for j in range(k):
                    x = columna + j - radio
                    if 0 <= x < ancho:
                        suma += imagen[y, x] * mascara[i, j]
            salida[fila, columna] = suma
    return salida
