"""Máscaras de la práctica 01. Bloque 1 - Michael Lata.

Todas devuelven ndarray float32 (k, k) C-contiguo, mismos coeficientes que practica01/src/modelo/mascara.cpp.
"""
import numpy as np


def media(k: int) -> np.ndarray:
    """k x k con 1/(k*k). k impar (3, 9 y 21 en los experimentos)."""
    raise NotImplementedError


def nitidez() -> np.ndarray:
    """[0,-1,0; -1,5,-1; 0,-1,0]"""
    raise NotImplementedError


def laplaciano() -> np.ndarray:
    """[0,1,0; 1,-4,1; 0,1,0]"""
    raise NotImplementedError


def crear(nombre: str, k: int = 3) -> np.ndarray:
    """'media' | 'nitidez' | 'laplaciano'. Nitidez y laplaciano solo existen en 3x3."""
    raise NotImplementedError
