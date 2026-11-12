"""Convolución en GPU con PyCUDA. Bloque 2 - Jordi Romero.

Kernel: kernels/convolucion.cu, adaptado de practica01/src/modelo/convolucion_gpu.cu
(un hilo por píxel, bloques e hilos solo en X, lectura de vecinos en memoria global).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TiemposGPU:
    h2d_ms: float = 0.0     # CPU -> GPU (imagen + máscara)
    kernel_ms: float = 0.0  # solo el kernel, con eventos CUDA
    d2h_ms: float = 0.0     # GPU -> CPU
    total_ms: float = 0.0   # transferencias + kernel (decir en el informe si incluye mem_alloc)


class ConvolucionPyCUDA:
    """Se crea una vez: inicializa el contexto y compila/carga el kernel (SourceModule).

    Esos dos tiempos se guardan aparte (contexto_ms, compilacion_ms) y no se repiten en
    cada medición del kernel.
    """

    contexto_ms: float
    compilacion_ms: float

    def __init__(self) -> None:
        raise NotImplementedError

    def ejecutar(self, imagen: np.ndarray, mascara: np.ndarray, hilos: int = 256
                 ) -> tuple[np.ndarray, TiemposGPU]:
        """Reserva, copia, lanza el kernel con bloques = ceil(n / hilos), trae el resultado.

        Tipos: float32 contiguo para imagen/máscara/salida, np.int32 para ancho, alto y k.
        """
        raise NotImplementedError
