"""Convolución en GPU con PyCUDA. Bloque 2 - Jordi Romero.

Kernel: kernels/convolucion.cu, adaptado de practica01/src/modelo/convolucion_gpu.cu
(un hilo por píxel, bloques e hilos solo en X, lectura de vecinos en memoria global).
"""
import atexit
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

MAX_HILOS_POR_BLOQUE = 1024
RUTA_KERNEL = Path(__file__).resolve().parent / "kernels" / "convolucion.cu"


@dataclass
class TiemposGPU:
    h2d_ms: float = 0.0     # CPU -> GPU (imagen + máscara)
    kernel_ms: float = 0.0  # solo el kernel, con eventos CUDA
    d2h_ms: float = 0.0     # GPU -> CPU
    total_ms: float = 0.0   # transferencias + kernel; mem_alloc y free quedan fuera, como en CUDA C

    def ancho_banda_gbs(self, bytes_movidos: int) -> float:
        """GB/s efectivos del kernel: bytes_movidos / tiempo del kernel.

        Se cuenta el tráfico mínimo con memoria global: leer la imagen una vez y escribir la
        salida una vez (2 * n * 4 bytes). Los accesos repetidos a vecinos los sirve la caché.
        """
        if self.kernel_ms <= 0:
            return 0.0
        return bytes_movidos / (self.kernel_ms / 1000.0) / 1e9


class ConvolucionPyCUDA:
    """Se crea una vez: inicializa el contexto y compila/carga el kernel (SourceModule).

    Esos dos tiempos se guardan aparte (contexto_ms, compilacion_ms) y no se repiten en
    cada medición del kernel.
    """

    contexto_ms: float
    compilacion_ms: float

    def __init__(self) -> None:
        inicio = time.perf_counter()
        cuda.init()
        self.contexto = cuda.Device(0).make_context()
        atexit.register(self.contexto.pop)
        self.contexto_ms = (time.perf_counter() - inicio) * 1000

        inicio = time.perf_counter()
        # cache_dir=False obliga a compilar siempre, así compilacion_ms es el costo real de nvcc.
        modulo = SourceModule(RUTA_KERNEL.read_text(), cache_dir=False)
        self.kernel = modulo.get_function("kernel_convolucion")
        self.compilacion_ms = (time.perf_counter() - inicio) * 1000

    def ejecutar(self, imagen: np.ndarray, mascara: np.ndarray, hilos: int = 256
                 ) -> tuple[np.ndarray, TiemposGPU]:
        """Reserva, copia, lanza el kernel con bloques = ceil(n / hilos), trae el resultado.

        Tipos: float32 contiguo para imagen/máscara/salida, np.int32 para ancho, alto y k.
        """
        if not 1 <= hilos <= MAX_HILOS_POR_BLOQUE:
            raise ValueError(f"hilos por bloque debe estar entre 1 y {MAX_HILOS_POR_BLOQUE}, "
                             f"se recibió {hilos}")
        for nombre, matriz in (("imagen", imagen), ("mascara", mascara)):
            if matriz.dtype != np.float32 or not matriz.flags["C_CONTIGUOUS"]:
                raise ValueError(f"{nombre} debe ser float32 y C-contigua")

        alto, ancho = imagen.shape
        k = mascara.shape[0]
        n = ancho * alto
        salida = np.zeros((alto, ancho), dtype=np.float32)
        tiempos = TiemposGPU()
        # Con 0 bloques el lanzamiento es inválido: una imagen vacía no tiene nada que calcular.
        if n == 0:
            return salida, tiempos
        bloques = (n + hilos - 1) // hilos

        dev_entrada = cuda.mem_alloc(imagen.nbytes)
        dev_salida = cuda.mem_alloc(salida.nbytes)
        dev_mascara = cuda.mem_alloc(mascara.nbytes)
        inicio, tras_copia_ida, tras_kernel, fin = (cuda.Event() for _ in range(4))

        inicio.record()
        cuda.memcpy_htod(dev_entrada, imagen)
        cuda.memcpy_htod(dev_mascara, mascara)
        tras_copia_ida.record()

        self.kernel(dev_entrada, dev_salida, np.int32(ancho), np.int32(alto),
                    dev_mascara, np.int32(k), block=(hilos, 1, 1), grid=(bloques, 1))
        tras_kernel.record()

        cuda.memcpy_dtoh(salida, dev_salida)
        fin.record()
        fin.synchronize()

        tiempos.h2d_ms = inicio.time_till(tras_copia_ida)
        tiempos.kernel_ms = tras_copia_ida.time_till(tras_kernel)
        tiempos.d2h_ms = tras_kernel.time_till(fin)
        tiempos.total_ms = inicio.time_till(fin)

        dev_entrada.free()
        dev_salida.free()
        dev_mascara.free()
        return salida, tiempos
