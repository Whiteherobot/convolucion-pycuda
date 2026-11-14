"""Medición de una configuración y escritura del CSV. Bloque 3 - Andrés Morocho.

Mismo orden que medicion.cu de la práctica 01, y por el mismo motivo:
  1. Python secuencial: 1 calentamiento + REPETICIONES medidas, seguidas.
  2. PyCUDA: 1 llamada «fría» (se registra) y calentamiento de al menos CALENTAMIENTO_GPU_S
     de trabajo continuo. En la P01, si la GPU esperaba a la CPU bajaba de reloj y de enlace
     PCIe, y cada repetición caía en un estado distinto.
  3. PyCUDA: REPETICIONES medidas seguidas; los tiempos se guardan en memoria y el CSV se
     escribe al final, para no dejar a la GPU sin trabajo entre repeticiones.

Alcance de cada tiempo:
  - Python: time.perf_counter() solo alrededor de secuencial.convolucion(). Pasar la imagen a
    listas (.tolist()) es preparación de datos y queda fuera, igual que leer el archivo.
  - PyCUDA: TiemposGPU de ConvolucionPyCUDA.ejecutar (eventos CUDA). h2d incluye imagen y
    máscara; total = h2d + kernel + d2h; mem_alloc/free fuera, como en CUDA C. El contexto y
    la compilación (contexto_ms, compilacion_ms) se miden una vez y se reportan aparte.

Columna `repeticion`: 0 es la llamada sin calentar (estado «calentamiento»); 1..N son las
medidas (estado «ok»). Las medianas y aceleraciones solo usan estado «ok». La fila 0 sirve
para la comparación con y sin calentamiento que pidió mantener el docente.
"""
import csv
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .secuencial import convolucion

REPETICIONES = 5
CALENTAMIENTO_GPU_S = 2.0
BYTES_FLOAT = 4

COLUMNAS_CSV = [
    "experimento", "version", "filtro", "resolucion", "k", "hilos", "bloques",
    "repeticion", "cpu_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms", "ancho_banda_gbs",
    "unidad", "estado", "nota",  # estado: ok | calentamiento | no_completada (motivo en nota)
]

# Lo mismo que registraba la P01 en gpu_estado.csv, cada 200 ms.
CONSULTA_GPU = ("timestamp,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,"
                "pcie.link.width.current,power.draw,temperature.gpu")


@dataclass
class Configuracion:
    experimento: int
    filtro: str
    k: int
    imagen: np.ndarray
    mascara: np.ndarray
    hilos: int = 256

    @property
    def resolucion(self) -> str:
        alto, ancho = self.imagen.shape
        return f"{ancho}x{alto}"

    @property
    def pixeles(self) -> int:
        return int(self.imagen.size)

    @property
    def bloques(self) -> int:
        return (self.pixeles + self.hilos - 1) // self.hilos

    @property
    def bytes_kernel(self) -> int:
        """Tráfico mínimo con memoria global: leer la imagen y escribir la salida una vez."""
        return 2 * self.pixeles * BYTES_FLOAT

    def describir(self) -> str:
        return f"{self.filtro} {self.k}x{self.k} {self.resolucion}"


@dataclass
class Resultado:
    """Filas del CSV de una versión en una configuración, y la última salida (para validar)."""
    filas: list[dict] = field(default_factory=list)
    salida: np.ndarray | None = None
    inicio: str = ""  # hora de la ventana medida, para ubicarla en gpu_estado.csv
    fin: str = ""


def hora() -> str:
    """Hora local con milisegundos, comparable con la columna timestamp de nvidia-smi."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _fila(c: Configuracion, version: str, repeticion: int, estado: str, *, cpu_ms=0.0,
          tiempos=None, gpu=False, nota="") -> dict:
    fila = {
        "experimento": c.experimento, "version": version, "filtro": c.filtro,
        "resolucion": c.resolucion, "k": c.k,
        "hilos": c.hilos if gpu else 0, "bloques": c.bloques if gpu else 0,
        "repeticion": repeticion, "cpu_ms": round(cpu_ms, 4),
        "h2d_ms": 0.0, "kernel_ms": 0.0, "d2h_ms": 0.0, "total_ms": 0.0, "ancho_banda_gbs": 0.0,
        "unidad": "ms", "estado": estado, "nota": nota,
    }
    if tiempos is not None:
        fila.update(h2d_ms=round(tiempos.h2d_ms, 4), kernel_ms=round(tiempos.kernel_ms, 4),
                    d2h_ms=round(tiempos.d2h_ms, 4), total_ms=round(tiempos.total_ms, 4),
                    ancho_banda_gbs=round(tiempos.ancho_banda_gbs(c.bytes_kernel), 2))
    return fila


def medir_secuencial(c: Configuracion, presupuesto_s: float,
                     repeticiones: int = REPETICIONES) -> Resultado:
    """Python secuencial. Si el calentamiento indica que las repeticiones no caben en
    presupuesto_s, no se mide más: se registra el tiempo transcurrido y la fila queda
    «no_completada» (la guía pide no inventar tiempos de lo que no terminó)."""
    imagen, mascara = c.imagen.tolist(), c.mascara.tolist()
    r = Resultado(inicio=hora())

    t0 = time.perf_counter()
    salida = convolucion(imagen, mascara)
    calentamiento_ms = (time.perf_counter() - t0) * 1000
    r.filas.append(_fila(c, "python", 0, "calentamiento", cpu_ms=calentamiento_ms))
    print(f"      python calentamiento: {calentamiento_ms / 1000:.1f} s")

    estimado_s = calentamiento_ms / 1000 * repeticiones
    if estimado_s > presupuesto_s:
        nota = (f"{repeticiones} repeticiones tomarían ~{estimado_s / 60:.0f} min, más que el "
                f"presupuesto de {presupuesto_s / 60:.0f} min")
        r.filas.append(_fila(c, "python", 1, "no_completada", cpu_ms=calentamiento_ms, nota=nota))
        print(f"      [NO COMPLETADA] {nota}")
    else:
        for rep in range(1, repeticiones + 1):
            t0 = time.perf_counter()
            salida = convolucion(imagen, mascara)
            cpu_ms = (time.perf_counter() - t0) * 1000
            r.filas.append(_fila(c, "python", rep, "ok", cpu_ms=cpu_ms))
        medidos = [f["cpu_ms"] / 1000 for f in r.filas[1:]]
        print(f"      python: mediana {np.median(medidos):.2f} s "
              f"(min {min(medidos):.2f}, max {max(medidos):.2f})")

    r.fin = hora()
    r.salida = np.array(salida, dtype=np.float32)
    return r


def medir_pycuda(gpu, c: Configuracion, repeticiones: int = REPETICIONES,
                 calentamiento_s: float = CALENTAMIENTO_GPU_S) -> Resultado:
    """PyCUDA: llamada fría, calentamiento por tiempo y repeticiones seguidas."""
    r = Resultado()
    _, frio = gpu.ejecutar(c.imagen, c.mascara, c.hilos)
    r.filas.append(_fila(c, "pycuda", 0, "calentamiento", tiempos=frio, gpu=True))

    # Por tiempo y no por número de llamadas: una llamada va de ~10 ms (3x3) a ~250 ms (21x21).
    inicio = time.perf_counter()
    llamadas, ultimo = 1, frio
    while time.perf_counter() - inicio < calentamiento_s:
        _, ultimo = gpu.ejecutar(c.imagen, c.mascara, c.hilos)
        llamadas += 1
    print(f"      calentamiento GPU: {llamadas} llamadas en "
          f"{(time.perf_counter() - inicio) * 1000:.0f} ms, kernel "
          f"{frio.kernel_ms:.3f} -> {ultimo.kernel_ms:.3f} ms")

    medidos = []
    r.inicio = hora()
    for _ in range(repeticiones):
        r.salida, t = gpu.ejecutar(c.imagen, c.mascara, c.hilos)
        medidos.append(t)
    r.fin = hora()

    for rep, t in enumerate(medidos, start=1):
        r.filas.append(_fila(c, "pycuda", rep, "ok", tiempos=t, gpu=True))
    kernel = np.median([t.kernel_ms for t in medidos])
    total = np.median([t.total_ms for t in medidos])
    banda = np.median([t.ancho_banda_gbs(c.bytes_kernel) for t in medidos])
    print(f"      pycuda {c.hilos} hilos x {c.bloques} bloques: kernel {kernel:.3f} ms, "
          f"total {total:.3f} ms, {banda:.1f} GB/s  (medida {r.inicio} a {r.fin})")
    return r


class EscritorCSV:
    """CSV que se escribe y vacía al terminar cada configuración (nunca entre repeticiones):
    si la corrida larga de Python se corta, lo ya medido no se pierde."""

    def __init__(self, ruta, columnas=COLUMNAS_CSV):
        self.archivo = open(ruta, "w", newline="")
        self.escritor = csv.DictWriter(self.archivo, fieldnames=columnas)
        self.escritor.writeheader()

    def escribir(self, filas: list[dict]) -> None:
        self.escritor.writerows(filas)
        self.archivo.flush()

    def cerrar(self) -> None:
        self.archivo.close()


class MonitorGPU:
    """Registra el estado de la GPU con nvidia-smi cada 200 ms mientras dura el bloque `with`.
    Sin nvidia-smi (por ejemplo, en un equipo sin NVIDIA) avisa y no hace nada."""

    def __init__(self, ruta, intervalo_ms: int = 200):
        self.ruta, self.intervalo_ms, self.proceso = ruta, intervalo_ms, None

    def __enter__(self):
        if shutil.which("nvidia-smi") is None:
            print("[AVISO] nvidia-smi no está disponible: no se registra el estado de la GPU")
            return self
        self.proceso = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={CONSULTA_GPU}", "--format=csv",
             f"--loop-ms={self.intervalo_ms}", "-f", str(self.ruta)])
        return self

    def __exit__(self, *_):
        if self.proceso is not None:
            self.proceso.terminate()
            self.proceso.wait()
