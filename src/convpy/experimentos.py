"""Los cuatro experimentos obligatorios. Bloque 3 - Andrés Morocho.

| Experimento | Configuración                          | Versiones medidas aquí |
| 1 Filtros   | media, nitidez, laplaciano 3x3; 4000   | Python y PyCUDA        |
| 2 Resolución| media 9x9; 2000, 4000, 6000            | Python y PyCUDA        |
| 3 Máscara   | media 3x3, 9x9, 21x21; 4000            | Python y PyCUDA        |
| 4 Hilos     | media 9x9; 4000; 128, 256, 512         | PyCUDA                 |

Un factor a la vez; fuera del experimento de hilos se fija 256. C/C++ y CUDA C se miden con el
mismo binario de la práctica 01 (scripts/correr_todo.sh), en el mismo equipo y con el mismo
protocolo (5 repeticiones, 2 s de calentamiento GPU).

Python secuencial se mide una sola vez por configuración distinta: media 3x3 en 4000 está en
los exp. 1 y 3, y media 9x9 en 4000 en los exp. 2 y 3. Las filas se repiten en cada experimento
con una nota, y así se ahorran unos 10 min de CPU sin cambiar el dato.

Salidas en <resultados>/:
  mediciones.csv     una fila por repetición (COLUMNAS_CSV de medicion.py)
  validacion.csv     error máximo y medio de CUDA C, Python y PyCUDA contra C/C++, sin recortar
  ventanas_gpu.csv   hora de inicio y fin de cada bloque GPU medido (para cruzar con gpu_estado)
  gpu_estado.csv     nvidia-smi cada 200 ms durante toda la corrida
  pycuda_inicio.csv  costo único de crear el contexto y compilar el kernel

Uso: python -m convpy medir [--experimentos 1 2 3 4] [--sin-python] [--presupuesto-min 60]
"""
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import imagen, mascaras
from .medicion import (CALENTAMIENTO_GPU_S, REPETICIONES, Configuracion, EscritorCSV, MonitorGPU, hora,
                       medir_pycuda, medir_secuencial)
from .validacion import TOLERANCIA, errores

RAIZ = Path(__file__).resolve().parents[2]
HERRAMIENTA = RAIZ / "build" / "volcar_crudo"

# (número, nombre, [(filtro, k, resolución, hilos)], ¿incluye Python?)
EXPERIMENTOS = [
    (1, "Tipo de filtro", [("media", 3, 4000, 256), ("nitidez", 3, 4000, 256),
                           ("laplaciano", 3, 4000, 256)], True),
    (2, "Resolución", [("media", 9, 2000, 256), ("media", 9, 4000, 256),
                       ("media", 9, 6000, 256)], True),
    (3, "Tamaño de máscara", [("media", 3, 4000, 256), ("media", 9, 4000, 256),
                              ("media", 21, 4000, 256)], True),
    (4, "Hilos por bloque", [("media", 9, 4000, 128), ("media", 9, 4000, 256),
                             ("media", 9, 4000, 512)], False),
]

COLUMNAS_VALIDACION = ["experimento", "filtro", "resolucion", "k", "hilos", "version",
                       "referencia", "error_max", "error_medio", "dentro_tolerancia"]
COLUMNAS_VENTANAS = ["experimento", "version", "filtro", "resolucion", "k", "hilos",
                     "inicio", "fin"]


def _argumentos(args: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m convpy medir")
    p.add_argument("--experimentos", type=int, nargs="+", default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4])
    p.add_argument("--sin-python", action="store_true",
                   help="no medir Python secuencial (prueba rápida de la parte GPU)")
    p.add_argument("--presupuesto-min", type=float, default=60.0,
                   help="minutos máximos de Python secuencial por configuración (def. 60)")
    p.add_argument("--repeticiones", type=int, default=REPETICIONES)
    p.add_argument("--calentamiento-s", type=float, default=CALENTAMIENTO_GPU_S,
                   help="segundos de trabajo GPU continuo antes de medir (def. 2)")
    p.add_argument("--datos", default=str(RAIZ / "datos"))
    p.add_argument("--resultados", default=str(RAIZ / "resultados"))
    return p.parse_args(args)


class Referencia:
    """Salida float sin recortar de C/C++ y CUDA C (herramientas/volcar_crudo, bloque 2).

    Se calcula una vez por (filtro, k, imagen). Si la herramienta no está, la referencia pasa a
    ser Python secuencial y se dice en la columna `referencia` de validacion.csv."""

    def __init__(self, carpeta: Path):
        self.carpeta = carpeta
        self.cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.disponible = HERRAMIENTA.exists() or self._compilar()

    @staticmethod
    def _compilar() -> bool:
        if shutil.which("nvcc") is None:
            print("[AVISO] sin nvcc: la validación usará Python secuencial como referencia")
            return False
        try:
            subprocess.run([str(RAIZ / "herramientas" / "compilar.sh")], check=True)
            return True
        except (OSError, subprocess.CalledProcessError) as fallo:
            print(f"[AVISO] no se pudo compilar volcar_crudo ({fallo})")
            return False

    def obtener(self, ruta_pgm: str, filtro: str, k: int, alto: int, ancho: int):
        """Devuelve (c_cpu, c_gpu) o None si no hay herramienta."""
        if not self.disponible:
            return None
        clave = (ruta_pgm, filtro, k)
        if clave not in self.cache:
            self.carpeta.mkdir(parents=True, exist_ok=True)
            prefijo = self.carpeta / f"{Path(ruta_pgm).stem}_{filtro}{k}"
            subprocess.run([str(HERRAMIENTA), ruta_pgm, filtro, str(k), str(prefijo)], check=True)
            par = tuple(imagen.cargar_crudo(f"{prefijo}_{v}.f32", ancho, alto)
                        for v in ("cpu", "gpu"))
            for v in ("cpu", "gpu"):  # 6000x6000 son 144 MB por archivo
                Path(f"{prefijo}_{v}.f32").unlink()
            self.cache = {clave: par}  # solo la última: las configuraciones van agrupadas
        return self.cache[clave]


def _validar(c: Configuracion, ruta_pgm: str, ref: Referencia, salidas: dict) -> list[dict]:
    filas = []
    c_ref = ref.obtener(ruta_pgm, c.filtro, c.k, *c.imagen.shape)
    if c_ref is not None:
        referencia, nombre_ref = c_ref[0], "c_cpu"
        salidas = {"cuda_c": c_ref[1], **salidas}
    elif "python" in salidas:
        referencia, nombre_ref = salidas.pop("python"), "python"
    else:
        return filas
    for version, salida in salidas.items():
        maximo, medio = errores(referencia, salida)
        ok = maximo < TOLERANCIA
        print(f"      {version:7s} vs {nombre_ref}: error max {maximo:.2e}, medio {medio:.2e} "
              f"{'OK' if ok else 'FUERA DE TOLERANCIA'}")
        filas.append({"experimento": c.experimento, "filtro": c.filtro,
                      "resolucion": c.resolucion, "k": c.k, "hilos": c.hilos,
                      "version": version, "referencia": nombre_ref,
                      "error_max": f"{maximo:.3e}", "error_medio": f"{medio:.3e}",
                      "dentro_tolerancia": ok})
    return filas


def main(args: list[str]) -> None:
    op = _argumentos(args)
    salida = Path(op.resultados)
    salida.mkdir(parents=True, exist_ok=True)
    presupuesto_s = op.presupuesto_min * 60

    from .pycuda_conv import ConvolucionPyCUDA  # aquí y no arriba: importar PyCUDA exige GPU

    print(f"=== Experimentos {op.experimentos}: {op.repeticiones} repeticiones, "
          f"Python {'omitido' if op.sin_python else f'hasta {op.presupuesto_min:.0f} min por config.'} ===")
    gpu = ConvolucionPyCUDA()
    print(f"PyCUDA: contexto {gpu.contexto_ms:.1f} ms, compilación del kernel "
          f"{gpu.compilacion_ms:.1f} ms (una vez, fuera de las mediciones)")
    with open(salida / "pycuda_inicio.csv", "w", newline="") as f:
        csv.writer(f).writerows([["contexto_ms", "compilacion_ms"],
                                 [f"{gpu.contexto_ms:.3f}", f"{gpu.compilacion_ms:.3f}"]])

    ref = Referencia(salida / "crudos")
    mediciones = EscritorCSV(salida / "mediciones.csv")
    validacion = EscritorCSV(salida / "validacion.csv", COLUMNAS_VALIDACION)
    ventanas = EscritorCSV(salida / "ventanas_gpu.csv", COLUMNAS_VENTANAS)
    imagenes: dict[int, np.ndarray] = {}
    python_hecho: dict[tuple, tuple[int, object]] = {}  # (filtro, k, res) -> (exp, Resultado)
    fuera_tolerancia = 0
    inicio_total = time.perf_counter()

    with MonitorGPU(salida / "gpu_estado.csv"):
        for numero, nombre, configuraciones, con_python in EXPERIMENTOS:
            if numero not in op.experimentos:
                continue
            print(f"\n[Exp {numero}] {nombre}  ({hora()})")
            for filtro, k, lado, hilos in configuraciones:
                ruta = f"{op.datos}/imagen_{lado}.pgm"
                if lado not in imagenes:
                    if not Path(ruta).exists():
                        print(f"  [AVISO] falta {ruta}: se omite")
                        continue
                    imagenes = {lado: imagen.cargar_gris(ruta)}  # una a la vez: 6000 = 144 MB
                c = Configuracion(numero, filtro, k, imagenes[lado], mascaras.crear(filtro, k), hilos)
                print(f"    {c.describir()}, {hilos} hilos")
                salidas = {}

                if con_python and not op.sin_python:
                    clave = (filtro, k, lado)
                    if clave in python_hecho:
                        exp_origen, r = python_hecho[clave]
                        print(f"      python: se reutiliza la medición del exp {exp_origen}")
                        reuso = f"misma medición que el exp {exp_origen}"
                        filas = [{**f, "experimento": numero,
                                  "nota": "; ".join(filter(None, [f["nota"], reuso]))}
                                 for f in r.filas]
                    else:
                        r = medir_secuencial(c, presupuesto_s, op.repeticiones)
                        python_hecho[clave] = (numero, r)
                        filas = r.filas
                    mediciones.escribir(filas)
                    salidas["python"] = r.salida

                r = medir_pycuda(gpu, c, op.repeticiones, op.calentamiento_s)
                mediciones.escribir(r.filas)
                ventanas.escribir([{"experimento": numero, "version": "pycuda", "filtro": filtro,
                                    "resolucion": c.resolucion, "k": k, "hilos": hilos,
                                    "inicio": r.inicio, "fin": r.fin}])
                salidas["pycuda"] = r.salida

                filas = _validar(c, ruta, ref, salidas)
                fuera_tolerancia += sum(not f["dentro_tolerancia"] for f in filas)
                validacion.escribir(filas)

    for escritor in (mediciones, validacion, ventanas):
        escritor.cerrar()
    print(f"\n=== Terminado en {(time.perf_counter() - inicio_total) / 60:.1f} min: "
          f"{fuera_tolerancia} comparaciones fuera de tolerancia. Resultados en {salida}/ ===")
    if fuera_tolerancia:
        raise SystemExit(1)
