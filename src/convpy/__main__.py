"""Programa principal (punto de entrada). Bloque 1 - Michael Lata.

  python -m convpy demo <imagen o URL> [--k 9] [--hilos 256] [--secuencial] [--ver]
  python -m convpy filtrar <entrada> <salida> <filtro> <k> secuencial|pycuda [hilos]
  python -m convpy probar
  python -m convpy medir [opciones]     (bloque 3, ver evaluacion/experimentos.py)

`demo` aplica los tres filtros a cualquier imagen y explica cada paso (para la exposición).
`filtrar` aplica un solo filtro; guarda la imagen visible en <salida> y el float sin recortar
en <salida>.npy.

Mapa del paquete (una sección por carpeta):
  imagenes/     [1] cargar cualquier imagen (o URL) y pasarla a gris float32; guardar
  filtros/      [2] media.py, nitidez.py, laplaciano.py: la máscara de cada filtro
  convolucion/  [3] secuencial.py (Python puro) y pycuda_gpu.py + kernels/convolucion.cu
  evaluacion/   [4] validacion.py, medicion.py, experimentos.py
"""
import sys
import time

import numpy as np

from . import filtros, imagenes
from .convolucion.secuencial import convolucion

USO = __doc__


def filtrar(args: list[str]) -> None:
    entrada, salida, filtro, k, version = args[:5]
    hilos = int(args[5]) if len(args) > 5 else 256
    img = imagenes.cargar_gris(entrada)                      # [1] carga
    mascara = filtros.crear(filtro, int(k))                  # [2] filtro
    print(f"{entrada}: {img.shape[1]}x{img.shape[0]}, {filtro} {k}x{k}, {version}")

    if version == "secuencial":                              # [3] convolución
        # Pasar a listas es preparación de datos: queda fuera del tiempo medido.
        img_lista, mascara_lista = img.tolist(), mascara.tolist()
        inicio = time.perf_counter()
        resultado = convolucion(img_lista, mascara_lista)
        print(f"convolución: {(time.perf_counter() - inicio) * 1000:.2f} ms")
        resultado = np.array(resultado, dtype=np.float32)
    elif version == "pycuda":
        from .convolucion.pycuda_gpu import ConvolucionPyCUDA
        resultado, t = ConvolucionPyCUDA().ejecutar(img, mascara, hilos)
        print(f"CPU->GPU {t.h2d_ms:.3f} ms | kernel {t.kernel_ms:.3f} ms | "
              f"GPU->CPU {t.d2h_ms:.3f} ms | total {t.total_ms:.3f} ms ({hilos} hilos)")
    else:
        raise SystemExit(f"versión desconocida: {version}")

    imagenes.guardar_visual(salida, resultado)               # [4] guardar
    imagenes.guardar_crudo(salida + ".npy", resultado)
    print(f"guardado {salida}")


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["demo"]:
        from . import demo
        demo.main(args[1:])
    elif args[:1] == ["filtrar"] and len(args) >= 6:
        filtrar(args[1:])
    elif args[:1] == ["probar"]:
        import pytest
        raise SystemExit(pytest.main(["-v", "tests"]))
    elif args[:1] == ["medir"]:
        from .evaluacion import experimentos
        experimentos.main(args[1:])
    else:
        print(USO)


if __name__ == "__main__":
    main()
