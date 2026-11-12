"""Programa principal. Bloque 1 - Michael Lata.

  python -m convpy filtrar <entrada> <salida> <filtro> <k> secuencial|pycuda [hilos]
  python -m convpy probar
  python -m convpy medir                (bloque 3)

`filtrar` guarda la imagen visible en <salida> y el float sin recortar en <salida>.npy.
"""
import sys
import time

from . import imagen, mascaras
from .secuencial import convolucion_secuencial

USO = __doc__


def filtrar(args: list[str]) -> None:
    entrada, salida, filtro, k, version = args[:5]
    hilos = int(args[5]) if len(args) > 5 else 256
    img = imagen.cargar_gris(entrada)
    mascara = mascaras.crear(filtro, int(k))
    print(f"{entrada}: {img.shape[1]}x{img.shape[0]}, {filtro} {k}x{k}, {version}")

    if version == "secuencial":
        inicio = time.perf_counter()
        resultado = convolucion_secuencial(img, mascara)
        print(f"convolución: {(time.perf_counter() - inicio) * 1000:.2f} ms")
    elif version == "pycuda":
        from .pycuda_conv import ConvolucionPyCUDA
        resultado, t = ConvolucionPyCUDA().ejecutar(img, mascara, hilos)
        print(f"CPU->GPU {t.h2d_ms:.3f} ms | kernel {t.kernel_ms:.3f} ms | "
              f"GPU->CPU {t.d2h_ms:.3f} ms | total {t.total_ms:.3f} ms ({hilos} hilos)")
    else:
        raise SystemExit(f"versión desconocida: {version}")

    imagen.guardar_visual(salida, resultado)
    imagen.guardar_crudo(salida + ".npy", resultado)
    print(f"guardado {salida}")


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["filtrar"] and len(args) >= 6:
        filtrar(args[1:])
    elif args[:1] == ["probar"]:
        import pytest
        raise SystemExit(pytest.main(["-v", "tests"]))
    elif args[:1] == ["medir"]:
        from . import experimentos
        experimentos.main(args[1:])
    else:
        print(USO)


if __name__ == "__main__":
    main()
