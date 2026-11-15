"""Demostración en clase: carga cualquier imagen y le aplica los tres filtros. Bloque 1 - Michael Lata.

  python -m convpy demo <imagen o URL> [--k 9] [--hilos 256] [--secuencial] [--ver]

Recorre las cuatro secciones del programa en orden y lo imprime paso a paso:
  [1] imagenes/     cargar (o descargar) y pasar a grises float32
  [2] filtros/      crear la máscara de cada filtro
  [3] convolucion/  aplicarla con PyCUDA (y, con --secuencial, también en Python y comparar)
  [4] imagenes/     recortar a 0-255 y guardar en resultados/demo/
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from . import filtros, imagenes
from .convolucion.secuencial import convolucion_secuencial
from .evaluacion.validacion import TOLERANCIA, errores

SALIDA = imagenes.carga.RAIZ / "resultados" / "demo"
# Python secuencial hace ~1e7 multiplicaciones por segundo (13,4 s para 4000x4000 con 3x3).
# Si la comparación tardaría más que esto, se hace sobre un recorte central.
LIMITE_SECUENCIAL_S = 20
LADO_RECORTE = 512


def _corta(ruta: Path) -> str:
    try:
        return str(ruta.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(ruta)


def _titulo(texto: str) -> None:
    print(f"\n{texto}\n" + "-" * len(texto))


def _cargar(ruta: str) -> tuple[np.ndarray, str]:
    _titulo("[1] CARGA  (convpy/imagenes/carga.py)")
    if imagenes.es_url(ruta):
        print(f"descargando {ruta}")
        try:
            ruta = str(imagenes.descargar(ruta))
        except OSError as motivo:  # HTTPError y URLError heredan de OSError
            raise SystemExit(f"no se pudo descargar ({motivo}). Bajarla con el navegador y "
                             "pasar la ruta del archivo.") from None
        print(f"guardada en {_corta(Path(ruta))}")
    print(f"archivo:   {_corta(Path(ruta))}  ({imagenes.describir(ruta)})")
    img = imagenes.cargar_gris(ruta)
    print(f"en gris:   {img.shape[1]}x{img.shape[0]} = {img.size:,} píxeles, float32, "
          f"valores {img.min():.0f}..{img.max():.0f}")
    return img, Path(ruta).stem


def _mascaras(k_media: int) -> dict[str, np.ndarray]:
    _titulo("[2] FILTROS  (convpy/filtros/)")
    resultado = {}
    for nombre, modulo in filtros.CATALOGO.items():
        k = k_media if nombre == "media" else 3
        m = filtros.crear(nombre, k)
        resultado[nombre] = m
        print(f"{nombre} {k}x{k}: {modulo.QUE_HACE}  (suma de coeficientes = {m.sum():.0f})")
        if k <= 3:
            for fila in m:
                print("    " + " ".join(f"{v:5.2f}" for v in fila))
        else:
            print(f"    todos los coeficientes = 1/{k * k} = {m[0, 0]:.5f}")
    return resultado


def _gpu(img: np.ndarray, masks: dict[str, np.ndarray], hilos: int):
    _titulo("[3] CONVOLUCIÓN EN GPU  (convpy/convolucion/pycuda_gpu.py)")
    from .convolucion.pycuda_gpu import ConvolucionPyCUDA  # importar PyCUDA exige GPU
    gpu = ConvolucionPyCUDA()
    print(f"contexto CUDA {gpu.contexto_ms:.0f} ms + compilar kernel {gpu.compilacion_ms:.0f} ms "
          "(una sola vez)")
    n = img.size
    bloques = (n + hilos - 1) // hilos
    print(f"lanzamiento: {n:,} hilos útiles = {bloques:,} bloques x {hilos} hilos "
          f"(sobran {bloques * hilos - n} hilos en el último bloque)")
    gpu.ejecutar(img, masks["media"], hilos)  # 1 llamada para despertar la GPU
    salidas = {}
    for nombre, m in masks.items():
        salida, t = gpu.ejecutar(img, m, hilos)
        salidas[nombre] = salida
        print(f"{nombre:>10}: CPU->GPU {t.h2d_ms:7.3f} ms | kernel {t.kernel_ms:7.3f} ms | "
              f"GPU->CPU {t.d2h_ms:7.3f} ms | total {t.total_ms:7.3f} ms")
    return salidas, gpu


def _secuencial(img: np.ndarray, masks: dict[str, np.ndarray], gpu_salidas: dict | None,
                gpu=None) -> dict[str, np.ndarray]:
    _titulo("[3b] CONVOLUCIÓN SECUENCIAL EN PYTHON  (convpy/convolucion/secuencial.py)")
    trabajo = img.size * sum(m.size for m in masks.values())
    zona = (slice(None), slice(None))
    if trabajo / 1e7 > LIMITE_SECUENCIAL_S:
        alto, ancho = img.shape
        y0, x0 = max(0, (alto - LADO_RECORTE) // 2), max(0, (ancho - LADO_RECORTE) // 2)
        zona = (slice(y0, y0 + LADO_RECORTE), slice(x0, x0 + LADO_RECORTE))
        print(f"la imagen completa tardaría ~{trabajo / 1e7:.0f} s en Python: se usa un recorte "
              f"central de {LADO_RECORTE}x{LADO_RECORTE}")
    parte = np.ascontiguousarray(img[zona])
    salidas = {}
    for nombre, m in masks.items():
        inicio = time.perf_counter()
        salida = convolucion_secuencial(parte, m)
        ms = (time.perf_counter() - inicio) * 1000
        salidas[nombre] = salida
        linea = f"{nombre:>10}: {ms:10.1f} ms"
        if gpu_salidas is not None:
            if zona == (slice(None), slice(None)):
                referencia = gpu_salidas[nombre]
            else:  # en el recorte, la GPU calcula sobre el mismo recorte para comparar
                referencia = gpu.ejecutar(parte, m)[0]
            maximo, medio = errores(salida, referencia)
            estado = "IGUALES" if maximo <= TOLERANCIA else "DISTINTAS"
            linea += f" | error máx. vs GPU {maximo:.2e} (tolerancia {TOLERANCIA}) -> {estado}"
        print(linea)
    return salidas if zona == (slice(None), slice(None)) else {}


def _guardar(img: np.ndarray, salidas: dict[str, np.ndarray], base: str, k: int,
             carpeta: Path) -> Path:
    _titulo("[4] GUARDAR  (convpy/imagenes/guardado.py)")
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, s in salidas.items():
        ruta = carpeta / f"{base}_{nombre}.png"
        imagenes.guardar_visual(str(ruta), s)
        fuera = 100 * ((s < 0) | (s > 255)).mean()
        print(f"{nombre:>10}: rango real {s.min():8.1f} .. {s.max():7.1f} | "
              f"{fuera:5.1f} % recortado a 0-255 | {_corta(ruta)}")
    return _panel(img, salidas, base, k, carpeta)


def _panel(img: np.ndarray, salidas: dict[str, np.ndarray], base: str, k: int,
           carpeta: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paneles = [("Original (gris)", imagenes.a_rango_visual(img))]
    for nombre, s in salidas.items():
        lado = k if nombre == "media" else 3
        paneles.append((f"{nombre.capitalize()} {lado}x{lado}", imagenes.a_rango_visual(s)))
    if "laplaciano" in salidas:  # solo para verlo: |valor| estirado, no es la salida del programa
        lap = np.abs(salidas["laplaciano"])
        tope = np.percentile(lap, 99) or 1.0
        paneles.append(("Laplaciano |valor| realzado\n(solo para verlo)",
                        (np.clip(lap / tope, 0, 1) * 255).astype(np.uint8)))
    fig, axs = plt.subplots(1, len(paneles), figsize=(3.2 * len(paneles), 3.8),
                            constrained_layout=True)
    for ax, (titulo, datos) in zip(axs, paneles):
        ax.imshow(datos, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")
    ruta = carpeta / f"{base}_panel.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"\npanel comparativo: {_corta(ruta)}")
    return ruta


def main(args: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m convpy demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("imagen", help="ruta a cualquier imagen (jpg, png, pgm...) o URL http(s)")
    p.add_argument("--k", type=int, default=9, help="tamaño de la media (impar), por defecto 9")
    p.add_argument("--hilos", type=int, default=256, help="hilos por bloque, por defecto 256")
    p.add_argument("--secuencial", action="store_true",
                   help="también en Python secuencial y comparar con la GPU")
    p.add_argument("--sin-gpu", action="store_true", help="solo Python secuencial")
    p.add_argument("--salida", default=str(SALIDA))
    p.add_argument("--ver", action="store_true", help="abrir el panel al terminar")
    a = p.parse_args(args)

    img, base = _cargar(a.imagen)
    masks = _mascaras(a.k)
    salidas, gpu = None, None
    if not a.sin_gpu:
        try:
            salidas, gpu = _gpu(img, masks, a.hilos)
        except ImportError as motivo:
            print(f"PyCUDA no disponible ({motivo}): se sigue solo con Python secuencial")
    if a.secuencial or salidas is None:
        en_python = _secuencial(img, masks, salidas, gpu)
        if salidas is None:
            if not en_python:
                raise SystemExit("sin GPU la imagen completa es muy grande para Python: "
                                 "usar una imagen más chica")
            salidas = en_python
    panel = _guardar(img, salidas, base, a.k, Path(a.salida))
    if a.ver:
        subprocess.Popen(["xdg-open", str(panel)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
