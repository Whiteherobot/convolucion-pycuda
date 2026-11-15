#!/usr/bin/env python3
"""Evidencia visual de los filtros para el informe. Bloque 3 - Andrés Morocho.

Uso: python3 scripts/figuras_filtros.py [carpeta]      (por defecto resultados/)

Toma datos/imagen_4000.pgm (la resolución de los exp. 1 y 3) y le aplica media 9x9, nitidez y
laplaciano con PyCUDA, la versión que se estudia en esta práctica. Si PyCUDA no está (un equipo
sin NVIDIA), usa una convolución NumPy equivalente y lo dice en la figura: es solo para ver el
efecto, no es ninguna de las versiones medidas.

Salidas en <carpeta>/:
  graficas/evidencia_filtros.png  fila 1: imagen completa; fila 2: recorte 400x400 de la misma
                                  región a resolución completa (salida recortada a 0-255 como
                                  la guarda el programa)
  rangos_filtros.csv              mínimo, máximo y % de píxeles fuera de 0-255 ANTES del recorte
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from convpy import filtros, imagenes  # noqa: E402

ENTRADA = RAIZ / "datos" / "imagen_4000.pgm"
FILTROS = [("media", 9), ("nitidez", 3), ("laplaciano", 3)]
LADO_RECORTE = 400
# Esquina superior izquierda del recorte, en fracción del lado: borde del sombrero y ramas, donde
# se nota el efecto de cada filtro (el centro de la foto es una zona lisa).
RECORTE_REL = (0.34, 0.375)  # (fila, columna)


def convolucion_numpy(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Correlación con ceros fuera, sumando una copia desplazada por coeficiente."""
    k = m.shape[0]
    r = k // 2
    ext = np.pad(img, r)
    salida = np.zeros_like(img)
    for i in range(k):
        for j in range(k):
            if m[i, j] != 0:
                salida += m[i, j] * ext[i:i + img.shape[0], j:j + img.shape[1]]
    return salida


def filtrar(img: np.ndarray):
    try:
        from convpy.convolucion.pycuda_gpu import ConvolucionPyCUDA
        gpu = ConvolucionPyCUDA()
        return "PyCUDA", {f: gpu.ejecutar(img, filtros.crear(*f))[0] for f in FILTROS}
    except Exception as motivo:  # sin GPU o sin PyCUDA
        print(f"  PyCUDA no disponible ({type(motivo).__name__}): se usa NumPy solo para la figura")
        return "NumPy (sin GPU)", {f: convolucion_numpy(img, filtros.crear(*f)) for f in FILTROS}


def main() -> int:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else RAIZ / "resultados")
    (base / "graficas").mkdir(parents=True, exist_ok=True)
    if not ENTRADA.exists():
        print(f"Falta {ENTRADA}: generar con practica01/scripts/preparar_imagenes.py")
        return 1
    img = imagenes.cargar_gris(str(ENTRADA))
    print(f"Evidencia visual con {ENTRADA.name} ({img.shape[1]}x{img.shape[0]})")
    origen, salidas = filtrar(img)

    alto, ancho = img.shape
    y0 = min(int(alto * RECORTE_REL[0]), alto - LADO_RECORTE)
    x0 = min(int(ancho * RECORTE_REL[1]), ancho - LADO_RECORTE)
    paneles = [("Original", img)] + [
        (f"{f.capitalize()} {k}×{k}", s) for (f, k), s in salidas.items()]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.8), constrained_layout=True)
    for col, (titulo, datos) in enumerate(paneles):
        visible = imagenes.a_rango_visual(datos)
        axs[0, col].imshow(visible, cmap="gray", vmin=0, vmax=255)
        axs[0, col].add_patch(plt.Rectangle((x0, y0), LADO_RECORTE, LADO_RECORTE, fill=False,
                                            edgecolor="yellow", linewidth=1))
        axs[1, col].imshow(visible[y0:y0 + LADO_RECORTE, x0:x0 + LADO_RECORTE], cmap="gray",
                           vmin=0, vmax=255, interpolation="nearest")
        axs[0, col].set_title(titulo)
        axs[1, col].set_title(f"recorte {LADO_RECORTE}×{LADO_RECORTE} (recuadro amarillo)",
                              fontsize=8)
        for ax in axs[:, col]:
            ax.axis("off")
    fig.suptitle(f"{ENTRADA.name}: filtros aplicados con {origen}; salida recortada a 0–255 "
                 "como la guarda el programa")
    ruta = base / "graficas" / "evidencia_filtros.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok        {ruta.relative_to(base)}")

    ruta = base / "rangos_filtros.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "filtro", "k", "calculado_con", "minimo", "maximo",
                    "pct_bajo_0", "pct_sobre_255"])
        for (filtro, k), s in salidas.items():
            w.writerow([ENTRADA.name, filtro, k, origen, f"{s.min():.2f}", f"{s.max():.2f}",
                        f"{100 * (s < 0).mean():.2f}", f"{100 * (s > 255).mean():.2f}"])
    print(f"  ok        {ruta.relative_to(base)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
