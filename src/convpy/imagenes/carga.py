"""SECCIÓN 1 · CARGA DE IMÁGENES. Bloque 1 - Michael Lata.

Contrato con el resto del programa: toda imagen viaja como ndarray float32, 2D (alto, ancho),
C-contiguo, con valores 0-255. Así la misma matriz sirve para la versión secuencial y para PyCUDA.

Acepta cualquier formato que lea Pillow (PGM, JPG, PNG, WEBP, BMP...) y también una URL: la
descarga primero a datos/descargas/ y luego la abre como un archivo más.
"""
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[3]
DESCARGAS = RAIZ / "datos" / "descargas"


def es_url(ruta: str) -> bool:
    return ruta.startswith(("http://", "https://"))


def descargar(url: str, carpeta: Path = DESCARGAS) -> Path:
    """Baja la imagen de `url` a `carpeta` y devuelve la ruta local."""
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = Path(urllib.parse.urlparse(url).path).name or "imagen_descargada"
    destino = carpeta / nombre
    pedido = urllib.request.Request(url, headers={"User-Agent": "convpy/1.0"})
    with urllib.request.urlopen(pedido, timeout=30) as respuesta:
        destino.write_bytes(respuesta.read())
    return destino


def describir(ruta: str) -> str:
    """Texto con formato, modo de color y tamaño original (antes de pasar a grises)."""
    with Image.open(ruta) as img:
        return f"{img.format} {img.mode} {img.width}x{img.height}"


def cargar_gris(ruta: str) -> np.ndarray:
    """Lee la imagen, la pasa a escala de grises y devuelve float32 (alto, ancho) contiguo.

    convert("L") aplica la luminancia ITU-R 601: gris = 0.299 R + 0.587 G + 0.114 B.
    Una imagen que ya es gris (PGM) no cambia.
    """
    if es_url(ruta):
        ruta = str(descargar(ruta))
    with Image.open(ruta) as img:
        gris = img.convert("L")
        return np.ascontiguousarray(np.asarray(gris, dtype=np.float32))
