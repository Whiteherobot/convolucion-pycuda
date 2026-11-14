"""Las cuatro versiones contra C/C++ secuencial, sin recortar. Bloque 2 - Jordi Romero.

El volcado de C y CUDA C sale de herramientas/volcar_crudo (se compila con nvcc).
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import imagen_aleatoria
from convpy import imagen, mascaras
from convpy.secuencial import convolucion_secuencial
from convpy.validacion import TOLERANCIA, errores

RAIZ = Path(__file__).resolve().parents[1]
HERRAMIENTA = RAIZ / "build" / "volcar_crudo"
ANCHO, ALTO = 400, 300
CASOS = [("media", 3), ("media", 9), ("nitidez", 3), ("laplaciano", 3)]


@pytest.fixture(scope="session")
def herramienta():
    if not HERRAMIENTA.exists():
        try:
            subprocess.run([str(RAIZ / "herramientas" / "compilar.sh")], check=True)
        except (OSError, subprocess.CalledProcessError) as fallo:
            pytest.skip(f"no se pudo compilar volcar_crudo: {fallo}")
    return HERRAMIENTA


@pytest.fixture(scope="module")
def entrada_pgm(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("equivalencia") / "entrada.pgm"
    imagen.guardar_visual(str(ruta), imagen_aleatoria(ANCHO, ALTO))
    return ruta


@pytest.mark.parametrize("filtro,k", CASOS)
def test_cuatro_versiones_coinciden(gpu, herramienta, entrada_pgm, filtro, k):
    prefijo = entrada_pgm.parent / f"{filtro}{k}"
    subprocess.run([str(herramienta), str(entrada_pgm), filtro, str(k), str(prefijo)], check=True)
    c_cpu = imagen.cargar_crudo(f"{prefijo}_cpu.f32", ANCHO, ALTO)
    c_gpu = imagen.cargar_crudo(f"{prefijo}_gpu.f32", ANCHO, ALTO)

    entrada = imagen.cargar_gris(str(entrada_pgm))
    mascara = mascaras.crear(filtro, k)
    python_cpu = convolucion_secuencial(entrada, mascara)
    python_gpu, _ = gpu.ejecutar(entrada, mascara)

    for nombre, otra in (("CUDA C", c_gpu), ("Python", python_cpu), ("PyCUDA", python_gpu)):
        maximo, medio = errores(c_cpu, otra)
        print(f"{nombre:7s} {filtro} {k}x{k}: error máximo {maximo:.2e}, medio {medio:.2e}")
        assert maximo < TOLERANCIA, nombre
