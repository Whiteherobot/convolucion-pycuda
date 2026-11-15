"""Imagen 1001x777, no múltiplo de 128/256/512 hilos: bordes y cobertura total. Bloque 2 - Jordi Romero."""
import numpy as np
import pytest

from conftest import imagen_aleatoria
from convpy import mascaras
from convpy.secuencial import convolucion_secuencial
from convpy.validacion import TOLERANCIA, errores

ANCHO, ALTO = 1001, 777


@pytest.fixture(scope="module")
def entrada():
    return imagen_aleatoria(ANCHO, ALTO)


@pytest.fixture(scope="module")
def esperado(entrada):
    return convolucion_secuencial(entrada, mascaras.crear("laplaciano"))


@pytest.mark.parametrize("hilos", [128, 256, 512])
def test_cobertura_total_con_cualquier_numero_de_hilos(gpu, entrada, esperado, hilos):
    salida, _ = gpu.ejecutar(entrada, mascaras.crear("laplaciano"), hilos)
    maximo, _ = errores(esperado, salida)
    assert maximo < TOLERANCIA
    # El último píxel es el que cubre el bloque parcial.
    assert salida[-1, -1] == pytest.approx(esperado[-1, -1], abs=TOLERANCIA)


def test_bordes_con_ceros_fuera_de_la_imagen(gpu):
    unos = np.ones((5, 7), dtype=np.float32)
    salida, _ = gpu.ejecutar(unos, mascaras.crear("media", 3))
    # Esquina: 4 vecinos dentro de 9; borde: 6 de 9; interior: 9 de 9.
    assert salida[0, 0] == pytest.approx(4 / 9, abs=TOLERANCIA)
    assert salida[0, 3] == pytest.approx(6 / 9, abs=TOLERANCIA)
    assert salida[2, 3] == pytest.approx(1.0, abs=TOLERANCIA)


@pytest.mark.parametrize("hilos", [0, 1025])
def test_hilos_fuera_de_rango(gpu, hilos):
    with pytest.raises(ValueError):
        gpu.ejecutar(np.ones((4, 4), dtype=np.float32), mascaras.crear("media", 3), hilos)


def test_rechaza_entrada_que_no_es_float32(gpu):
    with pytest.raises(ValueError):
        gpu.ejecutar(np.ones((4, 4), dtype=np.float64), mascaras.crear("media", 3))
