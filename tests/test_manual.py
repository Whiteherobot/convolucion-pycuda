"""Matriz 5x5 de la figura 1 con valores calculados a mano. Bloque 2 - Jordi Romero."""
import pytest

from convpy.secuencial import convolucion_secuencial
from conftest import FIGURA1_MASCARA, FIGURA1_VALORES
from convpy.validacion import TOLERANCIA


def comprobar(salida):
    # 35(-2)+40(-1)+41(0)+40(-1)+40(1)+42(1)+42(0)+46(1)+50(2) = 78
    assert salida[1, 1] == pytest.approx(78, abs=TOLERANCIA)
    assert salida[1, 2] == pytest.approx(87, abs=TOLERANCIA)
    # esquina: solo 4 vecinos dentro, 35(1)+40(1)+40(1)+40(2) = 195
    assert salida[0, 0] == pytest.approx(195, abs=TOLERANCIA)


def test_secuencial_figura1():
    comprobar(convolucion_secuencial(FIGURA1_VALORES, FIGURA1_MASCARA))


def test_pycuda_figura1(gpu):
    salida, _ = gpu.ejecutar(FIGURA1_VALORES, FIGURA1_MASCARA)
    comprobar(salida)
