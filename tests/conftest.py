import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from convpy import imagen, mascaras

FIGURA1_VALORES = np.array([[35, 40, 41, 45, 50],
                            [40, 40, 42, 46, 52],
                            [42, 46, 50, 55, 55],
                            [48, 52, 56, 58, 60],
                            [56, 60, 65, 70, 75]], dtype=np.float32)
FIGURA1_MASCARA = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
SEMILLA = 2026


def imagen_aleatoria(ancho: int, alto: int) -> np.ndarray:
    """Valores enteros 0..255; la semilla fija hace la prueba repetible."""
    generador = np.random.default_rng(SEMILLA)
    return generador.integers(0, 256, size=(alto, ancho)).astype(np.float32)


@pytest.fixture(scope="session")
def gpu():
    from convpy.pycuda_conv import ConvolucionPyCUDA
    return ConvolucionPyCUDA()


@pytest.fixture(scope="session")
def mascaras_prueba():
    return [mascaras.crear("media", 3), mascaras.crear("media", 9),
            mascaras.crear("nitidez"), mascaras.crear("laplaciano")]
