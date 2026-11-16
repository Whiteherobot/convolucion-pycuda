"""Protocolo de medición y CSV sin GPU (se reemplaza PyCUDA por una GPU simulada). Bloque 3.

Comprueba la forma de lo que exporta `python -m convpy medir`, no tiempos reales: así se puede
correr en cualquier equipo antes de la corrida larga en el de la RTX 4060.
"""
import csv
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from convpy.entrada import imagen
from convpy.evaluacion.medicion import COLUMNAS_CSV
from convpy.convolucion.secuencial import convolucion_secuencial


@dataclass
class TiemposFalsos:
    h2d_ms: float = 1.0
    kernel_ms: float = 0.5
    d2h_ms: float = 1.0
    total_ms: float = 2.5

    def ancho_banda_gbs(self, bytes_movidos: int) -> float:
        return bytes_movidos / (self.kernel_ms / 1000.0) / 1e9


class GPUSimulada:
    contexto_ms = 1.0
    compilacion_ms = 2.0

    def ejecutar(self, img, mascara, hilos=256):
        return convolucion_secuencial(img, mascara), TiemposFalsos()


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    """Corre una vez los 4 experimentos con imágenes de 24x24 con el nombre de las reales."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _correr(tmp_path_factory.mktemp("medir"), monkeypatch)


def _correr(tmp_path, monkeypatch):
    falso = types.ModuleType("convpy.convolucion.pycuda_conv")
    falso.ConvolucionPyCUDA = GPUSimulada
    monkeypatch.setitem(sys.modules, "convpy.convolucion.pycuda_conv", falso)
    from convpy.evaluacion import experimentos
    monkeypatch.setattr(experimentos, "HERRAMIENTA", tmp_path / "no_existe")
    monkeypatch.setattr(experimentos.Referencia, "_compilar", staticmethod(lambda: False))

    datos = tmp_path / "datos"
    datos.mkdir()
    rng = np.random.default_rng(1)
    for lado in (2000, 4000, 6000):
        imagen.guardar_visual(str(datos / f"imagen_{lado}.pgm"),
                              rng.integers(0, 256, (24, 24)).astype(np.float32))
    resultados = tmp_path / "resultados"
    experimentos.main(["--datos", str(datos), "--resultados", str(resultados),
                       "--repeticiones", "2", "--calentamiento-s", "0.01"])
    return resultados


def leer(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def test_csv_tiene_las_columnas_y_todas_las_filas(corrida):
    filas = leer(corrida / "mediciones.csv")
    assert list(filas[0].keys()) == COLUMNAS_CSV
    # exp 1-3: 9 configuraciones x (python + pycuda) x (1 fría + 2); exp 4: 3 x pycuda x 3
    assert len(filas) == 9 * 2 * 3 + 3 * 3
    assert {f["estado"] for f in filas} == {"ok", "calentamiento"}
    assert not [f for f in filas if f["experimento"] == "4" and f["version"] == "python"]


def test_python_repetido_se_reutiliza_con_nota(corrida):
    filas = leer(corrida / "mediciones.csv")
    reusadas = [f for f in filas if f["version"] == "python" and f["nota"]]
    # media 3x3 y media 9x9 en 4000 aparecen de nuevo en el exp 3
    assert {(f["experimento"], f["k"]) for f in reusadas} == {("3", "3"), ("3", "9")}


def test_pycuda_tiene_bloques_y_ancho_de_banda(corrida):
    gpu = [f for f in leer(corrida / "mediciones.csv") if f["version"] == "pycuda"]
    hilos = {f["hilos"]: f["bloques"] for f in gpu if f["experimento"] == "4"}
    assert hilos == {"128": "5", "256": "3", "512": "2"}  # ceil(576 / hilos)
    assert all(float(f["ancho_banda_gbs"]) > 0 for f in gpu)


def test_validacion_y_archivos_auxiliares(corrida):
    validacion = leer(corrida / "validacion.csv")
    assert validacion and all(f["dentro_tolerancia"] == "True" for f in validacion)
    assert len(leer(corrida / "ventanas_gpu.csv")) == 12
    assert leer(corrida / "pycuda_inicio.csv")[0]["compilacion_ms"] == "2.000"


def test_python_no_completada_si_no_cabe_en_el_presupuesto(tmp_path):
    from convpy.evaluacion.medicion import Configuracion, medir_secuencial
    from convpy.entrada import mascaras
    c = Configuracion(2, "media", 3, np.ones((8, 8), np.float32), mascaras.media(3))
    r = medir_secuencial(c, presupuesto_s=0.0)
    assert [f["estado"] for f in r.filas] == ["calentamiento", "no_completada"]
    assert "presupuesto" in r.filas[1]["nota"]
    assert r.salida.shape == (8, 8)


def test_graficas_lee_lo_que_exporta_medir(corrida):
    """graficas.py corre sin errores sobre la salida de medir (sin la P01: se omiten esas partes)."""
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "graficas.py"
    r = subprocess.run([sys.executable, str(script), str(corrida)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "con error 0" in r.stdout
    assert (corrida / "resumen.csv").exists() and (corrida / "tablas" / "tabla_tiempos.tex").exists()
