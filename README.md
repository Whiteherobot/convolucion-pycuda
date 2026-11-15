# Convolución de imágenes con PyCUDA

Práctica 02 · Computación Paralela · Universidad Politécnica Salesiana

Continuación de la práctica 01 ([convolucion-cuda](https://github.com/Whiteherobot/convolucion-cuda),
commit `b2b8014`, copiada en `practica01/`). El programa anfitrión pasa a Python, se añade una
convolución secuencial en Python y el kernel CUDA de la práctica 01 se ejecuta con PyCUDA. Se comparan
cuatro versiones: C/C++ secuencial, CUDA C, Python secuencial y PyCUDA.

**Integrantes:** Michael Lata, Jordy Romero y Andrés Morocho.

> En construcción. El reparto de tareas está en [PLAN.md](PLAN.md).

## Requisitos

- GPU NVIDIA con controlador y CUDA Toolkit (`nvcc`)
- Python 3 con las dependencias de `requirements.txt`

## Cómo ejecutarlo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Imágenes de entrada (las mismas de la práctica 01)
python3 practica01/scripts/preparar_imagenes.py practica01/datos/original.jpg

# Aplicar un filtro
export PYTHONPATH=src
python3 -m convpy filtrar datos/imagen_4000.pgm resultados/media.pgm media 9 pycuda 256
python3 -m convpy filtrar datos/imagen_2000.pgm resultados/media.pgm media 3 secuencial

# Datos del equipo y prueba de viabilidad de Python secuencial
sh scripts/info_equipo.sh
python3 scripts/viabilidad.py

# Pruebas
pytest -v
```

### Mediciones (equipo con GPU NVIDIA)

```bash
scripts/correr_todo.sh --rapida          # prueba de todo el flujo, ~10 min -> resultados/prueba_rapida/
scripts/correr_todo.sh                   # corrida completa, ~1 h 30 min   -> resultados/
scripts/correr_todo.sh --continuar       # retoma una corrida cortada sin repetir lo medido
scripts/correr_todo.sh --solo-graficas   # rehace tablas y gráficas desde los CSV (no necesita GPU)
```

En el mismo equipo corre las pruebas, la práctica 01 (C/C++ y CUDA C, sin tocar `practica01/`) y la 02
(Python y PyCUDA), y genera las tablas LaTeX (`resultados/tablas/`) y las gráficas (`resultados/graficas/`).
Si un paso falla, sigue con los demás y lo indica en el resumen final. Todo queda en `registro_corrida.txt`.

## Estructura

```text
src/convpy/         programa Python: imágenes, máscaras, secuencial, PyCUDA, validación y medición
src/convpy/kernels/ kernel CUDA C adaptado de la práctica 01
herramientas/       volcado de la salida float de C/C++ y CUDA C para comparar
tests/              prueba manual, prueba de límites y equivalencia de las cuatro versiones
scripts/            datos del equipo y gráficas
practica01/         práctica 01 sin cambios (C/C++ y CUDA C)
resultados/         CSV, tablas y gráficas
informe/            informe
presentacion/       diapositivas
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
