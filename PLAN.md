# Plan de trabajo por bloques

Práctica 02 · Computación Paralela — Convolución de imágenes de alta resolución mediante PyCUDA.
Cierre: **lunes 16 de noviembre de 2026, 14:00**. Equipo: Michael Lata, Jordi Romero y Andrés Morocho.

Mismo reparto que la práctica 01. Cada bloque tiene sus archivos y una interfaz fija
(las firmas de `src/convpy/`). Mientras se respeten, cada uno trabaja en su rama sin pisar a los demás.

| Bloque | Responsable | Rama | Para cuándo | Estado |
|---|---|---|---|---|
| 1. Entorno, programa Python y convolución secuencial | Michael Lata | `bloque-1-python` | vie 13 | Terminado |
| 2. Kernel PyCUDA, volcado C/CUDA C y validación | Jordi Romero | `bloque-2-pycuda` | sáb 14 | Pendiente |
| 3. Medición, experimentos, CSV, gráficas, informe y presentación | Andrés Morocho | `bloque-3-experimentos` | dom 15 | Pendiente |

> [!IMPORTANT]
> **Comentario del Ing. León sobre la P01** (hay que cumplirlo en esta): *«Muy buen trabajo en la
> validación y en la comparación de tiempos antes y después del calentamiento. Registrar el estado de la
> GPU ayudó a explicar por qué cambiaban los tiempos entre corridas. Para próximas prácticas, medir
> también cuántos datos lee la GPU por segundo les ayudará a comprobar si el acceso a la memoria limita
> el rendimiento.»* → se mantiene el calentamiento y el registro del estado de la GPU, y se agrega el
> **ancho de banda efectivo** del kernel (ver bloques 2 y 3).

Contrato común: imágenes y máscaras como `np.float32`, 2D, C-contiguas; acumulación en float32;
ceros fuera de la imagen; correlación (las máscaras son simétricas); recorte a 0–255 **solo al guardar**.

La práctica 01 está intacta en `practica01/` (commit `b2b8014` de
[convolucion-cuda](https://github.com/Whiteherobot/convolucion-cuda/commit/b2b801459a0b30f796832213c633fa87a60e77b1)).
No se modifica: lo nuevo de C/CUDA C va en `herramientas/`.

---

## Bloque 1 — Entorno y Python (Michael Lata)

Archivos: `src/convpy/imagen.py`, `mascaras.py`, `secuencial.py`, `__main__.py`, `scripts/info_equipo.sh`, `scripts/viabilidad.py`, `requirements.txt`.

- [x] Entorno: venv, `pip install -r requirements.txt` y fijar versiones exactas (`pip freeze`).
- [x] `scripts/info_equipo.sh` (reusa el de la P01 y agrega SO, núcleos, versiones de Python y el ancho de banda pico de la memoria de la GPU) → `resultados/equipo.txt`: CPU, núcleos, RAM, GPU, memoria de GPU,
      SO, Python, NumPy, PyCUDA, controlador y CUDA Toolkit.
- [x] `imagen.py`: cargar en grises a float32 contiguo, guardar visual (redondeo + recorte), guardar/leer crudo (`.npy` y `.f32`).
- [x] `mascaras.py`: media k×k, nitidez y laplaciano con los coeficientes de la práctica 01.
- [x] `secuencial.py`: convolución con bucles explícitos (sin filtros de librería, JIT, multiproceso ni C).
- [x] `python -m convpy filtrar ...` para secuencial y pycuda.
- [x] Prueba de viabilidad: medir Python secuencial en una imagen chica y estimar 2000/4000/6000 con 3, 9 y 21;
      pasarle a Andrés qué configuraciones son viables (las otras se marcan «no completada»).
- [x] Generar `datos/imagen_*.pgm` con el script de la práctica 01 (ver `datos/README.md`).

**Resultados del bloque 1:**
- Python secuencial da 78 y 87 en la figura 1, y en `imagen_2000` media 3×3 la imagen guardada es
  idéntica píxel a píxel a la de C de la P01 (0 diferencias).
- Viabilidad (`resultados/viabilidad_python.csv`, estimado con un recorte 500×500, **no son tiempos medidos**):
  ~2,3 s/MP con 3×3, ~17 s/MP con 9×9 y ~83 s/MP con 21×21. Todas las configuraciones de la guía son
  viables dejándolas correr: la más pesada es media 21×21 en 4000 (~22 min por ejecución, ~2,2 h con
  calentamiento + 5). Todo el Python secuencial suma ~4 h → correrlo de noche en un solo bloque.
- Python secuencial acumula en `np.float32` (como el `float` de C), no en el float64 de Python.

**Entrega a los demás:** `imagen.py`, `mascaras.py` y `secuencial.py` funcionando (Jordi y Andrés los importan).

## Bloque 2 — PyCUDA y validación (Jordi Romero)

Archivos: `src/convpy/kernels/convolucion.cu`, `src/convpy/pycuda_conv.py`, `src/convpy/validacion.py`,
`herramientas/` y `tests/`.

- [ ] `kernels/convolucion.cu`: el kernel de la práctica 01 sin cambios de lógica
      (un hilo por píxel, solo X, memoria global, control `tid < ancho*alto`, ceros fuera).
- [ ] `ConvolucionPyCUDA.__init__`: contexto (`pycuda.autoinit` o `cuda.Device(0).make_context()`) y
      `SourceModule` → `get_function`; guardar `contexto_ms` y `compilacion_ms` aparte.
- [ ] `ejecutar`: `cuda.mem_alloc`, `memcpy_htod`, lanzamiento con `block=(hilos,1,1)`,
      `grid=(ceil(n/hilos),1)`, `np.int32` para ancho/alto/k, `memcpy_dtoh`. Eventos `cuda.Event()`
      para CPU→GPU, kernel, GPU→CPU y total; `synchronize()` antes de `time_till()`. Documentar si la reserva entra al total.
- [ ] Ancho de banda efectivo (comentario del docente): agregar a `TiemposGPU` un método
      `ancho_banda_gbs(bytes)` = bytes ÷ (kernel_ms / 1000) / 1e9, con bytes = lo que el kernel lee y escribe
      de memoria global; dejar explicado en el informe qué bytes se cuentan.
- [ ] Verificar `flags['C_CONTIGUOUS']` y `dtype == float32` antes de transferir.
- [ ] `herramientas/`: programa que compile con los fuentes de `practica01/src/modelo` y vuelque la salida
      **float sin recortar** (`.f32`) de C/C++ y CUDA C, para comparar las cuatro versiones.
- [ ] `validacion.errores` y la justificación de `TOLERANCIA`.
- [ ] Tests (ver `tests/README.md`): figura 1 (78 y 87), límites 1001×777, equivalencia de las cuatro.

**Entrega a los demás:** `ConvolucionPyCUDA` y `TiemposGPU` funcionando; tabla de errores para el informe.

## Bloque 3 — Experimentos y entrega (Andrés Morocho)

Archivos: `src/convpy/medicion.py`, `experimentos.py`, `scripts/graficas.py`, `resultados/`, `informe/`, `presentacion/`.

- [ ] `medicion.py`: calentamiento + ≥5 repeticiones; `perf_counter` para Python secuencial; tiempos GPU de `TiemposGPU`.
- [ ] CSV con `COLUMNAS_CSV` (versión, filtro, resolución, máscara, hilos/bloques, repetición, tiempos, unidad, estado).
- [ ] Registrar el estado de la GPU en cada ventana de medición, como en la P01 (`gpu_estado.csv`:
      P-state, relojes de GPU y memoria, PCIe, temperatura, con `nvidia-smi --query-gpu=...`), y antes de
      medir liberar la GPU (`systemctl --user stop ollama.service`).
- [ ] Mantener la comparación de tiempos con y sin calentamiento (lo valoró el docente).
- [ ] Columna `ancho_banda_gbs` en el CSV/resumen, y una gráfica que la compare con el pico de la GPU
      (`resultados/equipo.txt`: 256 GB/s en la RTX 4060 Laptop) para responder si el kernel está limitado por memoria.
- [ ] Los 4 experimentos (un factor a la vez, 256 hilos fijos fuera del de hilos).
- [ ] Volver a correr `practica01` (C/C++ y CUDA C) **en el mismo equipo** que PyCUDA, o justificar la reutilización.
- [ ] Configuraciones de Python secuencial no viables: registrar tiempo transcurrido y motivo, marcar «no completada»,
      y hacer la comparación de las cuatro en una resolución menor común.
- [ ] `graficas.py`: mediana y variabilidad; aceleración CUDA C vs C/C++ y PyCUDA vs Python (kernel y con transferencias);
      CUDA C vs PyCUDA en condiciones equivalentes. Sin aceleraciones con ejecuciones incompletas.
- [ ] Evidencia visual: original + 3 filtros, recorte de la misma región.
- [ ] Informe PDF (contexto con la P01, migración, entorno, flujo, validación, tablas, gráficas, 6 preguntas
      de análisis, conclusiones, contribuciones, firmas, enlace al repo y hash del commit de entrega).
- [ ] Diapositivas (10 min, los tres exponen).

---

## Integración

1. Cada uno trabaja en su rama y abre un Pull Request hacia `main`.
2. Antes del PR: `pytest` pasa y `python -m convpy filtrar ...` corre.
3. No cambiar las firmas de `src/convpy/` sin avisar al grupo.
4. El commit de entrega se etiqueta `entrega` y su hash va en el informe.
5. Las mediciones finales se corren en un solo equipo (RTX 4060 Laptop, el de la P01).

## Contribuciones (para el informe)

| Integrante | Bloque | Aporte |
|---|---|---|
| Michael Lata | 1 | _(completar)_ |
| Jordi Romero | 2 | _(completar)_ |
| Andrés Morocho | 3 | _(completar)_ |
