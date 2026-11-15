# Plan de trabajo por bloques

> Nota (después de la entrega): `src/convpy/` se ordenó por secciones (`imagenes/`, `filtros/`, `convolucion/`, `evaluacion/`). Las rutas de abajo son las originales; el mapa actual está en el README.

Práctica 02 · Computación Paralela — Convolución de imágenes de alta resolución mediante PyCUDA.
Cierre: **lunes 16 de noviembre de 2026, 14:00**. Equipo: Michael Lata, Jordy Romero y Andrés Morocho.

Mismo reparto que la práctica 01. Cada bloque tiene sus archivos y una interfaz fija
(las firmas de `src/convpy/`). Mientras se respeten, cada uno trabaja en su rama sin pisar a los demás.

| Bloque | Responsable | Rama | Para cuándo | Estado |
|---|---|---|---|---|
| 1. Entorno, programa Python y convolución secuencial | Michael Lata | `bloque-1-python` | vie 13 | Terminado |
| 2. Kernel PyCUDA, volcado C/CUDA C y validación | Jordy Romero | `bloque-2-pycuda` | sáb 14 | Terminado |
| 3. Medición, experimentos, CSV, gráficas, informe y presentación | Andrés Morocho | `bloque-3-experimentos` | dom 15 | Terminado |

> [!IMPORTANT]
> **Comentario del Ing. León sobre la P01** (hay que cumplirlo en esta): *«Muy buen trabajo en la
> validación y en la comparación de tiempos antes y después del calentamiento. Registrar el estado de la
> GPU ayudó a explicar por qué cambiaban los tiempos entre corridas. Para próximas prácticas, medir
> también cuántos datos lee la GPU por segundo les ayudará a comprobar si el acceso a la memoria limita
> el rendimiento.»* → se mantiene el calentamiento y el registro del estado de la GPU, y se agrega el
> **ancho de banda efectivo** del kernel (ver bloques 2 y 3).

Contrato común: imágenes y máscaras como `np.float32`, 2D, C-contiguas (Python secuencial las pasa a listas);
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
- [x] `secuencial.py`: convolución **a mano** con listas de Python y bucles explícitos. Por indicación del docente en clase, el cálculo no usa NumPy (ni filtros de librería, JIT, multiproceso ni C); NumPy solo recibe y devuelve los datos.
- [x] `python -m convpy filtrar ...` para secuencial y pycuda.
- [x] Prueba de viabilidad: medir Python secuencial en una imagen chica y estimar 2000/4000/6000 con 3, 9 y 21;
      pasarle a Andrés qué configuraciones son viables (las otras se marcan «no completada»).
- [x] Generar `datos/imagen_*.pgm` con el script de la práctica 01 (ver `datos/README.md`).

**Resultados del bloque 1:**
- Python secuencial da 78 y 87 en la figura 1, y en `imagen_2000` media 3×3 la imagen guardada es
  idéntica píxel a píxel a la de C de la P01 (0 diferencias).
- Viabilidad (`resultados/viabilidad_python.csv`, estimado con un recorte 500×500, **no son tiempos medidos**):
  ~0,75 s/MP con 3×3, ~5,3 s/MP con 9×9 y ~26 s/MP con 21×21. Todas las configuraciones de la guía son
  viables: la más pesada es media 21×21 en 4000 (~7 min por ejecución, ~42 min con calentamiento + 5).
  Todo el Python secuencial suma ~1,3 h.
- Para medir Python secuencial (bloque 3): pasar a listas con `.tolist()` **antes** de arrancar
  `perf_counter()` y medir solo `secuencial.convolucion(lista_imagen, lista_mascara)`; esa conversión es preparación de datos.
- Precisión: Python puro acumula en `float` de Python (64 bits) y el resultado se guarda en `float32`;
  C, CUDA C y PyCUDA acumulan en `float` (32 bits). La diferencia es del orden de 1e-5, dentro de
  la tolerancia; explicarlo en el informe (pregunta 2).

**Entrega a los demás:** `imagen.py`, `mascaras.py` y `secuencial.py` funcionando (Jordy y Andrés los importan).

## Bloque 2 — PyCUDA y validación (Jordy Romero)

Archivos: `src/convpy/kernels/convolucion.cu`, `src/convpy/pycuda_conv.py`, `src/convpy/validacion.py`,
`herramientas/` y `tests/`.

- [x] `kernels/convolucion.cu`: el kernel de la práctica 01 sin cambios de lógica
      (un hilo por píxel, solo X, memoria global, control `tid < ancho*alto`, ceros fuera).
- [x] `ConvolucionPyCUDA.__init__`: contexto (`pycuda.autoinit` o `cuda.Device(0).make_context()`) y
      `SourceModule` → `get_function`; guardar `contexto_ms` y `compilacion_ms` aparte.
- [x] `ejecutar`: `cuda.mem_alloc`, `memcpy_htod`, lanzamiento con `block=(hilos,1,1)`,
      `grid=(ceil(n/hilos),1)`, `np.int32` para ancho/alto/k, `memcpy_dtoh`. Eventos `cuda.Event()`
      para CPU→GPU, kernel, GPU→CPU y total; `synchronize()` antes de `time_till()`. Documentar si la reserva entra al total.
- [x] Ancho de banda efectivo (comentario del docente): agregar a `TiemposGPU` un método
      `ancho_banda_gbs(bytes)` = bytes ÷ (kernel_ms / 1000) / 1e9, con bytes = lo que el kernel lee y escribe
      de memoria global; dejar explicado en el informe qué bytes se cuentan.
- [x] Verificar `flags['C_CONTIGUOUS']` y `dtype == float32` antes de transferir.
- [x] `herramientas/`: programa que compile con los fuentes de `practica01/src/modelo` y vuelque la salida
      **float sin recortar** (`.f32`) de C/C++ y CUDA C, para comparar las cuatro versiones.
- [x] `validacion.errores` y la justificación de `TOLERANCIA`.
- [x] Tests (ver `tests/README.md`): figura 1 (78 y 87), límites 1001×777, equivalencia de las cuatro.

**Entrega a los demás:** `ConvolucionPyCUDA` y `TiemposGPU` funcionando; tabla de errores para el informe.

## Bloque 3 — Experimentos y entrega (Andrés Morocho)

Archivos: `src/convpy/medicion.py`, `experimentos.py`, `scripts/graficas.py`, `resultados/`, `informe/`, `presentacion/`.

- [x] `medicion.py`: calentamiento + ≥5 repeticiones; `perf_counter` para Python secuencial; tiempos GPU de `TiemposGPU`.
- [x] CSV con `COLUMNAS_CSV` (versión, filtro, resolución, máscara, hilos/bloques, repetición, tiempos, unidad, estado).
- [x] Registrar el estado de la GPU en cada ventana de medición, como en la P01 (`gpu_estado.csv`:
      P-state, relojes de GPU y memoria, PCIe, temperatura, con `nvidia-smi --query-gpu=...`), y antes de
      medir liberar la GPU (`systemctl --user stop ollama.service`).
- [x] Mantener la comparación de tiempos con y sin calentamiento (lo valoró el docente).
- [x] Columna `ancho_banda_gbs` en el CSV/resumen, y una gráfica que la compare con el pico de la GPU
      (`resultados/equipo.txt`: 256 GB/s en la RTX 4060 Laptop) para responder si el kernel está limitado por memoria.
- [x] Los 4 experimentos (un factor a la vez, 256 hilos fijos fuera del de hilos).
- [x] Volver a correr `practica01` (C/C++ y CUDA C) **en el mismo equipo** que PyCUDA, o justificar la reutilización.
- [x] Configuraciones de Python secuencial no viables: registrar tiempo transcurrido y motivo, marcar «no completada»,
      y hacer la comparación de las cuatro en una resolución menor común.
> **Corrida en el equipo con GPU (Michael):** `git pull` en `bloque-3-experimentos`; luego
> `scripts/correr_todo.sh --rapida` (~10 min, prueba; resultados en `resultados/prueba_rapida/`) y
> `scripts/correr_todo.sh` (~1 h 30 min, laptop enchufada). Hace pytest, la P01 (compilada en `build/p01` y corrida
> fuera de `practica01/`), la P02, las tablas, las gráficas y la evidencia visual. Si se corta:
> `scripts/correr_todo.sh --continuar`. Al terminar: `git add resultados && git commit && git push`.
> Notas del protocolo: la fila `repeticion = 0` es la llamada sin calentar (estado `calentamiento`);
> Python secuencial se mide una vez por configuración distinta y se repite en los exp. que la comparten;
> si una configuración de Python no cabe en `--presupuesto-min` (60 por defecto) queda `no_completada`.

- [x] `graficas.py`: mediana y variabilidad; aceleración CUDA C vs C/C++ y PyCUDA vs Python (kernel y con transferencias);
      CUDA C vs PyCUDA en condiciones equivalentes. Sin aceleraciones con ejecuciones incompletas.
- [x] Evidencia visual: original + 3 filtros, recorte de la misma región.
- [x] Corrida completa en el equipo de Michael (14-nov, commit medido `a4061fb`, 88 min, todo OK,
      12 configuraciones sin `no_completada`, 33 validaciones dentro de la tolerancia).
- [x] Estado de la GPU por configuración (`estado_gpu_ventanas.csv`, `tabla_estado_gpu.tex`,
      `estado_gpu_p02_ventanas.png`): las 24 mediciones en P0, reloj máximo y PCIe gen 4.
- [x] (texto completo en `informe/informe.txt` con los datos reales; solo falta el hash de `entrega`; se arma en Overleaf) Informe PDF (contexto con la P01, migración, entorno, flujo, validación, tablas, gráficas, 6 preguntas
      de análisis, conclusiones, contribuciones, firmas, enlace al repo y hash del commit de entrega).
- [x] Diapositivas (10 min, los tres exponen).

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
| Michael Lata | 1 | Entorno, imágenes, máscaras, secuencial en Python, CLI, viabilidad y corrida de las mediciones |
| Jordy Romero | 2 | Kernel PyCUDA, eventos, ancho de banda, volcado de C/CUDA C, errores y pruebas |
| Andrés Morocho | 3 | Medición, experimentos, CSV, gráficas, evidencia visual, informe y diapositivas |
