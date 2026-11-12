"""Medición de tiempos y CSV. Bloque 3 - Andrés Morocho.

CPU: time.perf_counter() solo alrededor de la convolución secuencial.
GPU: TiemposGPU que devuelve ConvolucionPyCUDA.ejecutar (eventos CUDA, sincronizado).
Calentamiento previo + al menos 5 repeticiones completas por configuración viable.
"""

COLUMNAS_CSV = [
    "experimento", "version", "filtro", "resolucion", "k", "hilos", "bloques",
    "repeticion", "cpu_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms",
    "unidad", "estado",  # estado: ok | no_completada (con motivo en el registro)
]
