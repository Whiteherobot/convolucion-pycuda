"""Prueba de viabilidad de Python secuencial. Bloque 1 - Michael Lata.

Mide una vez la convolución en un recorte de 500x500 de imagen_4000 y escala por número de
píxeles para estimar cada configuración. Son ESTIMACIONES para planificar los experimentos,
no tiempos medidos: no van al informe como resultados.
Uso: PYTHONPATH=src python scripts/viabilidad.py
"""
import time

from convpy import imagen, mascaras
from convpy.secuencial import convolucion_secuencial

LADO = 500
REPETICIONES = 6  # calentamiento + 5 medidas

recorte = imagen.cargar_gris("datos/imagen_4000.pgm")[:LADO, :LADO].copy()
lineas = ["k,s_por_megapixel,resolucion,estimado_una_s,estimado_6_ejecuciones_min"]
for k in (3, 9, 21):
    inicio = time.perf_counter()
    convolucion_secuencial(recorte, mascaras.media(k))
    s_mp = (time.perf_counter() - inicio) / (LADO * LADO / 1e6)
    for lado in (2000, 4000, 6000):
        una = s_mp * lado * lado / 1e6
        lineas.append(f"{k},{s_mp:.2f},{lado},{una:.0f},{una * REPETICIONES / 60:.1f}")

with open("resultados/viabilidad_python.csv", "w") as f:
    f.write("\n".join(lineas) + "\n")
print("\n".join(lineas))
