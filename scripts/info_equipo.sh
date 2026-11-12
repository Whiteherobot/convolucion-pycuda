#!/bin/sh
# Datos del equipo para la práctica 02: los de la práctica 01 más las versiones de Python.
# Bloque 1 - Michael Lata. Uso: sh scripts/info_equipo.sh (desde la raíz, con el venv activo)
sh practica01/scripts/info_equipo.sh > /dev/null
{
    cat resultados/equipo.txt
    echo "Sistema operativo"
    grep PRETTY_NAME /etc/os-release
    echo "Núcleos físicos"
    lscpu -p=Core | grep -v '^#' | sort -u | wc -l
    echo "Python"
    python3 -c "import sys, numpy, pycuda; print(sys.version.split()[0], '| NumPy', numpy.__version__, '| PyCUDA', pycuda.VERSION_TEXT)"
    echo "Ancho de banda pico de la memoria (2 x reloj x bus / 8)"
    python3 -c "
import pycuda.driver as cuda
cuda.init(); d = cuda.Device(0); A = cuda.device_attribute
reloj = d.get_attribute(A.MEMORY_CLOCK_RATE); bus = d.get_attribute(A.GLOBAL_MEMORY_BUS_WIDTH)
print(f'{reloj / 1e3:.0f} MHz, bus {bus} bits -> {2 * reloj * 1e3 * bus / 8 / 1e9:.1f} GB/s')"
} > resultados/equipo.tmp && mv resultados/equipo.tmp resultados/equipo.txt
cat resultados/equipo.txt
