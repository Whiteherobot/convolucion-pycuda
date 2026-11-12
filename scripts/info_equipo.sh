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
} > resultados/equipo.tmp && mv resultados/equipo.tmp resultados/equipo.txt
cat resultados/equipo.txt
