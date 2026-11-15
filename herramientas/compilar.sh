#!/bin/sh
# Compila volcar_crudo con los fuentes de la práctica 01. Bloque 2 - Jordi Romero
# Uso: herramientas/compilar.sh   ->   build/volcar_crudo
set -e
raiz=$(cd "$(dirname "$0")/.." && pwd)
p1="$raiz/practica01"
mkdir -p "$raiz/build"
nvcc -O3 -std=c++17 -I"$p1/common" -I"$p1/src/modelo" \
    "$raiz/herramientas/volcar_crudo.cu" \
    "$p1/src/modelo/imagen.cpp" "$p1/src/modelo/mascara.cpp" \
    "$p1/src/modelo/convolucion_cpu.cpp" "$p1/src/modelo/convolucion_gpu.cu" \
    -o "$raiz/build/volcar_crudo"
