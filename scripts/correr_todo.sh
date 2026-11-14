#!/usr/bin/env bash
# Corrida completa de la práctica 02 en el equipo con GPU NVIDIA. Bloque 3 - Andrés Morocho
#
# Uso, desde la raíz del repo, con la laptop enchufada:
#   scripts/correr_todo.sh                 # todo (~1,5 h: Python secuencial es lo lento)
#   scripts/correr_todo.sh --sin-python    # prueba rápida (~3 min) de que la parte GPU funciona
#
# Pasos (si uno falla, se detiene):
#   1. Comprueba GPU, nvcc, cmake y PyCUDA; libera la GPU (ollama).
#   2. Genera datos/imagen_{2000,4000,6000}.pgm si faltan.
#   3. Datos del equipo y commit medido      -> resultados/equipo.txt
#   4. pytest                                -> resultados/pruebas.txt
#   5. Práctica 01 (C/C++ y CUDA C), mismo equipo y protocolo, sin tocar practica01/
#                                            -> resultados/p01_{mediciones,gpu_estado,medir_consola}.*
#   6. Práctica 02 (Python y PyCUDA)         -> resultados/{mediciones,validacion,ventanas_gpu,
#                                               gpu_estado,pycuda_inicio}.csv y medir_consola.txt
# Al final: git add resultados && git commit && git push.
set -euo pipefail
cd "$(dirname "$0")/.."
R=resultados
mkdir -p "$R" build

# La corrida es larga: que el equipo no se suspenda a la mitad.
if [ -z "${CORRIDA_INHIBIDA:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
    export CORRIDA_INHIBIDA=1
    exec systemd-inhibit --what=sleep:idle --why="mediciones práctica 02" "$PWD/scripts/correr_todo.sh" "$@"
fi

paso() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
falta() { echo "[ERROR] $*" >&2; exit 1; }

paso "1. Comprobaciones"
for programa in nvidia-smi nvcc cmake; do
    command -v "$programa" >/dev/null || falta "no se encontró $programa"
done
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
python3 -c "import numpy, PIL, matplotlib, pytest, pycuda" ||
    falta "faltan dependencias: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
export PYTHONPATH="$PWD/src"
# Liberar la GPU de otros procesos (en la P01 ollama ocupaba memoria y reloj).
systemctl --user stop ollama.service 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
    sed 's/^/  proceso en la GPU: /' || true

paso "2. Imágenes de entrada"
if [ ! -f datos/imagen_6000.pgm ]; then
    python3 practica01/scripts/preparar_imagenes.py practica01/datos/original.jpg
fi
ls -l datos/imagen_*.pgm

paso "3. Datos del equipo"
sh scripts/info_equipo.sh
{
    echo "Commit medido"
    git rev-parse HEAD
    git status --porcelain --untracked-files=no | sed 's/^/  modificado: /'
    echo "Fecha de la corrida"
    date '+%Y-%m-%d %H:%M:%S %Z'
} >> "$R/equipo.txt"

paso "4. Pruebas"
python3 -m pytest -v tests 2>&1 | tee "$R/pruebas.txt"

paso "5. Práctica 01: C/C++ y CUDA C"
cmake -S practica01 -B build/p01 -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build/p01 -j
# El binario escribe en ./resultados y lee ./datos: se corre en una carpeta aparte para no
# modificar practica01/resultados.
corrida=build/p01_corrida
rm -rf "$corrida" && mkdir -p "$corrida/resultados"
ln -s "$PWD/datos" "$corrida/datos"
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw,temperature.gpu \
    --format=csv --loop-ms=200 -f "$R/p01_gpu_estado.csv" &
monitor=$!
(cd "$corrida" && ../p01/convolucion medir) 2>&1 | tee "$R/p01_medir_consola.txt"
kill "$monitor" 2>/dev/null || true
cp "$corrida/resultados/mediciones.csv" "$R/p01_mediciones.csv"

paso "6. Práctica 02: Python secuencial y PyCUDA"
python3 -u -m convpy medir "$@" 2>&1 | tee "$R/medir_consola.txt"

paso "Listo"
ls -l "$R"
cat <<'FIN'

Sube los resultados para que Andrés siga con las gráficas y el informe:
  git add resultados
  git commit -m "Resultados de la corrida completa"
  git push
FIN
