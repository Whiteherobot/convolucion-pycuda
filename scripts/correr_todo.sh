#!/usr/bin/env bash
# Corrida de la práctica 02 en el equipo con GPU NVIDIA. Bloque 3 - Andrés Morocho
#
# Uso, desde la raíz del repo y con la laptop enchufada:
#   scripts/correr_todo.sh --rapida        prueba de todo el flujo, unos 10 min
#                                          (resultados aparte, en resultados/prueba_rapida/)
#   scripts/correr_todo.sh                 corrida completa, alrededor de 1 h 30 min
#                                          (resultados en resultados/)
#   scripts/correr_todo.sh --continuar     retoma una corrida completa que se cortó, sin repetir
#                                          lo ya medido
#   scripts/correr_todo.sh --solo-graficas vuelve a generar tablas y gráficas desde los CSV
#
# Pasos: 1 comprobaciones, 2 imágenes, 3 datos del equipo, 4 pruebas, 5 práctica 01 (C/C++ y
# CUDA C), 6 práctica 02 (Python y PyCUDA), 7 tablas y gráficas, 8 evidencia visual.
# Si falla un paso del 3 al 8, se avisa y se sigue con los demás: lo ya medido no se pierde y el
# resumen final dice qué falló. Todo lo que sale en pantalla queda en registro_corrida.txt.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RAIZ=$PWD

MODO=completa
case "${1:-}" in
    "") ;;
    --rapida) MODO=rapida ;;
    --continuar) MODO=continuar ;;
    --solo-graficas) MODO=graficas ;;
    *) echo "Opción desconocida: $1"; sed -n '4,11p' "$0"; exit 2 ;;
esac
if [ "$MODO" = rapida ]; then R=resultados/prueba_rapida; else R=resultados; fi
mkdir -p "$R" build

# Que el equipo no se suspenda a la mitad (solo si systemd lo permite; si no, se sigue igual).
if [ "$MODO" != graficas ] && [ -z "${CORRIDA_INHIBIDA:-}" ] &&
        command -v systemd-inhibit >/dev/null 2>&1 &&
        systemd-inhibit --what=sleep:idle --why=prueba true >/dev/null 2>&1; then
    export CORRIDA_INHIBIDA=1
    exec systemd-inhibit --what=sleep:idle --why="mediciones práctica 02" \
        "$RAIZ/scripts/correr_todo.sh" "$@"
fi

# Una corrida nueva empieza con registros limpios; --continuar los conserva y agrega.
if [ "$MODO" = completa ] || [ "$MODO" = rapida ]; then
    rm -f "$R/registro_corrida.txt" "$R/medir_consola.txt" "$R"/gpu_estado_continuacion_*.csv
fi
exec > >(tee -a "$R/registro_corrida.txt") 2>&1

INICIO=$(date +%s)
NOMBRES=()
ESTADOS=()
DURACIONES=()
MONITOR=""
trap '[ -n "$MONITOR" ] && kill "$MONITOR" 2>/dev/null' EXIT

hora() { date +%H:%M:%S; }
duracion() { printf '%d min %02d s' $(($1 / 60)) $(($1 % 60)); }
titulo() { printf '\n[%s] ===== %s =====\n' "$(hora)" "$*"; }
abortar() {
    printf '\n[%s] ERROR: %s\n' "$(hora)" "$*"
    echo "La corrida se detuvo antes de medir; no hay resultados que subir."
    exit 1
}
anotar() { NOMBRES+=("$1"); ESTADOS+=("$2"); DURACIONES+=("$3"); }

# paso "<nombre>" <función>: corre la función, mide su duración y anota OK / FALLO / NO SE REPITE.
paso() {
    local nombre=$1 funcion=$2 t0 codigo
    titulo "Paso $((${#NOMBRES[@]} + 1)): $nombre"
    t0=$(date +%s)
    "$funcion"
    codigo=$?
    case $codigo in
        0) anotar "$nombre" "OK" "$(duracion $(($(date +%s) - t0)))" ;;
        99) anotar "$nombre" "NO SE REPITE" "-" ;;
        *) anotar "$nombre" "FALLO (código $codigo)" "$(duracion $(($(date +%s) - t0)))"
           printf '[%s] AVISO: el paso "%s" falló; se sigue con los demás.\n' "$(hora)" "$nombre" ;;
    esac
}

# ── Pasos ────────────────────────────────────────────────────────────────────────────────────

activar_venv() {
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
        # shellcheck disable=SC1091
        . .venv/bin/activate && echo "  entorno virtual: .venv"
    fi
}

comprobar() {
    local faltan=0 programa otros
    for programa in nvidia-smi nvcc cmake git; do
        if command -v "$programa" >/dev/null; then echo "  encontrado: $programa"
        else echo "  FALTA: $programa"; faltan=1; fi
    done
    [ $faltan -eq 0 ] || abortar "faltan programas"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader |
        sed 's/^/  GPU: /' || abortar "nvidia-smi no responde"
    activar_venv
    python3 -c "import numpy, PIL, matplotlib, pytest, pycuda.driver" 2>/dev/null ||
        abortar "faltan dependencias de Python: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    echo "  $(python3 --version 2>&1) con todas las dependencias"
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "  AVISO: hay cambios sin commit; el hash en equipo.txt no describirá exactamente el código medido"
    fi
    echo "  commit $(git rev-parse --short HEAD) en la rama $(git rev-parse --abbrev-ref HEAD)"
    # En la P01, ollama ocupaba memoria y reloj de la GPU.
    if systemctl --user stop ollama.service 2>/dev/null; then echo "  ollama detenido"; fi
    otros=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$otros" ]; then
        echo "  AVISO: otros procesos usan la GPU y pueden alterar los tiempos:"
        echo "$otros" | sed 's/^/    /'
    else
        echo "  GPU libre: ningún otro proceso la usa"
    fi
}

imagenes() {
    if [ -f datos/imagen_2000.pgm ] && [ -f datos/imagen_4000.pgm ] && [ -f datos/imagen_6000.pgm ]; then
        echo "  ya existen datos/imagen_2000.pgm, imagen_4000.pgm e imagen_6000.pgm"
    else
        python3 practica01/scripts/preparar_imagenes.py practica01/datos/original.jpg | sed 's/^/  /'
        [ -f datos/imagen_6000.pgm ] || abortar "no se pudieron generar las imágenes"
    fi
}

equipo() {
    sh scripts/info_equipo.sh > /dev/null || return 1
    {
        echo "Commit medido"
        git rev-parse HEAD
        echo "Fecha de la corrida"
        date '+%Y-%m-%d %H:%M:%S %Z'
        echo "Modo"
        echo "$MODO"
    } >> resultados/equipo.txt
    [ "$R" = resultados ] || cp resultados/equipo.txt "$R/equipo.txt"
    echo "  guardado en $R/equipo.txt:"
    sed 's/^/    /' "$R/equipo.txt"
}

pruebas() {
    echo "  pytest: figura 1, límites 1001x777, equivalencia de las 4 versiones y protocolo de medición"
    python3 -m pytest -v tests 2>&1 | tee "$R/pruebas.txt" |
        grep -E "PASSED|FAILED|ERROR|SKIPPED|passed|failed|error" | sed 's/^/  /'
    return "${PIPESTATUS[0]}"
}

practica01() {
    local corrida=build/p01_corrida codigo
    if [ "$MODO" = continuar ] && grep -q "Experimentos completados" "$R/p01_medir_consola.txt" 2>/dev/null; then
        echo "  la P01 ya terminó en la corrida anterior"
        return 99
    fi
    echo "  compilando practica01 en build/p01 (practica01/ no se modifica)"
    if ! { cmake -S practica01 -B build/p01 -DCMAKE_BUILD_TYPE=Release > /dev/null &&
           cmake --build build/p01 -j > /dev/null; }; then
        echo "  ERROR: la práctica 01 no compiló"
        return 1
    fi
    echo "  midiendo C/C++ y CUDA C: 4 experimentos, 5 repeticiones (unos 3 min)"
    # El binario escribe en ./resultados y lee ./datos: se corre en una carpeta aparte.
    rm -rf "$corrida" && mkdir -p "$corrida/resultados" && ln -s "$RAIZ/datos" "$corrida/datos"
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw,temperature.gpu \
        --format=csv --loop-ms=200 -f "$R/p01_gpu_estado.csv" < /dev/null > /dev/null 2>&1 &
    MONITOR=$!
    (cd "$corrida" && ../p01/convolucion medir) 2>&1 | tee "$R/p01_medir_consola.txt" | sed 's/^/  /'
    codigo=${PIPESTATUS[0]}
    kill "$MONITOR" 2>/dev/null
    MONITOR=""
    cp "$corrida/resultados/mediciones.csv" "$R/p01_mediciones.csv" || return 1
    echo "  guardado: $R/p01_mediciones.csv, p01_medir_consola.txt y p01_gpu_estado.csv"
    return "$codigo"
}

practica02() {
    local opciones=()
    case $MODO in
        rapida)
            opciones=(--python-en 1 --repeticiones 2)
            echo "  prueba rápida: PyCUDA en los 4 experimentos; Python solo en el exp. 1, con 2 repeticiones" ;;
        continuar)
            opciones=(--continuar)
            echo "  retomando la corrida anterior" ;;
        *)
            echo "  corrida completa: Python secuencial es lo lento (unos 75 min en total)."
            echo "  Cada configuración muestra su avance; no cerrar la terminal ni suspender el equipo." ;;
    esac
    PYTHONPATH="$RAIZ/src" python3 -u -m convpy medir --resultados "$R" \
        ${opciones[@]+"${opciones[@]}"} 2>&1 | tee -a "$R/medir_consola.txt"
    return "${PIPESTATUS[0]}"
}

graficas() { python3 scripts/graficas.py "$R"; }
evidencia() { python3 scripts/figuras_filtros.py "$R"; }

# ── Corrida ──────────────────────────────────────────────────────────────────────────────────

titulo "Práctica 02: corrida $MODO, resultados en $R/"
echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$MODO" = graficas ]; then
    activar_venv
else
    titulo "Paso 1: Comprobaciones"
    comprobar
    anotar "Comprobaciones" "OK" "-"
    titulo "Paso 2: Imágenes de entrada"
    imagenes
    anotar "Imágenes de entrada" "OK" "-"
    paso "Datos del equipo" equipo
    paso "Pruebas (pytest)" pruebas
    paso "Práctica 01: C/C++ y CUDA C" practica01
    paso "Práctica 02: Python y PyCUDA" practica02
fi
paso "Tablas y gráficas" graficas
paso "Evidencia visual" evidencia

titulo "Resumen"
printf '  %-20s %-12s %s\n' "ESTADO" "DURACIÓN" "PASO"
fallos=0
for i in "${!NOMBRES[@]}"; do
    printf '  %-20s %-12s %s\n' "${ESTADOS[$i]}" "${DURACIONES[$i]}" "${NOMBRES[$i]}"
    case ${ESTADOS[$i]} in FALLO*) fallos=1 ;; esac
done
echo "  Tiempo total: $(duracion $(($(date +%s) - INICIO)))"
echo "  Resultados en $R/ (registro completo en $R/registro_corrida.txt)"
echo
if [ $fallos -eq 0 ]; then
    echo "Todo terminó bien."
else
    echo "Hubo pasos con FALLO: revisar sus mensajes arriba. Lo que sí se midió quedó guardado;"
    echo "conviene subirlo igual para revisarlo. Una medición cortada se retoma con:"
    echo "  scripts/correr_todo.sh --continuar"
fi
if [ "$MODO" != graficas ]; then
    echo
    echo "Para que Andrés reciba los resultados:"
    echo "  git add resultados"
    echo "  git commit -m \"Resultados: corrida $MODO\""
    echo "  git push"
fi
exit $fallos
