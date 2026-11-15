#!/usr/bin/env python3
"""Tablas y gráficas del informe a partir de las mediciones. Bloque 3 - Andrés Morocho.

Uso: python3 scripts/graficas.py [carpeta]        (por defecto resultados/)

Entradas en <carpeta>/ (scripts/correr_todo.sh las deja ahí):
  mediciones.csv          P02: Python secuencial y PyCUDA (python -m convpy medir)
  p01_mediciones.csv      P01 corrida de nuevo en el mismo equipo: C/C++ y CUDA C
  p01_medir_consola.txt   consola de la P01: de ahí sale el kernel frío de CUDA C
  validacion.csv, pycuda_inicio.csv, ventanas_gpu.csv, gpu_estado*.csv, p01_gpu_estado.csv,
  equipo.txt (ancho de banda pico de la GPU)
Si falta una entrada, solo se omiten las salidas que la necesitan, con el motivo.

Salidas:
  resumen.csv             una fila por (experimento, versión, configuración): mediana, desviación,
                          mínimo y máximo de cada tiempo y del ancho de banda, y la llamada fría
  aceleraciones.csv       CUDA C vs C/C++, PyCUDA vs Python y Python vs C/C++, por configuración
  cudac_vs_pycuda.csv     las dos versiones GPU lado a lado (mismo kernel, mismo equipo)
  estado_gpu_ventanas.csv estado de la GPU (nvidia-smi) en cada configuración medida, P01 y P02
  tablas/*.tex            tablas para el informe (\\input{...})
  graficas/*.png          gráficas para el informe y las diapositivas

Definiciones (las mismas de la P01):
  solo cuentan las repeticiones con estado «ok»; la 0 es la llamada sin calentar
  tiempo de una versión secuencial = cpu_ms; de una versión GPU = total_ms (con transferencias)
  aceleración de cómputo = mediana secuencial / mediana kernel
  aceleración con transferencias = mediana secuencial / mediana total GPU
  ancho de banda efectivo = 2 x píxeles x 4 B / tiempo de kernel (leer imagen y escribir salida)
  no se calcula ninguna aceleración con una configuración «no_completada»
"""
import csv
import re
import statistics
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VERSIONES = ["c", "python", "cuda_c", "pycuda"]
NOMBRE = {"c": "C/C++", "python": "Python", "cuda_c": "CUDA C", "pycuda": "PyCUDA"}
COLOR = {"c": "#4C72B0", "python": "#55A868", "cuda_c": "#C44E52", "pycuda": "#DD8452"}
COLOR_PARTE = {"h2d": "#55A868", "kernel": "#C44E52", "d2h": "#8172B2"}
GPU = ("cuda_c", "pycuda")
METRICAS = ["cpu", "h2d", "kernel", "d2h", "total", "banda"]
PICO_REFERENCIA_GBS = 256.0  # RTX 4060 Laptop, por si equipo.txt no trae el cálculo

EXPERIMENTOS = {
    1: ("Exp. 1: tipo de filtro (4000×4000, 3×3, 256 hilos)", "filtro"),
    2: ("Exp. 2: resolución (media 9×9, 256 hilos)", "lado"),
    3: ("Exp. 3: tamaño de máscara (media, 4000×4000, 256 hilos)", "k"),
    4: ("Exp. 4: hilos por bloque (media 9×9, 4000×4000)", "hilos"),
}
ORDEN_FILTRO = {"media": 0, "nitidez": 1, "laplaciano": 2}

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.3})


class FaltanDatos(Exception):
    """Salida que no se puede generar con lo que hay (se omite con aviso, no es un error)."""


# ── Carga ────────────────────────────────────────────────────────────────────────────────────

def _f(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return 0.0


def _banda(lado: int, kernel_ms: float) -> float:
    return 2 * lado * lado * 4 / (kernel_ms / 1000) / 1e9 if kernel_ms > 0 else 0.0


def leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_p02(ruta: Path) -> list[dict]:
    filas = []
    for r in leer_csv(ruta):
        filas.append({
            "exp": int(r["experimento"]), "version": r["version"], "filtro": r["filtro"],
            "lado": int(r["resolucion"].split("x")[0]), "k": int(r["k"]),
            "hilos": int(r["hilos"]), "rep": int(r["repeticion"]), "estado": r["estado"],
            "nota": r.get("nota", ""), "cpu": _f(r["cpu_ms"]), "h2d": _f(r["h2d_ms"]),
            "kernel": _f(r["kernel_ms"]), "d2h": _f(r["d2h_ms"]), "total": _f(r["total_ms"]),
            "banda": _f(r["ancho_banda_gbs"])})
    return filas


def cargar_p01(ruta: Path) -> list[dict]:
    """Formato de practica01/src/vista/csv.cpp: implementacion cpu|gpu, mascara «9x9»."""
    filas = []
    for r in leer_csv(ruta):
        lado, kernel = int(r["resolucion"].split("x")[0]), _f(r["kernel_ms"])
        gpu = r["implementacion"] == "gpu"
        filas.append({
            "exp": int(r["experimento"]), "version": "cuda_c" if gpu else "c",
            "filtro": r["filtro"], "lado": lado, "k": int(r["mascara"].split("x")[0]),
            "hilos": int(r["hilos_por_bloque"]), "rep": int(r["repeticion"]), "estado": "ok",
            "nota": "", "cpu": _f(r["cpu_ms"]), "h2d": _f(r["h2d_ms"]), "kernel": kernel,
            "d2h": _f(r["d2h_ms"]), "total": _f(r["total_ms"]),
            "banda": _banda(lado, kernel) if gpu else 0.0})
    return filas


def frio_p01(ruta: Path) -> dict:
    """Kernel de la primera y de la última llamada del calentamiento de CUDA C, leídos de la
    consola de la P01 (su CSV no guarda la llamada fría). Clave (exp, filtro, lado, k, hilos)."""
    frio, pendiente, exp = {}, None, 0
    calent = re.compile(r"calentamiento GPU: \d+ llamadas .* kernel ([\d.]+) -> ([\d.]+) ms")
    config = re.compile(r"^\s+(\w+) (\d+)x\d+ (\d+)x\d+, (\d+) hilos")
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := re.match(r"\[Exp (\d)\]", linea.strip()):
            exp = int(m.group(1))
        elif m := calent.search(linea):
            pendiente = (float(m.group(1)), float(m.group(2)))
        elif (m := config.match(linea)) and pendiente:
            filtro, k, lado, hilos = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            frio[(exp, filtro, lado, k, hilos)] = pendiente
            pendiente = None
    return frio


def pico_gbs(ruta: Path) -> tuple[float, str]:
    if ruta.exists():
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        if m := re.search(r"Ancho de banda pico.*?\n.*?([\d.]+) GB/s", texto):
            return float(m.group(1)), "calculado en equipo.txt"
    return PICO_REFERENCIA_GBS, "valor de referencia de la RTX 4060 Laptop"


# ── Resumen ──────────────────────────────────────────────────────────────────────────────────

def _est(valores: list[float]) -> dict:
    return {"mediana": statistics.median(valores), "min": min(valores), "max": max(valores),
            "desv": statistics.stdev(valores) if len(valores) > 1 else 0.0}


def resumir(filas: list[dict], frio_cudac: dict) -> dict:
    """{(exp, version, filtro, lado, k, hilos): resumen}. Las versiones secuenciales usan hilos=0.
    En el exp. 4 de la P01 la CPU se midió en cada una de las tres configuraciones: sus filas se
    juntan en una sola referencia secuencial, como hacía la P01."""
    grupos = defaultdict(list)
    for f in filas:
        hilos = f["hilos"] if f["version"] in GPU else 0
        grupos[(f["exp"], f["version"], f["filtro"], f["lado"], f["k"], hilos)].append(f)

    resumen = {}
    for clave, g in grupos.items():
        ok = [f for f in g if f["estado"] == "ok"]
        incompletas = [f for f in g if f["estado"] == "no_completada"]
        frias = [f for f in g if f["estado"] == "calentamiento"]
        estado = "no_completada" if incompletas else "ok" if ok else "sin_datos"
        r = {"n": len(ok), "estado": estado,
             "nota": (incompletas[0]["nota"] if incompletas else
                      next((f["nota"] for f in g if f["nota"]), ""))}
        version = clave[1]
        usadas = ["cpu"] if version in ("c", "python") else ["h2d", "kernel", "d2h", "total", "banda"]
        for m in usadas:
            r[m] = _est([f[m] for f in ok]) if ok else None
        if incompletas:
            r["transcurrido_ms"] = incompletas[0]["cpu"]
        if frias:
            r["frio"] = {m: frias[0][m] for m in usadas}
        elif version == "cuda_c" and clave[:1] + clave[2:] in frio_cudac:
            primero, ultimo = frio_cudac[clave[:1] + clave[2:]]
            r["frio"] = {"kernel": primero, "kernel_fin_calentamiento": ultimo}
        resumen[clave] = r
    return resumen


def tiempo(r: dict | None, metrica: str | None = None):
    """Mediana de la métrica (o del tiempo principal de la versión), None si no hay."""
    if r is None or r["estado"] != "ok":
        return None
    metrica = metrica or ("cpu" if r.get("cpu") else "total")
    return r[metrica]["mediana"] if r.get(metrica) else None


def configuraciones(resumen: dict, exp: int) -> list[tuple]:
    """(filtro, lado, k, hilos) del experimento, en el orden del eje variable."""
    confs = {(c[2], c[3], c[4], c[5]) for c in resumen if c[0] == exp and c[1] in GPU}
    return sorted(confs, key=lambda c: (ORDEN_FILTRO.get(c[0], 9), c[1], c[2], c[3]))


def buscar(resumen, exp, version, filtro, lado, k, hilos):
    return resumen.get((exp, version, filtro, lado, k, hilos if version in GPU else 0))


def unicas(resumen: dict) -> list[tuple]:
    """Configuraciones distintas de los exp. 1-3 (256 hilos), con el primer experimento que la
    tiene. Sirven para las gráficas que comparan versiones sin repetir la misma medición."""
    vistas, salida = set(), []
    for exp in (1, 2, 3):
        for filtro, lado, k, hilos in configuraciones(resumen, exp):
            if (filtro, lado, k) not in vistas:
                vistas.add((filtro, lado, k))
                salida.append((exp, filtro, lado, k, hilos))
    return salida


def etiqueta(filtro, lado, k, hilos=None, corta=False) -> str:
    texto = f"{filtro} {k}×{k}" + ("" if corta else f"\n{lado}×{lado}")
    return texto + (f"\n{hilos} hilos" if hilos and hilos != 256 else "")


def valor_eje(exp, filtro, lado, k, hilos) -> str:
    return {1: filtro, 2: f"{lado}×{lado}", 3: f"{k}×{k}", 4: f"{hilos} hilos"}[exp]


# ── CSV de salida ────────────────────────────────────────────────────────────────────────────

def escribir_resumen(resumen: dict, ruta: Path) -> None:
    columnas = ["experimento", "version", "filtro", "resolucion", "k", "hilos", "n", "estado"]
    for m in METRICAS:
        columnas += [f"{m}_{e}" for e in ("mediana", "desv", "min", "max")]
    columnas += ["frio_kernel_ms", "frio_total_ms", "frio_cpu_ms", "nota"]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for (exp, version, filtro, lado, k, hilos), r in sorted(
                resumen.items(), key=lambda x: (x[0][0], VERSIONES.index(x[0][1]), x[0][2:])):
            fila = {"experimento": exp, "version": version, "filtro": filtro,
                    "resolucion": f"{lado}x{lado}", "k": k, "hilos": hilos, "n": r["n"],
                    "estado": r["estado"], "nota": r["nota"]}
            for m in METRICAS:
                for e in ("mediana", "desv", "min", "max"):
                    fila[f"{m}_{e}"] = f"{r[m][e]:.4f}" if r.get(m) else ""
            frio = r.get("frio", {})
            for m in ("kernel", "total", "cpu"):
                fila[f"frio_{m}_ms"] = f"{frio[m]:.4f}" if m in frio else ""
            w.writerow(fila)


PARES = [("CUDA C vs C/C++", "c", "cuda_c"), ("PyCUDA vs Python", "python", "pycuda"),
         ("Python vs C/C++", "python", "c")]


def aceleraciones(resumen: dict) -> list[dict]:
    filas = []
    for exp in EXPERIMENTOS:
        for filtro, lado, k, hilos in configuraciones(resumen, exp):
            for nombre, lenta, rapida in PARES:
                a = buscar(resumen, exp, lenta, filtro, lado, k, hilos)
                b = buscar(resumen, exp, rapida, filtro, lado, k, hilos)
                ta = tiempo(a)
                if ta is None or tiempo(b) is None:
                    continue  # falta una versión o está incompleta: no hay aceleración
                fila = {"experimento": exp, "filtro": filtro, "resolucion": f"{lado}x{lado}",
                        "k": k, "hilos": hilos, "comparacion": nombre,
                        "lenta_ms": round(ta, 4)}
                if rapida in GPU:
                    fila.update(kernel_ms=round(tiempo(b, "kernel"), 4),
                                total_ms=round(tiempo(b, "total"), 4),
                                aceleracion_computo=round(ta / tiempo(b, "kernel"), 2),
                                aceleracion_con_transferencias=round(ta / tiempo(b, "total"), 2))
                else:  # Python vs C/C++: cuántas veces más lento es Python
                    fila.update(rapida_ms=round(tiempo(b), 4), factor=round(ta / tiempo(b), 1))
                filas.append(fila)
    return filas


def cudac_vs_pycuda(resumen: dict) -> list[dict]:
    filas = []
    for exp in EXPERIMENTOS:
        for filtro, lado, k, hilos in configuraciones(resumen, exp):
            a = buscar(resumen, exp, "cuda_c", filtro, lado, k, hilos)
            b = buscar(resumen, exp, "pycuda", filtro, lado, k, hilos)
            if tiempo(a) is None or tiempo(b) is None:
                continue
            fila = {"experimento": exp, "filtro": filtro, "resolucion": f"{lado}x{lado}",
                    "k": k, "hilos": hilos}
            for m in ("h2d", "kernel", "d2h", "total", "banda"):
                fila[f"cuda_c_{m}"] = round(tiempo(a, m), 4)
                fila[f"pycuda_{m}"] = round(tiempo(b, m), 4)
            for m in ("kernel", "total"):
                fila[f"diferencia_{m}_pct"] = round(100 * (tiempo(b, m) / tiempo(a, m) - 1), 1)
            filas.append(fila)
    return filas


def escribir_filas(filas: list[dict], ruta: Path) -> None:
    if not filas:
        raise FaltanDatos("no hay pares de versiones completas")
    columnas = list(dict.fromkeys(c for f in filas for c in f))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


# ── Tablas LaTeX ─────────────────────────────────────────────────────────────────────────────

CABECERA_TEX = "% Generado por scripts/graficas.py a partir de resultados/; no editar a mano.\n"
NOMBRE_EXP = {1: "Filtro", 2: "Resolución", 3: "Máscara", 4: "Hilos"}


def ms(v) -> str:
    if v is None:
        return "--"
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 1 else f"{v:.3f}"


def tex(texto: str) -> str:
    return texto.replace("×", r"$\times$").replace("%", r"\%")


def celda(r, metrica=None, escala=1.0) -> str:
    if r is None:
        return "--"
    if r["estado"] == "no_completada":
        return "n.c."
    v = tiempo(r, metrica)
    return ms(v / escala) if v is not None else "--"


def tabla(ruta: Path, columnas: str, encabezado: list[str], filas: list[list[str]],
          separar_por: list | None = None, nota: str = "") -> None:
    if not filas:
        raise FaltanDatos("sin filas")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_TEX + (f"% {nota}\n" if nota else ""))
        f.write(f"\\begin{{tabular}}{{{columnas}}}\n\\toprule\n")
        for linea in encabezado:
            f.write(linea + r" \\" + "\n")
        f.write("\\midrule\n")
        anterior = None
        for i, fila in enumerate(filas):
            grupo = separar_por[i] if separar_por else None
            if anterior is not None and grupo != anterior:
                f.write("\\midrule\n")
            anterior = grupo
            f.write(" & ".join(fila) + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def filas_por_experimento(resumen, constructor):
    filas, grupos = [], []
    for exp in EXPERIMENTOS:
        for i, conf in enumerate(configuraciones(resumen, exp)):
            primera = f"{exp} ({NOMBRE_EXP[exp]})" if i == 0 else ""
            fila = constructor(exp, *conf)
            if fila:
                filas.append([primera, tex(valor_eje(exp, *conf))] + fila)
                grupos.append(exp)
    return filas, grupos


def tabla_tiempos(resumen, ruta):
    def fila(exp, filtro, lado, k, hilos):
        b = lambda v: buscar(resumen, exp, v, filtro, lado, k, hilos)  # noqa: E731
        return [celda(b("c")), celda(b("python"), escala=1000),
                celda(b("cuda_c"), "kernel"), celda(b("cuda_c"), "total"),
                celda(b("pycuda"), "kernel"), celda(b("pycuda"), "total")]
    filas, grupos = filas_por_experimento(resumen, fila)
    tabla(ruta, "llrrrrrr", [
        r"Exp. & Valor & C/C++ & Python & \multicolumn{2}{c}{CUDA C (ms)} & \multicolumn{2}{c}{PyCUDA (ms)}",
        r" & & (ms) & (s) & kernel & total & kernel & total"], filas, grupos)


def tabla_variabilidad(resumen, ruta):
    """Coeficiente de variación (desviación / mediana) del tiempo principal de cada versión."""
    def cv(r, metrica=None):
        if r is None or r["estado"] != "ok":
            return "--"
        m = metrica or ("cpu" if r.get("cpu") else "total")
        e = r[m]
        return f"{100 * e['desv'] / e['mediana']:.1f}" if e["mediana"] else "--"

    def fila(exp, filtro, lado, k, hilos):
        b = lambda v: buscar(resumen, exp, v, filtro, lado, k, hilos)  # noqa: E731
        return [cv(b("c")), cv(b("python")), cv(b("cuda_c"), "kernel"), cv(b("cuda_c")),
                cv(b("pycuda"), "kernel"), cv(b("pycuda"))]
    filas, grupos = filas_por_experimento(resumen, fila)
    tabla(ruta, "llrrrrrr", [
        r"Exp. & Valor & C/C++ & Python & \multicolumn{2}{c}{CUDA C} & \multicolumn{2}{c}{PyCUDA}",
        r" & & & & kernel & total & kernel & total"], filas, grupos)


def tabla_aceleraciones(acel, ruta):
    por_conf = defaultdict(dict)
    for a in acel:
        por_conf[(a["experimento"], a["filtro"], int(a["resolucion"].split("x")[0]), a["k"],
                  a["hilos"])][a["comparacion"]] = a
    filas, grupos = [], []
    vistos = set()
    for (exp, filtro, lado, k, hilos), comp in por_conf.items():
        def par(nombre):
            a = comp.get(nombre)
            return ([f"{a['aceleracion_computo']:.0f}", f"{a['aceleracion_con_transferencias']:.0f}"]
                    if a else ["--", "--"])
        python_c = comp.get("Python vs C/C++")
        primera = f"{exp} ({NOMBRE_EXP[exp]})" if exp not in vistos else ""
        vistos.add(exp)
        filas.append([primera, tex(valor_eje(exp, filtro, lado, k, hilos))] + par("CUDA C vs C/C++")
                     + par("PyCUDA vs Python") + [f"{python_c['factor']:.0f}" if python_c else "--"])
        grupos.append(exp)
    tabla(ruta, "llrrrrr", [
        r"Exp. & Valor & \multicolumn{2}{c}{CUDA C vs C/C++} & \multicolumn{2}{c}{PyCUDA vs Python} & Python",
        r" & & cómputo & con transf. & cómputo & con transf. & vs C/C++"], filas, grupos)


def tabla_gpu(comparacion, ruta):
    filas, grupos, vistos = [], [], set()
    for c in comparacion:
        exp = c["experimento"]
        primera = f"{exp} ({NOMBRE_EXP[exp]})" if exp not in vistos else ""
        vistos.add(exp)
        lado = int(c["resolucion"].split("x")[0])
        filas.append([primera, tex(valor_eje(exp, c["filtro"], lado, c["k"], c["hilos"]))] +
                     [ms(c[f"{v}_{m}"]) for m in ("h2d", "kernel", "d2h", "total")
                      for v in GPU] +
                     [f"{c['diferencia_kernel_pct']:+.1f}", f"{c['diferencia_total_pct']:+.1f}"])
        grupos.append(exp)
    tabla(ruta, "llrrrrrrrrrr", [
        r"Exp. & Valor & \multicolumn{2}{c}{CPU$\to$GPU} & \multicolumn{2}{c}{Kernel} & "
        r"\multicolumn{2}{c}{GPU$\to$CPU} & \multicolumn{2}{c}{Total} & \multicolumn{2}{c}{PyCUDA vs CUDA C (\%)}",
        r" & & C & Py & C & Py & C & Py & C & Py & kernel & total"], filas, grupos)


def tabla_banda(resumen, pico, ruta):
    def fila(exp, filtro, lado, k, hilos):
        datos = []
        for v in GPU:
            r = buscar(resumen, exp, v, filtro, lado, k, hilos)
            b = tiempo(r, "banda")
            datos += [f"{b:.1f}", f"{100 * b / pico:.0f}"] if b else ["--", "--"]
        return datos
    filas, grupos = filas_por_experimento(resumen, fila)
    tabla(ruta, "llrrrr", [
        r"Exp. & Valor & \multicolumn{2}{c}{CUDA C} & \multicolumn{2}{c}{PyCUDA}",
        r" & & GB/s & \% del pico & GB/s & \% del pico"], filas, grupos,
        nota=f"Pico de la memoria de la GPU usado para el %: {pico:.1f} GB/s")


def tabla_calentamiento(resumen, ruta):
    def fila(exp, filtro, lado, k, hilos):
        datos = []
        for v in GPU:
            r = buscar(resumen, exp, v, filtro, lado, k, hilos)
            caliente, frio = tiempo(r, "kernel"), (r or {}).get("frio", {}).get("kernel")
            datos += ([ms(frio), ms(caliente), f"{frio / caliente:.1f}"]
                      if frio and caliente else ["--", "--", "--"])
        return datos
    filas, grupos = filas_por_experimento(resumen, fila)
    tabla(ruta, "llrrrrrr", [
        r"Exp. & Valor & \multicolumn{3}{c}{CUDA C, kernel (ms)} & \multicolumn{3}{c}{PyCUDA, kernel (ms)}",
        r" & & frío & caliente & factor & frío & caliente & factor"], filas, grupos)


def tabla_validacion(validacion, ruta):
    por_version = defaultdict(list)
    for v in validacion:
        por_version[v["version"]].append(v)
    filas = []
    for version in ("cuda_c", "python", "pycuda"):
        vs = por_version.get(version)
        if not vs:
            continue
        maximo = max(vs, key=lambda v: float(v["error_max"]))
        dentro = sum(v["dentro_tolerancia"] == "True" for v in vs)
        filas.append([NOMBRE[version], tex(vs[0]["referencia"].replace("c_cpu", "C/C++")),
                      f"{float(maximo['error_max']):.2e}",
                      tex(f"{maximo['filtro']} {maximo['k']}×{maximo['k']} {maximo['resolucion'].replace('x', '×')}"),
                      f"{max(float(v['error_medio']) for v in vs):.2e}", f"{dentro}/{len(vs)}"])
    tabla(ruta, "llrlrr", [r"Versión & Referencia & Error máx. & Peor caso & Error medio máx. & "
                           r"Dentro de tol."], filas)


def tabla_inicio(ruta_csv, ruta):
    fila = leer_csv(ruta_csv)[0]
    tabla(ruta, "rr", [r"Crear el contexto (ms) & Compilar el kernel (ms)"],
          [[ms(_f(fila["contexto_ms"])), ms(_f(fila["compilacion_ms"]))]])


# ── Gráficas ─────────────────────────────────────────────────────────────────────────────────

def _barra_error(r, metrica):
    e = r[metrica]
    return [[e["mediana"] - e["min"]], [e["max"] - e["mediana"]]]


SERIES = [("c", None, "C/C++"), ("python", None, "Python"),
          ("cuda_c", "kernel", "CUDA C kernel"), ("cuda_c", "total", "CUDA C total"),
          ("pycuda", "kernel", "PyCUDA kernel"), ("pycuda", "total", "PyCUDA total")]


def _estilo(version, metrica):
    kernel = metrica == "kernel"
    return {"color": COLOR[version], "alpha": 0.55 if kernel else 1.0,
            "hatch": "//" if kernel else None}


def grafica_experimento(resumen, exp, ruta):
    """Tiempo (mediana, barras de error = mínimo y máximo) de cada versión, escala log."""
    confs = configuraciones(resumen, exp)
    if not confs:
        raise FaltanDatos(f"sin datos GPU del exp. {exp}")
    titulo = EXPERIMENTOS[exp][0]
    series = [s for s in SERIES if exp != 4 or s[0] in GPU]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.8 / len(series)
    hay, no_completadas = False, []
    for j, (version, metrica, nombre) in enumerate(series):
        for i, conf in enumerate(confs):
            r = buscar(resumen, exp, version, *conf)
            v = tiempo(r, metrica)
            x = i + (j - (len(series) - 1) / 2) * ancho
            if v is None:
                if r is not None and r["estado"] == "no_completada":
                    no_completadas.append((x, version))
                continue
            m = metrica or ("cpu" if version in ("c", "python") else "total")
            ax.bar(x, v, ancho, yerr=_barra_error(r, m), capsize=2, edgecolor="black",
                   linewidth=0.4, label=nombre, **_estilo(version, metrica))
            hay = True
    if not hay:
        raise FaltanDatos(f"exp. {exp} sin tiempos completos")
    ax.set_yscale("log")
    for x, version in no_completadas:  # se marca dónde iría la barra, sin inventar el valor
        ax.text(x, ax.get_ylim()[0] * 1.5, "no completada", rotation=90, fontsize=6,
                ha="center", va="bottom", color=COLOR[version])
    ax.set_ylabel("tiempo (ms, escala log)")
    ax.set_xticks(range(len(confs)), [valor_eje(exp, *c) for c in confs])
    ax.set_title(titulo + "\nmediana de 5 repeticiones; barras de error: mínimo y máximo")
    _leyenda_unica(ax)
    _guardar(fig, ruta)


def _leyenda_unica(ax, ncol=3):
    """Leyenda debajo del gráfico, fuera de los ejes, para que nunca tape barras."""
    fig = ax.figure
    previas = getattr(fig, "_leyenda", ({}, ncol))[0]
    for h, etiqueta_serie in zip(*ax.get_legend_handles_labels()):
        previas.setdefault(etiqueta_serie, h)
    fig._leyenda = (previas, ncol)


def _guardar(fig, ruta):
    leyenda = getattr(fig, "_leyenda", None)
    if leyenda and leyenda[0]:
        unicos, ncol = leyenda
        filas = -(-len(unicos) // ncol)
        fig.tight_layout(rect=(0, 0.05 + 0.05 * filas, 1, 1))
        fig.legend(unicos.values(), unicos.keys(), loc="lower center", ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0))
    else:
        fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def grafica_hilos(resumen, ruta):
    confs = configuraciones(resumen, 4)
    if not confs:
        raise FaltanDatos("sin datos del exp. 4")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metrica in zip(axs, ("kernel", "total")):
        for j, version in enumerate(GPU):
            for i, conf in enumerate(confs):
                r = buscar(resumen, 4, version, *conf)
                if tiempo(r, metrica) is None:
                    continue
                ax.bar(i + (j - 0.5) * 0.38, tiempo(r, metrica), 0.38,
                       yerr=_barra_error(r, metrica), capsize=3, color=COLOR[version],
                       edgecolor="black", linewidth=0.4, label=NOMBRE[version])
        ax.set_xticks(range(len(confs)), [f"{c[3]}" for c in confs])
        ax.set_xlabel("hilos por bloque")
        ax.set_ylabel(f"{'kernel' if metrica == 'kernel' else 'total con transferencias'} (ms)")
        _leyenda_unica(ax, ncol=2)
    fig.suptitle(EXPERIMENTOS[4][0] + " - mediana, barras: mínimo y máximo")
    _guardar(fig, ruta)


def grafica_aceleraciones(acel, ruta):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    hay, maximo = False, 1.0
    for ax, (nombre, _, rapida) in zip(axs, PARES[:2]):
        datos = [a for a in acel if a["comparacion"] == nombre and a["experimento"] in (1, 2, 3)]
        vistos, filas = set(), []
        for a in datos:  # una barra por configuración distinta
            clave = (a["filtro"], a["resolucion"], a["k"])
            if clave not in vistos:
                vistos.add(clave)
                filas.append(a)
        if not filas:
            ax.text(0.5, 0.5, "sin datos completos", ha="center", transform=ax.transAxes)
            ax.set_title(nombre)
            continue
        hay = True
        x = range(len(filas))
        for desp, campo, nombre_serie, estilo in (
                (-0.2, "aceleracion_computo", "solo cómputo (kernel)", {"alpha": 0.55, "hatch": "//"}),
                (0.2, "aceleracion_con_transferencias", "con transferencias", {})):
            valores = [a[campo] for a in filas]
            barras = ax.bar([i + desp for i in x], valores, 0.4, color=COLOR[rapida],
                            edgecolor="black", linewidth=0.4, label=nombre_serie, **estilo)
            ax.bar_label(barras, fmt="%.0f×", fontsize=6, padding=1)
            maximo = max(maximo, *valores)
        ax.set_xticks(list(x), [f"{a['filtro']}\n{a['k']}×{a['k']}, {a['resolucion'].split('x')[0]}"
                                for a in filas], fontsize=7)
        ax.set_yscale("log")
        ax.set_title(nombre)
        _leyenda_unica(ax, ncol=2)
    if not hay:
        raise FaltanDatos("ninguna aceleración completa")
    axs[0].set_ylabel("aceleración (veces, escala log)")
    axs[0].set_ylim(top=maximo * 4)  # espacio para las etiquetas sobre las barras
    fig.suptitle("Aceleración: mediana secuencial / mediana GPU")
    _guardar(fig, ruta)


def grafica_desglose(resumen, ruta):
    confs = [c for c in unicas(resumen)
             if all(tiempo(buscar(resumen, c[0], v, *c[1:]), "total") for v in GPU)]
    if not confs:
        raise FaltanDatos("no hay configuraciones con CUDA C y PyCUDA completas")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (exp, *conf) in enumerate(confs):
        for j, version in enumerate(GPU):
            r = buscar(resumen, exp, version, *conf)
            x, base = i + (j - 0.5) * 0.4, 0.0
            for parte in ("h2d", "kernel", "d2h"):
                v = tiempo(r, parte)
                ax.bar(x, v, 0.38, bottom=base, color=COLOR_PARTE[parte], edgecolor="black",
                       linewidth=0.4, hatch="//" if version == "pycuda" else None,
                       label={"h2d": "CPU→GPU", "kernel": "kernel", "d2h": "GPU→CPU"}[parte])
                base += v
    ax.set_xticks(range(len(confs)), [etiqueta(*c[1:]) for c in confs], fontsize=7)
    ax.set_ylabel("tiempo (ms)")
    ax.set_title("CUDA C (liso) y PyCUDA (rayado): desglose del tiempo GPU, medianas")
    _leyenda_unica(ax, ncol=3)
    _guardar(fig, ruta)


def grafica_banda(resumen, pico, origen_pico, ruta):
    confs = [c for c in unicas(resumen)
             if any(tiempo(buscar(resumen, c[0], v, *c[1:]), "banda") for v in GPU)]
    confs += [(4, *c) for c in configuraciones(resumen, 4) if c[3] != 256]
    if not confs:
        raise FaltanDatos("sin ancho de banda")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (exp, *conf) in enumerate(confs):
        for j, version in enumerate(GPU):
            b = tiempo(buscar(resumen, exp, version, *conf), "banda")
            if not b:
                continue
            barra = ax.bar(i + (j - 0.5) * 0.4, b, 0.38, color=COLOR[version], edgecolor="black",
                           linewidth=0.4, label=NOMBRE[version])
            ax.bar_label(barra, labels=[f"{100 * b / pico:.0f}%"], fontsize=6, padding=1)
    ax.axhline(pico, color="black", linestyle="--", linewidth=1)
    ax.text(len(confs) - 0.5, pico, f"pico {pico:.0f} GB/s ({origen_pico})", ha="right",
            va="bottom", fontsize=7)
    ax.set_xticks(range(len(confs)), [etiqueta(*c[1:]) for c in confs], fontsize=7)
    ax.set_ylabel("GB/s efectivos (2 × píxeles × 4 B / kernel)")
    ax.set_ylim(0, pico * 1.15)
    ax.set_title("Ancho de banda efectivo del kernel frente al pico de la memoria de la GPU\n"
                 "(el % sobre cada barra es la fracción del pico)")
    _leyenda_unica(ax, ncol=2)
    _guardar(fig, ruta)


def grafica_calentamiento(resumen, ruta):
    confs = [c for c in unicas(resumen)
             if any((buscar(resumen, c[0], v, *c[1:]) or {}).get("frio") for v in GPU)]
    if not confs:
        raise FaltanDatos("sin llamadas frías registradas")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (exp, *conf) in enumerate(confs):
        for j, version in enumerate(GPU):
            r = buscar(resumen, exp, version, *conf)
            frio, caliente = (r or {}).get("frio", {}).get("kernel"), tiempo(r, "kernel")
            base = i + (j - 0.5) * 0.4
            if frio:
                ax.bar(base - 0.1, frio, 0.2, color=COLOR[version], alpha=0.45, hatch="..",
                       edgecolor="black", linewidth=0.4, label=f"{NOMBRE[version]} sin calentar")
            if caliente:
                ax.bar(base + 0.1, caliente, 0.2, color=COLOR[version], edgecolor="black",
                       linewidth=0.4, label=f"{NOMBRE[version]} tras calentar (mediana)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(confs)), [etiqueta(*c[1:]) for c in confs], fontsize=7)
    ax.set_ylabel("kernel (ms, escala log)")
    ax.set_title("Kernel en la primera llamada de cada configuración y tras 2 s de calentamiento")
    _leyenda_unica(ax, ncol=2)
    _guardar(fig, ruta)


def leer_estado_gpu(rutas: list[Path]) -> list[dict]:
    muestras = []
    for ruta in rutas:
        with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
            for fila in csv.reader(f):
                if len(fila) < 8 or fila[0].strip().startswith("timestamp"):
                    continue
                try:
                    t = datetime.strptime(fila[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                    numero = lambda s: float(re.sub(r"[^\d.]", "", s) or "nan")  # noqa: E731
                    muestras.append({"t": t, "pstate": fila[1].strip(), "sm": numero(fila[2]),
                                     "mem": numero(fila[3]), "pcie": numero(fila[4]),
                                     "potencia": numero(fila[6]), "temp": numero(fila[7])})
                except ValueError:
                    continue
    return sorted(muestras, key=lambda m: m["t"])


def ventanas_a_fechas(ventanas: list[tuple[str, str]], inicio: datetime) -> list[tuple]:
    """Las ventanas solo traen la hora: se les pone la fecha de la primera muestra (y un día más
    si la corrida pasó la medianoche)."""
    salida = []
    for a, b in ventanas:
        fechas = []
        for h in (a, b):
            t = datetime.combine(inicio.date(), datetime.strptime(h, "%H:%M:%S.%f").time())
            fechas.append(t + timedelta(days=1) if t < inicio - timedelta(minutes=1) else t)
        salida.append(tuple(fechas))
    return salida


def grafica_estado_gpu(muestras, ventanas, titulo, ruta):
    if len(muestras) < 2:
        raise FaltanDatos("sin muestras de nvidia-smi")
    t0 = muestras[0]["t"]
    minutos = [(m["t"] - t0).total_seconds() / 60 for m in muestras]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(minutos, [m["sm"] for m in muestras], lw=0.8, label="reloj SM (MHz)", color="#C44E52")
    ax1.plot(minutos, [m["mem"] for m in muestras], lw=0.8, label="reloj memoria (MHz)",
             color="#4C72B0")
    ax2.step(minutos, [m["pcie"] for m in muestras], lw=0.8, where="post", color="#55A868",
             label="generación PCIe")
    ax2b = ax2.twinx()
    ax2b.plot(minutos, [m["temp"] for m in muestras], lw=0.8, color="#8172B2",
              label="temperatura (°C)")
    ax2b.grid(False)
    for i, (a, b) in enumerate(ventanas_a_fechas([v[:2] for v in ventanas], t0)):
        for ax in (ax1, ax2):
            ax.axvspan((a - t0).total_seconds() / 60, (b - t0).total_seconds() / 60 + 0.02,
                       color="orange", alpha=0.35, lw=0,
                       label="ventana medida" if i == 0 and ax is ax1 else None)
    ax1.set_ylabel("MHz")
    ax2.set_ylabel("PCIe gen")
    ax2b.set_ylabel("°C")
    ax2.set_xlabel("minutos desde el inicio del registro")
    ax1.legend(loc="lower right", ncol=3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", ncol=2)
    ax1.set_title(titulo + " (nvidia-smi cada 200 ms; en naranja, las repeticiones medidas)")
    _guardar(fig, ruta)


def etiqueta_ventana(filtro, lado, k, hilos) -> str:
    return f"{filtro} {k}×{k}, {lado}×{lado}, {hilos} hilos"


def ventanas_p01(ruta: Path) -> list[tuple[str, str, str]]:
    """(inicio, fin, configuración) de cada ventana medida en la consola de la P01; la
    configuración sale de la línea siguiente («media 3x3 4000x4000, 256 hilos ...»)."""
    patron = re.compile(r"GPU medida de (\d\d:\d\d:\d\d\.\d+) a (\d\d:\d\d:\d\d\.\d+)"
                        r"(?:[ \t]*\n[ \t]*(\w+) (\d+)x\d+ (\d+)x\d+, (\d+) hilos)?")
    salida = []
    for a, b, filtro, k, lado, hilos in patron.findall(
            ruta.read_text(encoding="utf-8", errors="replace")):
        etq = etiqueta_ventana(filtro, lado, k, hilos) if filtro else ""
        salida.append((a, b, etq))
    return salida


def ventanas_p02(ruta: Path) -> list[tuple[str, str, str]]:
    return [(v["inicio"], v["fin"], etiqueta_ventana(v["filtro"], v["resolucion"].split("x")[0],
                                                      v["k"], v["hilos"]))
            for v in leer_csv(ruta)]


MARGEN_ANTES_S = 0.5  # las repeticiones duran menos que el periodo de 200 ms de nvidia-smi:
MARGEN_DESPUES_S = 0.3  # se toman las muestras del final del calentamiento y justo después


def estado_en_ventanas(muestras, ventanas, practica) -> list[dict]:
    """Estado de la GPU en cada configuración medida: muestras de nvidia-smi desde 0,5 s antes de
    las repeticiones (final del calentamiento, GPU ya ocupada) hasta 0,3 s después."""
    if not muestras:
        raise FaltanDatos("sin muestras de nvidia-smi")
    filas = []
    for (a, b), (_, _, etq) in zip(ventanas_a_fechas([v[:2] for v in ventanas], muestras[0]["t"]),
                                   ventanas):
        dentro = [m for m in muestras
                  if a - timedelta(seconds=MARGEN_ANTES_S) <= m["t"] <= b + timedelta(seconds=MARGEN_DESPUES_S)]
        if not dentro:
            filas.append({"practica": practica, "configuracion": etq, "muestras": 0})
            continue
        rango = lambda clave: (min(m[clave] for m in dentro), max(m[clave] for m in dentro))  # noqa: E731
        filas.append({"practica": practica, "configuracion": etq, "muestras": len(dentro),
                      "pstate": "/".join(sorted({m["pstate"] for m in dentro})),
                      "sm_min": rango("sm")[0], "sm_max": rango("sm")[1],
                      "mem_min": rango("mem")[0], "mem_max": rango("mem")[1],
                      "pcie_min": rango("pcie")[0], "temp_max": rango("temp")[1],
                      "potencia_max": rango("potencia")[1]})
    return filas


def tabla_estado_gpu(filas, ruta):
    def rango(f, clave):
        a, b = f[f"{clave}_min"], f[f"{clave}_max"]
        return f"{a:.0f}" if a == b else f"{a:.0f}--{b:.0f}"
    cuerpo, grupos = [], []
    for f in filas:
        if not f["muestras"]:
            cuerpo.append([f["practica"], tex(f["configuracion"]), "0"] + ["--"] * 6)
        else:
            cuerpo.append([f["practica"], tex(f["configuracion"]), str(f["muestras"]), f["pstate"],
                           rango(f, "sm"), rango(f, "mem"), f"{f['pcie_min']:.0f}",
                           f"{f['temp_max']:.1f}", f"{f['potencia_max']:.1f}"])
        grupos.append(f["practica"])
    tabla(ruta, "llrlrrrrr",
          [r"Práctica & Configuración & Muestras & Estado & SM (MHz) & Memoria (MHz) & "
           r"PCIe gen & T máx. ($^\circ$C) & P máx. (W)"], cuerpo, grupos,
          nota="muestras de nvidia-smi de 0,5 s antes a 0,3 s después de las repeticiones")


def grafica_estado_ventanas(muestras, ventanas, titulo, ruta):
    """Una vista por configuración: relojes desde 3 s antes de las repeticiones medidas hasta
    1 s después. En la P02 el registro dura más de una hora (casi todo Python en la CPU) y las
    repeticiones PyCUDA no se ven en la línea de tiempo completa."""
    if len(muestras) < 2 or not ventanas:
        raise FaltanDatos("sin muestras de nvidia-smi o sin ventanas")
    fechas = ventanas_a_fechas([v[:2] for v in ventanas], muestras[0]["t"])
    columnas = min(6, len(ventanas))
    filas_fig = -(-len(ventanas) // columnas)
    fig, axs = plt.subplots(filas_fig, columnas, figsize=(1.75 * columnas + 1, 2.1 * filas_fig + 0.6),
                            sharey=True, squeeze=False)
    for i, ((a, b), (_, _, etq)) in enumerate(zip(fechas, ventanas)):
        ax = axs[i // columnas][i % columnas]
        cerca = [m for m in muestras
                 if a - timedelta(seconds=3) <= m["t"] <= b + timedelta(seconds=1)]
        x = [(m["t"] - a).total_seconds() for m in cerca]
        ax.step(x, [m["sm"] for m in cerca], where="post", lw=1, color="#C44E52",
                label="reloj SM (MHz)")
        ax.step(x, [m["mem"] for m in cerca], where="post", lw=1, color="#4C72B0",
                label="reloj memoria (MHz)")
        ax.axvspan(-2, 0, color="grey", alpha=0.15, lw=0, label="calentamiento (2 s)")
        ax.axvspan(0, max((b - a).total_seconds(), 0.05), color="orange", alpha=0.6, lw=0,
                   label="repeticiones medidas")
        ax.set_xlim(-3, 1)
        ax.set_ylim(0, 9000)
        ax.set_title(etq.replace(", ", "\n", 1), fontsize=7)
        ax.tick_params(labelsize=7)
        if i % columnas == 0:
            ax.set_ylabel("MHz", fontsize=8)
        if i // columnas == filas_fig - 1:
            ax.set_xlabel("s respecto a las repeticiones", fontsize=7)
        _leyenda_unica(ax, ncol=4)
    for j in range(len(ventanas), filas_fig * columnas):
        axs[j // columnas][j % columnas].axis("off")
    fig.suptitle(titulo + " (nvidia-smi cada 200 ms)")
    _guardar(fig, ruta)


# ── Principal ────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados")
    dir_graf, dir_tab = base / "graficas", base / "tablas"
    dir_graf.mkdir(parents=True, exist_ok=True)
    dir_tab.mkdir(parents=True, exist_ok=True)
    print(f"Gráficas y tablas a partir de {base}/")

    entradas = {n: base / n for n in ("mediciones.csv", "p01_mediciones.csv",
                                      "p01_medir_consola.txt", "validacion.csv",
                                      "pycuda_inicio.csv", "ventanas_gpu.csv",
                                      "p01_gpu_estado.csv", "equipo.txt")}
    for nombre, ruta in entradas.items():
        print(f"  {'leído ' if ruta.exists() else 'FALTA '} {nombre}")

    filas = []
    if entradas["mediciones.csv"].exists():
        filas += cargar_p02(entradas["mediciones.csv"])
    if entradas["p01_mediciones.csv"].exists():
        filas += cargar_p01(entradas["p01_mediciones.csv"])
    if not filas:
        print("No hay mediciones: no se genera nada.")
        return 1
    frio = frio_p01(entradas["p01_medir_consola.txt"]) if entradas["p01_medir_consola.txt"].exists() else {}
    resumen = resumir(filas, frio)
    pico, origen_pico = pico_gbs(entradas["equipo.txt"])
    acel = aceleraciones(resumen)
    gpu = cudac_vs_pycuda(resumen)
    por_version = defaultdict(int)
    for f in filas:
        por_version[f["version"]] += f["estado"] == "ok"
    print("  repeticiones válidas: " + ", ".join(f"{NOMBRE[v]} {por_version[v]}" for v in VERSIONES))
    incompletas = [c for c, r in resumen.items() if r["estado"] == "no_completada"]
    for c in incompletas:
        print(f"  no completada: exp {c[0]} {NOMBRE[c[1]]} {c[2]} {c[4]}x{c[4]} {c[3]}  ({resumen[c]['nota']})")
    print(f"  ancho de banda pico: {pico:.1f} GB/s ({origen_pico})")

    def leer_si(nombre, funcion):
        if not entradas[nombre].exists():
            raise FaltanDatos(f"falta {nombre}")
        return funcion(entradas[nombre])

    estado_p02 = sorted(base.glob("gpu_estado*.csv"))

    def estado_ventanas():
        filas_estado = []
        for practica, ruta_estado, ventanas in (
                ("P01", [entradas["p01_gpu_estado.csv"]], ("p01_medir_consola.txt", ventanas_p01)),
                ("P02", estado_p02, ("ventanas_gpu.csv", ventanas_p02))):
            rutas = [p for p in ruta_estado if p.exists()]
            if rutas and entradas[ventanas[0]].exists():
                filas_estado += estado_en_ventanas(leer_estado_gpu(rutas), leer_si(*ventanas),
                                                   practica)
        if not filas_estado:
            raise FaltanDatos("sin registros de nvidia-smi con ventanas")
        return filas_estado
    tareas = [
        (base / "resumen.csv", lambda r: escribir_resumen(resumen, r)),
        (base / "aceleraciones.csv", lambda r: escribir_filas(acel, r)),
        (base / "cudac_vs_pycuda.csv", lambda r: escribir_filas(gpu, r)),
        (dir_tab / "tabla_tiempos.tex", lambda r: tabla_tiempos(resumen, r)),
        (dir_tab / "tabla_variabilidad.tex", lambda r: tabla_variabilidad(resumen, r)),
        (dir_tab / "tabla_aceleraciones.tex", lambda r: tabla_aceleraciones(acel, r)),
        (dir_tab / "tabla_cudac_pycuda.tex", lambda r: tabla_gpu(gpu, r)),
        (dir_tab / "tabla_ancho_banda.tex", lambda r: tabla_banda(resumen, pico, r)),
        (dir_tab / "tabla_calentamiento.tex", lambda r: tabla_calentamiento(resumen, r)),
        (dir_tab / "tabla_validacion.tex",
         lambda r: tabla_validacion(leer_si("validacion.csv", leer_csv), r)),
        (dir_tab / "tabla_inicio_pycuda.tex",
         lambda r: leer_si("pycuda_inicio.csv", lambda p: tabla_inicio(p, r))),
        (dir_graf / "exp1_filtros.png", lambda r: grafica_experimento(resumen, 1, r)),
        (dir_graf / "exp2_resolucion.png", lambda r: grafica_experimento(resumen, 2, r)),
        (dir_graf / "exp3_mascara.png", lambda r: grafica_experimento(resumen, 3, r)),
        (dir_graf / "exp4_hilos.png", lambda r: grafica_hilos(resumen, r)),
        (dir_graf / "aceleraciones.png", lambda r: grafica_aceleraciones(acel, r)),
        (dir_graf / "cudac_vs_pycuda_desglose.png", lambda r: grafica_desglose(resumen, r)),
        (dir_graf / "ancho_banda.png", lambda r: grafica_banda(resumen, pico, origen_pico, r)),
        (dir_graf / "calentamiento.png", lambda r: grafica_calentamiento(resumen, r)),
        (dir_graf / "estado_gpu_p02.png", lambda r: grafica_estado_gpu(
            leer_estado_gpu(estado_p02), leer_si("ventanas_gpu.csv", ventanas_p02),
            "Estado de la GPU durante la P02 (PyCUDA)", r)),
        (dir_graf / "estado_gpu_p02_ventanas.png", lambda r: grafica_estado_ventanas(
            leer_estado_gpu(estado_p02), leer_si("ventanas_gpu.csv", ventanas_p02),
            "PyCUDA: estado de la GPU alrededor de cada configuración medida", r)),
        (dir_graf / "estado_gpu_p01.png", lambda r: grafica_estado_gpu(
            leer_si("p01_gpu_estado.csv", lambda p: leer_estado_gpu([p])),
            leer_si("p01_medir_consola.txt", ventanas_p01),
            "Estado de la GPU durante la P01 (CUDA C)", r)),
        (base / "estado_gpu_ventanas.csv", lambda r: escribir_filas(estado_ventanas(), r)),
        (dir_tab / "tabla_estado_gpu.tex", lambda r: tabla_estado_gpu(estado_ventanas(), r)),
    ]

    generadas, omitidas, errores = 0, 0, 0
    print("Salidas:")
    for ruta, tarea in tareas:
        nombre = ruta.relative_to(base)
        try:
            tarea(ruta)
            generadas += 1
            print(f"  ok        {nombre}")
        except FaltanDatos as motivo:
            omitidas += 1
            print(f"  omitida   {nombre}: {motivo}")
        except Exception as fallo:  # una salida rota no debe impedir las demás
            errores += 1
            plt.close("all")
            print(f"  ERROR     {nombre}: {type(fallo).__name__}: {fallo}")
            traceback.print_exc(limit=3)
    print(f"Generadas {generadas}, omitidas {omitidas} (faltan datos), con error {errores}.")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
