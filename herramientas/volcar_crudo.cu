// Vuelca la salida float SIN recortar de C/C++ y CUDA C. Bloque 2 - Jordi Romero
//   volcar_crudo <entrada.pgm> <filtro> <k> <prefijo>   ->   <prefijo>_cpu.f32 y <prefijo>_gpu.f32
// Usa los fuentes de practica01/src/modelo tal cual, para comparar las cuatro versiones.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "imagen.h"
#include "mascara.h"
#include "convolucion_cpu.h"
#include "convolucion_gpu.h"

#define HILOS 256

static int guardar_f32(const char *ruta, const Imagen *img)
{
    FILE *archivo = fopen(ruta, "wb");
    size_t n = (size_t)img->ancho * img->alto;
    if (archivo == NULL)
        return -1;
    if (fwrite(img->datos, sizeof(float), n, archivo) != n) {
        fclose(archivo);
        return -1;
    }
    fclose(archivo);
    return 0;
}

// Las tres máscaras de la práctica 01; crear_mascara sola no rellena los coeficientes.
static int elegir_mascara(const char *nombre, int k, Mascara *m)
{
    if (strcmp(nombre, "media") == 0 && k > 0 && k % 2 == 1)
        *m = mascara_media(k);
    else if (strcmp(nombre, "nitidez") == 0 && k == 3)
        *m = mascara_nitidez();
    else if (strcmp(nombre, "laplaciano") == 0 && k == 3)
        *m = mascara_laplaciano();
    else
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    Imagen entrada, cpu, gpu;
    Mascara mascara;
    TiemposGPU tiempos;
    char ruta[512];

    if (argc != 5) {
        fprintf(stderr, "uso: %s <entrada.pgm> <filtro> <k> <prefijo>\n", argv[0]);
        return 1;
    }
    if (leer_pgm(argv[1], &entrada) != 0) {
        fprintf(stderr, "no se pudo leer %s\n", argv[1]);
        return 1;
    }
    if (elegir_mascara(argv[2], atoi(argv[3]), &mascara) != 0) {
        fprintf(stderr, "filtro no valido: %s %s\n", argv[2], argv[3]);
        return 1;
    }
    cpu = crear_imagen(entrada.ancho, entrada.alto);
    gpu = crear_imagen(entrada.ancho, entrada.alto);

    convolucion_cpu(&entrada, &cpu, &mascara);
    convolucion_gpu(&entrada, &gpu, &mascara, HILOS, &tiempos);

    snprintf(ruta, sizeof(ruta), "%s_cpu.f32", argv[4]);
    if (guardar_f32(ruta, &cpu) != 0)
        return 1;
    snprintf(ruta, sizeof(ruta), "%s_gpu.f32", argv[4]);
    if (guardar_f32(ruta, &gpu) != 0)
        return 1;

    liberar_imagen(&entrada);
    liberar_imagen(&cpu);
    liberar_imagen(&gpu);
    liberar_mascara(&mascara);
    return 0;
}
