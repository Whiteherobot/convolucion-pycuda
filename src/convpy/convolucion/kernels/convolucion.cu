// Kernel de convolución para PyCUDA. Bloque 2 - Jordy Romero
// Es el kernel_convolucion de practica01/src/modelo/convolucion_gpu.cu, sin cambios de lógica.

// Un hilo calcula un píxel de salida; bloques e hilos solo en X.
__global__ void kernel_convolucion(const float *entrada, float *salida, int ancho, int alto,
                                   const float *mascara, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int fila, columna, i, j;
    int radio = k / 2;

    // El último bloque puede tener hilos de más: esos no hacen nada.
    if (tid < ancho * alto) {
        fila = tid / ancho;
        columna = tid % ancho;
        float suma = 0.0f;
        for (i = 0; i < k; i++) {
            for (j = 0; j < k; j++) {
                int y = fila + i - radio;
                int x = columna + j - radio;
                if (y >= 0 && y < alto && x >= 0 && x < ancho)
                    suma += entrada[y * ancho + x] * mascara[i * k + j];
            }
        }
        salida[tid] = suma;
    }
}
