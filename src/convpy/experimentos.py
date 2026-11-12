"""Los cuatro experimentos obligatorios. Bloque 3 - Andrés Morocho.

| Experimento | Configuración                          | Versiones           |
| Filtros     | media, nitidez, laplaciano 3x3; 4000   | las cuatro          |
| Resolución  | media 9x9; 2000, 4000, 6000            | las cuatro          |
| Máscara     | media 3x3, 9x9, 21x21; 4000            | las cuatro          |
| Hilos       | media 9x9; 4000; 128, 256, 512         | CUDA C y PyCUDA     |

Un factor a la vez; fuera del experimento de hilos se fija 256.
"""
