# Pruebas (Bloque 2 - Jordi Romero)

- `test_manual.py`: matriz 5x5 de la figura 1 → 78 y 87 en Python secuencial y PyCUDA
  (C y CUDA C ya lo hacen con `practica01 ... probar`).
- `test_limites.py`: imagen 1001x777 (no múltiplo de 128/256/512 hilos): bordes y cobertura total.
- `test_equivalencia.py`: las cuatro versiones contra C/C++ secuencial; error máximo y medio
  antes del recorte, dentro de `TOLERANCIA`.

Ejecutar: `pytest -v`
