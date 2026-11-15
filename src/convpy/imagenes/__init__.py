"""SECCIÓN 1 · Imágenes: cargar (cualquier formato o URL -> gris float32) y guardar."""
from .carga import cargar_gris, describir, descargar, es_url
from .guardado import a_rango_visual, cargar_crudo, guardar_crudo, guardar_visual

__all__ = ["cargar_gris", "describir", "descargar", "es_url",
           "a_rango_visual", "guardar_visual", "guardar_crudo", "cargar_crudo"]
