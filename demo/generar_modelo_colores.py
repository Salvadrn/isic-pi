"""Genera demo/colores/: un modelo que reconoce 8 colores, entrenado con los notebooks 02 y 03 del proyecto
(el mismo código y la misma arquitectura que el modelo de lesiones), sobre imágenes sintéticas de colores sólidos.

Sirve para probar la cámara y el programa de la Pi mientras no existe el modelo real: le enseñas algo rojo y
debe decir «rojo». No sabe nada de piel.

Necesita el entorno de las pruebas (Python 3.12 con TensorFlow y ai-edge-litert; ver el README):
    python demo/generar_modelo_colores.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "tests"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import probar_nb02_entrenar as t2      # noqa: E402
import probar_nb03_exportar as t3      # noqa: E402

COLORES = [("rojo", 0), ("naranja", 30), ("amarillo", 58), ("verde", 120),
           ("cian", 185), ("azul", 225), ("morado", 275), ("rosa", 320)]
DESTINO = RAIZ / "demo" / "colores"

if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        t2.fabricar_datos_colores(d, colores=COLORES)
        t2.correr02(False, reemplazos={"EPOCAS_CABEZA": 6, "LR_CABEZA": 1e-2, "EPOCAS_AJUSTE": 1,
                                       "PACIENCIA": 20, "BATCH": 32, "CAPAS_A_AJUSTAR": 20})
        ns, _ = t3.correr03(False, reemplazos={"EXPORTAR_LIGERO": False})
        for nombre in ["modelo.tflite", "clases.txt", "muestras_pi.npz", "metricas_test.json"]:
            shutil.copy2(ns["SALIDA"] / nombre, DESTINO / nombre)
    m = json.loads((DESTINO / "metricas_test.json").read_text())["modelos"]["float32"]
    print(f"demo/colores listo: recall promedio en test {m['recall_promedio']:.3f}, {m['mb']:.1f} MB")
