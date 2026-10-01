"""Prueba el notebook 03 sin Colab: entrena un modelo chico con el notebook 02 sobre datos falsos y luego
ejecuta las celdas del 03.

Qué comprueba: que el TFLite float32 da las mismas probabilidades que Keras, que el ligero es más chico,
que todo se mide con ai-edge-litert (el intérprete de la Pi), que el zip trae lo mismo que la carpeta, que
las muestras para la Pi se reproducen leyendo el .tflite de disco, y que las guardas (sin 01, sin 02, ya
exportado) fallan con un mensaje claro. Con 8 clases de colores sólidos, el recall en test debe ser alto.

Qué NO comprueba: calidad con ISIC real, ni que el intérprete ARM de la Pi dé lo mismo que el de esta Mac
(eso lo comprueban las muestras en la sesión 4).

Uso (Python 3.12 con lo del test 02 + ai-edge-litert):
    python tests/probar_nb03_exportar.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import probar_nb02_entrenar as t2      # arma el dataset falso del 01 y trae el ejecutor del 02
from probar_nb01_datos import celdas_codigo

from ai_edge_litert.interpreter import Interpreter

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "03_exportar_tflite.ipynb"
ARCHIVOS = ["modelo.tflite", "modelo_ligero.tflite", "clases.txt", "muestras_pi.npz",
            "reporte_test.txt", "confusion_test.png", "metricas_test.json"]


def correr03(modo_prueba, forzar=False, reemplazos=None, esperar_error=None):
    ns = {"__name__": "__main__"}
    shutil.rmtree(Path.cwd() / "trabajo", ignore_errors=True)     # runtime nuevo
    for n, fuente in enumerate(celdas_codigo(NOTEBOOK)):
        fuente = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
        if "MODO_PRUEBA = " in fuente:
            fuente = re.sub(r"MODO_PRUEBA = \w+", f"MODO_PRUEBA = {modo_prueba}", fuente, count=1)
            fuente = re.sub(r"FORZAR = \w+", f"FORZAR = {forzar}", fuente, count=1)
            for k, v in (reemplazos or {}).items():
                fuente, c = re.subn(rf"^{k} = [^\s#]+", f"{k} = {v}", fuente, flags=re.M)
                assert c == 1, f"no pude reemplazar {k}"
        try:
            exec(compile(fuente, f"<celda03 {n}>", "exec"), ns)
        except (RuntimeError, FileNotFoundError) as e:
            if esperar_error and esperar_error in str(e):
                return ns, str(e)
            raise
    return ns, None


def verificar03(ns, modo_prueba, recall_minimo=None):
    S = ns["SALIDA"]
    for nombre in ARCHIVOS + ["isic_pi_modelo.zip"]:
        assert (S / nombre).exists() and (S / nombre).stat().st_size > 0, f"falta {nombre}"
    m = json.loads((S / "metricas_test.json").read_text())
    assert m["modo_prueba"] == modo_prueba and m["clases"] == ns["CLASES"]
    assert m["intérprete"] == "ai_edge_litert", m["intérprete"]
    f32, lig = m["modelos"]["float32"], m["modelos"]["ligero"]
    assert f32["dif_max_prob"] < 1e-4, f"float32 no coincide con Keras: {f32['dif_max_prob']}"
    assert f32["coincide_con_keras"] == 1.0
    assert lig["mb"] < f32["mb"] / 2, (lig["mb"], f32["mb"])
    reporte = (S / "reporte_test.txt").read_text()
    assert all(c in reporte for c in ns["CLASES"])
    with zipfile.ZipFile(S / "isic_pi_modelo.zip") as z:
        assert sorted(z.namelist()) == sorted(ARCHIVOS), z.namelist()
    # las muestras se reproducen con el intérprete de la Pi leyendo el archivo, como hará la sesión 4
    mu = np.load(S / "muestras_pi.npz")
    assert mu["x"].dtype == np.uint8 and mu["x"].shape[1:] == (224, 224, 3)
    assert len(mu["x"]) == len(mu["y"]) == len(mu["prob_float32"]) == len(mu["prob_ligero"]) > 0
    it = Interpreter(model_path=str(S / "modelo.tflite"))
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    assert ent["dtype"] == np.float32 and tuple(ent["shape"]) == (1, 224, 224, 3), ent
    assert tuple(sal["shape"]) == (1, len(ns["CLASES"])), sal
    for x, esperado in zip(mu["x"], mu["prob_float32"]):
        it.set_tensor(ent["index"], x[None].astype(np.float32))
        it.invoke()
        p = it.get_tensor(sal["index"])[0]
        assert np.allclose(p, esperado, atol=1e-5) and abs(p.sum() - 1) < 1e-4
    if recall_minimo is not None:
        assert f32["recall_promedio"] >= recall_minimo, f32["recall_promedio"]
    print(f"  float32 {f32['mb']:.1f} MB · recall prom. {f32['recall_promedio']:.3f} · dif. máx. con Keras {f32['dif_max_prob']:.1e} | "
          f"ligero {lig['mb']:.1f} MB · recall prom. {lig['recall_promedio']:.3f} · coincide {100 * lig['coincide_con_keras']:.1f} %")
    return m


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        print("== sin datos del 01 → error claro ==")
        _, msg = correr03(True, esperar_error="Corre primero el notebook 01")
        assert msg
        print("  OK:", msg[:70])

        t2.t1.correr(True)                  # 01 en modo prueba
        print("== con 01 pero sin 02 → error claro ==")
        _, msg = correr03(True, esperar_error="Corre primero el notebook 02")
        assert msg
        print("  OK:", msg[:70])

        print("== cadena 01 → 02 → 03 en MODO_PRUEBA ==")
        t2.correr02(True)
        ns, _ = correr03(True)
        verificar03(ns, True)
        _, msg = correr03(True, esperar_error="Ya hay un modelo exportado")
        assert msg
        print("  re-ejecución sin FORZAR → frena:", msg[:55], "…")
        ns, _ = correr03(True, forzar=True)
        verificar03(ns, True)
        print("  FORZAR=True → reexporta OK")

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        print("== colores sólidos: 02 entrena, 03 mide en test y exporta ==")
        t2.fabricar_datos_colores(d)
        t2.correr02(False, reemplazos={"EPOCAS_CABEZA": 4, "LR_CABEZA": 1e-2, "EPOCAS_AJUSTE": 1,
                                       "PACIENCIA": 20, "BATCH": 32, "CAPAS_A_AJUSTAR": 20})
        ns, _ = correr03(False)
        verificar03(ns, False, recall_minimo=0.9)
    print("TODO OK")
