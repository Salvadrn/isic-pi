"""Prueba el notebook 02 sin Colab ni GPU: primero fabrica los datos con el propio notebook 01 (dataset falso)
y luego ejecuta las celdas del 02 sobre ellos.

Qué comprueba: que class_weight se aplica con datos de tf.data, que se guardan todos los archivos, que el
modelo exportado no lleva capas de aumento y recibe píxeles 0–255, que el modelo guardado reproduce las
predicciones, que el test no se toca, que las guardas (sin datos / sin GPU / ya entrenado) fallan con un
mensaje claro y que el modelo aprende algo en datos falsos fáciles de separar (separables por color).

Qué NO comprueba: calidad real con ISIC (eso es el entrenamiento en Colab), velocidad en GPU ni Drive.
La falta de GPU de esta máquina se disimula para pasar la guarda del notebook; todo corre en CPU.

Uso (Python 3.12 con lo del test 01 + tensorflow):
    python tests/probar_nb02_entrenar.py
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import probar_nb01_datos as t1          # arma el dataset falso y parchea datasets.load_dataset
from probar_nb01_datos import celdas_codigo

import keras
import tensorflow as tf

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "02_entrenar.ipynb"
FALSA_GPU = [types.SimpleNamespace(name="/physical_device:GPU:0")]


def correr02(modo_prueba, forzar=False, reemplazos=None, esperar_error=None, gpu=True):
    ns = {"__name__": "__main__"}
    shutil.rmtree(Path.cwd() / "trabajo", ignore_errors=True)    # runtime nuevo: sin copias locales de corridas previas
    original = tf.config.list_physical_devices
    try:
        for n, fuente in enumerate(celdas_codigo(NOTEBOOK)):
            if "MODO_PRUEBA = " in fuente:
                fuente = re.sub(r"MODO_PRUEBA = \w+", f"MODO_PRUEBA = {modo_prueba}", fuente, count=1)
                fuente = re.sub(r"FORZAR = \w+", f"FORZAR = {forzar}", fuente, count=1)
                for k, v in (reemplazos or {}).items():
                    fuente, c = re.subn(rf"^{k} = [^\s#]+", f"{k} = {v}", fuente, flags=re.M)
                    assert c == 1, f"no pude reemplazar {k}"
            if gpu and "list_physical_devices" in fuente:       # la GPU falsa solo existe para la celda que la revisa
                tf.config.list_physical_devices = lambda *a, **k: FALSA_GPU
            try:
                exec(compile(fuente, f"<celda02 {n}>", "exec"), ns)
            except (RuntimeError, FileNotFoundError) as e:
                if esperar_error and esperar_error in str(e):
                    return ns, str(e)
                raise
            finally:
                tf.config.list_physical_devices = original
    finally:
        tf.config.list_physical_devices = original
    return ns, None


def micro_test_class_weight():
    """¿Esta versión de Keras aplica class_weight cuando los datos vienen de un tf.data con from_generator?"""
    x = np.random.default_rng(0).normal(size=(32, 4)).astype("float32")
    y = (np.arange(32) % 2).astype("int32")

    def gen():
        yield x, y

    firma = (tf.TensorSpec((None, 4), tf.float32), tf.TensorSpec((None,), tf.int32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=firma)
    perdidas = []
    for cw in (None, {0: 1.0, 1: 50.0}):
        keras.utils.set_random_seed(1)
        m = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2, activation="softmax")])
        m.compile(optimizer=keras.optimizers.SGD(0.0), loss=keras.losses.SparseCategoricalCrossentropy())
        h = m.fit(ds, epochs=1, class_weight=cw, verbose=0)
        perdidas.append(h.history["loss"][0])
    print(f"  class_weight con tf.data: pérdida sin pesos {perdidas[0]:.3f} | con pesos {perdidas[1]:.3f}")
    assert perdidas[1] > perdidas[0] * 5, "class_weight NO se está aplicando con tf.data"


def micro_test_pesos_con_flags():
    """Guardar pesos con las capas congeladas y cargarlos con otras capas descongeladas debe dejar el modelo
    idéntico. (Por eso el notebook guarda los mejores pesos en archivo y no con get_weights/set_weights.)"""
    base = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=None)
    base.trainable = False
    entrada = keras.Input((96, 96, 3))
    x = base(entrada, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    m = keras.Model(entrada, keras.layers.Dense(3, activation="softmax")(x))

    rng = np.random.default_rng(0)
    for v in m.weights:       # valores distintos en cada vector (gamma, beta, media y varianza de BatchNorm)
        forma = tuple(v.shape)
        v.assign(rng.normal(scale=0.05, size=forma) if len(forma) > 1 else rng.uniform(0.5, 1.5, size=forma))
    imgs = rng.uniform(0, 255, (4, 96, 96, 3)).astype("float32")
    p0 = m.predict(imgs, verbose=0)
    assert np.isfinite(p0).all()
    orden_antes = [getattr(v, "path", v.name) for v in m.weights]

    with tempfile.TemporaryDirectory() as d:
        ruta = str(Path(d) / "a.weights.h5")
        m.save_weights(ruta)
        base.trainable = True                                  # lo mismo que hace la fase 2 del notebook
        for capa in base.layers[:-20]:
            capa.trainable = False
        for capa in base.layers:
            if isinstance(capa, keras.layers.BatchNormalization):
                capa.trainable = False
        orden_despues = [getattr(v, "path", v.name) for v in m.weights]
        for v in m.weights:                                    # se destruyen los pesos
            v.assign(np.zeros(tuple(v.shape), "float32"))
        m.load_weights(ruta)
    p1 = m.predict(imgs, verbose=0)
    assert np.allclose(p0, p1, atol=1e-5), "cargar pesos tras cambiar los flags NO reproduce el modelo"
    cambio = orden_antes != orden_despues
    print("  save/load_weights robusto a congelar-descongelar: OK | "
          f"en esta versión de Keras el orden de get_weights {'SÍ' if cambio else 'no'} cambia con los flags")


def fabricar_datos_colores(base, por_clase=40):
    """Datos con el formato que deja el notebook 01, pero con 8 clases de colores sólidos distintos (más ruido).
    Un pipeline sano debe aprender esto casi perfecto; si no, hay un bug en el entrenamiento."""
    import colorsys
    clases = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
    rng = np.random.default_rng(7)
    destino = Path(base) / "isic-pi" / "datos"
    destino.mkdir(parents=True, exist_ok=True)

    def lote(n):
        x, y = [], []
        for i, _ in enumerate(clases):
            color = np.array(colorsys.hsv_to_rgb(i / len(clases), 0.9, 0.9)) * 255
            ruido = rng.normal(0, 20, (n, 224, 224, 3))
            x.append(np.clip(color + ruido, 0, 255).astype("uint8")); y += [i] * n
        return np.concatenate(x), np.array(y, dtype=np.int16)

    xtr, ytr = lote(por_clase)
    xte, yte = lote(4)
    idx_val = np.concatenate([np.flatnonzero(ytr == c)[:4] for c in range(len(clases))])
    idx_tr = np.setdiff1d(np.arange(len(ytr)), idx_val)
    np.save(destino / "train_x.npy", xtr); np.save(destino / "train_y.npy", ytr)
    np.save(destino / "test_x.npy", xte); np.save(destino / "test_y.npy", yte)
    np.savez(destino / "split.npz", idx_tr=idx_tr, idx_val=np.sort(idx_val))
    (destino / "clases.txt").write_text("\n".join(clases) + "\n")
    (destino / "meta.json").write_text(json.dumps({"clases": clases, "tamano": 224, "fecha": "prueba-colores"}))


def verificar_generador(ns):
    """Las imágenes que llegan a Keras deben corresponder a sus etiquetas (en validación y en train barajado)."""
    X, y, idx_val, idx_tr = ns["X"], ns["y"], ns["idx_val"], ns["idx_tr"]
    # validación: sin barajar, mismo orden y mismos píxeles que el arreglo en disco
    xs, ys = [], []
    for a, b in ns["ds_val"]:
        xs.append(a.numpy()); ys.append(b.numpy())
    assert np.array_equal(np.concatenate(ys), y[idx_val]) and np.array_equal(np.concatenate(ys), ns["y_val"])
    assert np.array_equal(np.concatenate(xs), X[idx_val].astype("float32"))
    # train barajado: cada imagen llega con la etiqueta que le toca (se identifica por una huella de sus píxeles)
    huella = lambda arr: [hashlib.blake2b(f.astype("uint8").tobytes(), digest_size=8).digest() for f in arr]
    esperado = dict(zip(huella(X[idx_tr]), y[idx_tr].tolist()))
    assert len(esperado) == len(idx_tr), "huellas repetidas: el test no es concluyente"
    vistos = 0
    for a, b in ns["ds_tr"]:
        for f, etiqueta in zip(huella(a.numpy()), b.numpy().tolist()):
            assert esperado[f] == etiqueta, "imagen y etiqueta desalineadas en el generador de train"
            vistos += 1
    assert vistos == len(idx_tr), (vistos, len(idx_tr))
    print(f"  generador OK: {len(idx_val)} imágenes de val en orden y {vistos} de train barajadas, todas con su etiqueta")


def verificar_salida(ns, modo_prueba, desbalanceado=True):
    S, T = ns["SALIDA"], ns["TRABAJO"]
    for nombre in ["modelo.keras", "clases.txt", "reporte_val.txt", "curvas.png", "confusion_val.png", "entrenamiento.json"]:
        assert (S / nombre).exists() and (S / nombre).stat().st_size > 0, f"falta {nombre}"
    info = json.loads((S / "entrenamiento.json").read_text())
    assert info["clases"] == ns["CLASES"] and info["modo_prueba"] == modo_prueba
    assert set(info["recall_por_clase_val"]) == set(ns["CLASES"])
    assert len(info["historial"]) >= 2 and {h["fase"] for h in info["historial"]} <= {"cabeza", "ajuste"}
    assert all("val_recall_macro" in h and "val_loss" in h and "loss" in h for h in info["historial"])
    assert not (T / "test_x.npy").exists(), "test_x.npy no debe copiarse en el notebook 02"
    m = keras.models.load_model(S / "modelo.keras", compile=False)
    nombres = [l.name for l in m.layers]
    assert "escalado" in nombres and not any("random" in n for n in nombres), nombres
    assert m.input_shape == (None, 224, 224, 3) and m.output_shape == (None, len(ns["CLASES"]))
    p = m.predict(np.full((2, 224, 224, 3), 128, "float32"), verbose=0)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-4)
    if desbalanceado:
        cw = info["pesos_por_clase"]
        assert cw["DF"] > cw["NV"], cw
    return info


if __name__ == "__main__":
    print("== micro-test de class_weight + tf.data ==")
    micro_test_class_weight()
    print("== micro-test de pesos guardados con flags distintos ==")
    micro_test_pesos_con_flags()

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)

        print("== sin datos del 01 → error claro ==")
        _, msg = correr02(True, esperar_error="Corre primero el notebook 01")
        assert msg
        print("  OK:", msg[:70])

        t1.correr(True)     # → isic-pi/datos_prueba
        t1.correr(False)    # → isic-pi/datos

        print("== MODO_PRUEBA (1 época por fase) ==")
        ns, _ = correr02(True)
        verificar_generador(ns)
        info = verificar_salida(ns, True)
        print(f"  OK — fase elegida {info['fase_elegida']}, recall prom. val {info['recall_promedio_val']:.3f}, "
              f"{len(info['historial'])} épocas")
        _, msg = correr02(True, esperar_error="Ya hay un modelo entrenado")
        assert msg
        print("  re-ejecución sin FORZAR → frena:", msg[:55], "…")
        correr02(True, forzar=True)
        print("  FORZAR=True → reentrena OK")

        print("== sin GPU y sin MODO_PRUEBA → error claro ==")
        _, msg = correr02(False, esperar_error="No hay GPU", gpu=False)
        assert msg
        print("  OK:", msg[:60])

        print("== corrida larga con los datos falsos del 01 (ruido con tinte; tarea difícil), 15+3 épocas ==")
        ns, _ = correr02(False, forzar=True, reemplazos={"EPOCAS_CABEZA": 15, "EPOCAS_AJUSTE": 3, "PACIENCIA": 15,
                                                         "BATCH": 32, "CAPAS_A_AJUSTAR": 20})
        info = verificar_salida(ns, False)
        cabeza = [h for h in info["historial"] if h["fase"] == "cabeza"]
        print(f"  pérdida de train {cabeza[0]['loss']:.2f} → {cabeza[-1]['loss']:.2f} | "
              f"recall promedio en val {info['recall_promedio_val']:.3f} (azar con 8 clases = 0.125)")
        assert cabeza[-1]["loss"] < cabeza[0]["loss"], "la pérdida de train no bajó"
        assert info["recall_promedio_val"] > 0.2, "no supera al azar"

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        fabricar_datos_colores(d)

        print("== 8 clases de colores sólidos, configuración por defecto, 8+1 épocas: debe aprender ==")
        ns, _ = correr02(False, reemplazos={"EPOCAS_CABEZA": 8, "EPOCAS_AJUSTE": 1, "PACIENCIA": 20,
                                            "BATCH": 32, "CAPAS_A_AJUSTAR": 20})
        info = verificar_salida(ns, False, desbalanceado=False)
        print(f"  recall promedio por época: {[round(h['val_recall_macro'], 2) for h in info['historial']]}")
        assert info["recall_promedio_val"] >= 0.75, f"debía aprender colores sólidos: {info['recall_promedio_val']}"

        print("== ajuste fino que destruye el modelo (lr 1.0) → debe quedarse con la fase 1 ==")
        ns, _ = correr02(False, forzar=True, reemplazos={"EPOCAS_CABEZA": 6, "LR_CABEZA": 1e-2, "EPOCAS_AJUSTE": 2,
                                                         "LR_AJUSTE": 1.0, "PACIENCIA": 20, "BATCH": 32,
                                                         "CAPAS_A_AJUSTAR": 20})
        info = verificar_salida(ns, False, desbalanceado=False)
        mejor_cabeza = max(h["val_recall_macro"] for h in info["historial"] if h["fase"] == "cabeza")
        print(f"  fase elegida: {info['fase_elegida']} | mejor recall fase 1: {mejor_cabeza:.3f} | "
              f"recall final guardado: {info['recall_promedio_val']:.3f}")
        assert info["fase_elegida"] == "cabeza", "el ajuste fino destruyó el modelo y aun así se eligió"
        assert abs(info["recall_promedio_val"] - mejor_cabeza) < 1e-6, "no se restauraron los pesos de la fase 1"
    print("TODO OK")
