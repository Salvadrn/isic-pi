"""Genera demo/objetos/: MobileNetV2 preentrenada en ImageNet (1000 objetos comunes: taza, teclado, plátano…)
con la misma entrada que el modelo del proyecto: píxeles 0–255, y el escalado a [-1, 1] dentro del modelo.

Sirve para probar la cámara y el programa mientras no existe el modelo real: le enseñas una taza a la webcam
y debe decir algo como «coffee_mug». No sabe nada de piel. No se entrena: usa los pesos públicos de ImageNet.

Necesita el entorno de las pruebas (Python 3.12 con TensorFlow y ai-edge-litert; ver el README):
    python demo/generar_modelo_objetos.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras                                    # noqa: E402
import numpy as np                              # noqa: E402
import tensorflow as tf                         # noqa: E402
from ai_edge_litert.interpreter import Interpreter   # noqa: E402
from keras import layers                        # noqa: E402
from PIL import Image                           # noqa: E402

DESTINO = Path(__file__).resolve().parent / "objetos"
INDICE = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)

    entrada = keras.Input(shape=(224, 224, 3), name="imagen")
    x = layers.Rescaling(scale=1 / 127.5, offset=-1.0, name="escalado")(entrada)
    salida = keras.applications.MobileNetV2(weights="imagenet", include_top=True)(x)
    modelo = keras.Model(entrada, salida, name="mobilenetv2_imagenet")
    contenido = tf.lite.TFLiteConverter.from_keras_model(modelo).convert()
    (DESTINO / "modelo.tflite").write_bytes(contenido)

    indice = json.loads(Path(keras.utils.get_file("imagenet_class_index.json", INDICE)).read_text())
    clases = [indice[str(i)][1] for i in range(1000)]
    assert all(c and " " not in c for c in clases), "clases.txt se lee separando por espacios"
    (DESTINO / "clases.txt").write_text("\n".join(clases) + "\n")

    # Muestras para --autoprueba: imágenes sintéticas suaves y lo que da este intérprete, para comparar con la Pi
    rng = np.random.default_rng(0)
    xs = np.stack([
        np.asarray(Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize((224, 224), Image.BILINEAR))
        for _ in range(8)
    ])
    it = Interpreter(model_content=contenido)
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    probs = []
    for x in xs:
        it.set_tensor(ent["index"], x[None].astype(np.float32))
        it.invoke()
        probs.append(it.get_tensor(sal["index"])[0].copy())
    probs = np.array(probs, dtype=np.float32)
    np.savez_compressed(DESTINO / "muestras_pi.npz", x=xs, y=probs.argmax(axis=1), prob_float32=probs)
    print(f"demo/objetos listo: {len(contenido) / 1e6:.1f} MB, {len(clases)} clases, {len(xs)} muestras")
