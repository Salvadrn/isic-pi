"""Prueba el notebook 01 sin Colab: ejecuta sus celdas de código con un dataset FALSO que tiene el mismo
esquema que Atoany/ISIC2019 (columnas `image` y `label`).

Qué comprueba: formas y tipos de los arreglos, etiquetas alineadas con las imágenes, descarte de
etiquetas que solo existen en test, imágenes no cuadradas / en grises / con canal alfa, split
estratificado sin solapes, y que el notebook se niega a rehacer datos sin FORZAR.

Qué NO comprueba: la descarga real de Hugging Face, el montaje de Drive ni el tiempo en Colab.

Uso (Python 3.12 con datasets, pandas, scikit-learn, matplotlib, tqdm, pillow, numpy):
    python tests/probar_nb01_datos.py
"""
import builtins
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

import datasets
from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "01_preparar_datos.ipynb"


def celdas_codigo(ruta):
    """Fuente de cada celda de código de un .ipynb, en orden."""
    nb = json.loads(Path(ruta).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


rng = np.random.default_rng(0)
CONTEOS = {"AK": 14, "BCC": 30, "BKL": 25, "DF": 12, "MEL": 45, "NV": 120, "SCC": 13, "VASC": 11}
TINTE = {c: rng.integers(40, 255, 3) for c in CONTEOS}


def foto(clase, i):
    """Imágenes de distintos tamaños y modos para ejercitar el recorte y la conversión."""
    formas = [(300, 300), (320, 240), (240, 320), (500, 500), (200, 150)]
    w, h = formas[i % len(formas)]
    base = rng.integers(0, 60, (h, w, 3), dtype=np.uint8) + (TINTE[clase] * 0.6).astype(np.uint8)
    img = Image.fromarray(base.astype(np.uint8), "RGB")
    if i % 11 == 0:
        img = img.convert("L")          # gris
    elif i % 13 == 0:
        img = img.convert("RGBA")       # con alfa
    return img


def hacer_split(conteos, extra_unk=0):
    imgs, labs = [], []
    k = 0
    for c, n in conteos.items():
        for _ in range(n):
            imgs.append(foto(c, k)); labs.append(c); k += 1
    for _ in range(extra_unk):
        imgs.append(foto("NV", k)); labs.append("UNK"); k += 1
    orden = rng.permutation(len(imgs))
    imgs = [imgs[i] for i in orden]
    labs = [labs[i] for i in orden]
    return Dataset.from_dict({"image": imgs, "label": labs},
                             features=Features({"image": HFImage(), "label": Value("string")}))


fake = DatasetDict({
    "train": hacer_split(CONTEOS),
    "test": hacer_split({c: max(3, n // 4) for c, n in CONTEOS.items()}, extra_unk=2),
})
datasets.load_dataset = lambda *a, **k: fake      # el notebook hace `from datasets import load_dataset`
builtins.display = print                           # en un notebook `display` ya existe


def correr(modo_prueba, forzar=False, esperar_error=None):
    """Ejecuta todas las celdas de código en un namespace nuevo (el directorio de trabajo actual hace de Drive)."""
    ns = {"__name__": "__main__"}
    for n, fuente in enumerate(celdas_codigo(NOTEBOOK)):
        fuente = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
        if "MODO_PRUEBA = " in fuente:
            fuente = re.sub(r"MODO_PRUEBA = \w+", f"MODO_PRUEBA = {modo_prueba}", fuente, count=1)
            fuente = re.sub(r"FORZAR = \w+", f"FORZAR = {forzar}", fuente, count=1)
        try:
            exec(compile(fuente, f"<celda01 {n}>", "exec"), ns)
        except RuntimeError as e:
            if esperar_error and esperar_error in str(e):
                return ns, str(e)
            raise
    return ns, None


def verificar(ns, modo_prueba):
    S = ns["SALIDA"]
    meta = json.loads((S / "meta.json").read_text())
    X = np.load(S / "train_x.npy"); y = np.load(S / "train_y.npy")
    Xt = np.load(S / "test_x.npy"); yt = np.load(S / "test_y.npy")
    sp = np.load(S / "split.npz")
    clases = (S / "clases.txt").read_text().split()
    assert clases == sorted(CONTEOS), clases
    assert X.shape == (len(y), 224, 224, 3) and X.dtype == np.uint8
    assert Xt.shape == (len(yt), 224, 224, 3)
    assert "UNK" not in clases, "UNK debe descartarse"
    assert set(sp["idx_tr"]).isdisjoint(sp["idx_val"])
    assert len(sp["idx_tr"]) + len(sp["idx_val"]) == len(y)
    assert meta["n_train"] == len(y) and meta["n_test"] == len(yt) and meta["modo_prueba"] == modo_prueba
    assert set(y[sp["idx_val"]]) == set(range(8)), "el split no es estratificado"
    # los píxeles y las etiquetas guardados coinciden con los del dataset que vio el notebook
    train_visto = ns["train_raw"]
    for i in [0, 1, 2, len(y) // 2, len(y) - 1]:
        assert np.array_equal(X[i], ns["preparar_imagen"](train_visto[i]["image"])), f"píxeles distintos en {i}"
        assert clases[y[i]] == train_visto[i]["label"], f"etiqueta distinta en {i}"
    test_visto = ns["test_raw"]
    assert len(test_visto) == len(yt)
    for i in [0, len(yt) // 2, len(yt) - 1]:
        assert np.array_equal(Xt[i], ns["preparar_imagen"](test_visto[i]["image"]))
        assert clases[yt[i]] == test_visto[i]["label"]
    return meta, len(y), len(yt)


if __name__ == "__main__":
    for modo in (False, True):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            ns, _ = correr(modo)
            meta, n, nt = verificar(ns, modo)
            print(f"[MODO_PRUEBA={modo}] OK — train {n}, test {nt}, {meta['n_val']} en validación")
            _, msg = correr(modo, esperar_error="Ya hay datos preparados")
            assert msg, "debía frenar sin FORZAR"
            print("  re-ejecución sin FORZAR → frena:", msg[:60], "…")
            ns2, _ = correr(modo, forzar=True)
            verificar(ns2, modo)
            print("  FORZAR=True → rehace OK")
    print("TODO OK")
