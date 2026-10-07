"""Versión local del notebook 01, para entrenar sin Colab (por ejemplo, en la Mac) con poco disco.

Baja los parquet de Atoany/ISIC2019 uno por uno, reduce cada imagen con preparar_imagen (la de
pi/clasificar.py, idéntica a la del notebook 01 según tests/probar_clasificar.py) y borra el parquet.
En disco nunca hay más de un parquet (~0.5 GB) además de los resultados. Si se interrumpe, al volver a
correrlo retoma desde el último archivo terminado.

Deja los mismos archivos que el notebook 01 (mismo orden de imágenes, mismas clases, mismo split con la
misma semilla) en <destino>/isic-pi/datos/, que es donde los busca local/correr_notebook.py.

Uso (Python 3.12 con datasets, scikit-learn, pillow y numpy):
    python local/preparar_datos.py --destino ~/isic-pi-datos
"""
import argparse
import importlib.util
import io
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
DATASET = "Atoany/ISIC2019"
TAMANO = 224
FRAC_VALIDACION = 0.10
SEMILLA = 42

spec = importlib.util.spec_from_file_location("clasificar", RAIZ / "pi" / "clasificar.py")
clasificar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clasificar)


def preparar_bytes(b):
    return clasificar.preparar_imagen(Image.open(io.BytesIO(b)), TAMANO)


def bajar(nombre, carpeta, intentos=4):
    for k in range(intentos):
        try:
            return Path(hf_hub_download(DATASET, nombre, repo_type="dataset", local_dir=carpeta))
        except Exception as e:                       # el WiFi se cae: reintentar
            print(f"    falló la descarga ({type(e).__name__}); reintento {k + 1}/{intentos - 1}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"no pude bajar {nombre}")


def procesar(nombre, carpeta_tmp, parciales, hilos):
    """Un parquet → parciales/<nombre>.npz con x (uint8) y etiquetas. Si ya existe, no lo repite."""
    salida = parciales / (Path(nombre).stem + ".npz")
    if salida.exists():
        return salida
    t0 = time.time()
    ruta = bajar(nombre, carpeta_tmp)
    t1 = time.time()
    xs, etiquetas = [], []
    with ThreadPoolExecutor(hilos) as pool:
        for lote in pq.ParquetFile(ruta).iter_batches(batch_size=256, columns=["image", "label"]):
            imagenes = lote.column("image").field("bytes").to_pylist()
            xs.append(np.stack(list(pool.map(preparar_bytes, imagenes))))
            etiquetas += lote.column("label").to_pylist()
    np.savez(salida.with_suffix(".tmp.npz"), x=np.concatenate(xs), etiquetas=np.array(etiquetas))
    salida.with_suffix(".tmp.npz").rename(salida)                     # solo existe si terminó completo
    ruta.unlink()
    print(f"  {nombre}: {len(etiquetas)} imágenes · descarga {t1 - t0:.0f} s · proceso {time.time() - t1:.0f} s", flush=True)
    return salida


def juntar(rutas):
    partes = [np.load(r) for r in rutas]
    return np.concatenate([p["x"] for p in partes]), [str(e) for p in partes for e in p["etiquetas"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", type=Path, default=Path.home() / "isic-pi-datos")
    ap.add_argument("--hilos", type=int, default=8)
    a = ap.parse_args()
    destino = a.destino.expanduser()
    datos = destino / "isic-pi" / "datos"
    parciales = destino / "parciales"
    tmp = destino / "descarga"
    for c in (datos, parciales, tmp):
        c.mkdir(parents=True, exist_ok=True)
    if (datos / "meta.json").exists():
        raise SystemExit(f"Ya hay datos preparados en {datos}. Bórralos si quieres rehacerlos.")

    archivos = sorted(f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".parquet"))
    de_train = [f for f in archivos if Path(f).name.startswith("train-")]
    de_test = [f for f in archivos if Path(f).name.startswith("test-")]
    print(f"{len(de_train)} parquet de train y {len(de_test)} de test", flush=True)
    t0 = time.time()
    rutas_train = [procesar(f, tmp, parciales, a.hilos) for f in de_train]
    rutas_test = [procesar(f, tmp, parciales, a.hilos) for f in de_test]

    X_train, etiquetas_train = juntar(rutas_train)
    X_test, etiquetas_test = juntar(rutas_test)
    CLASES = sorted(set(etiquetas_train))
    idx_clase = {c: i for i, c in enumerate(CLASES)}
    validos = [i for i, e in enumerate(etiquetas_test) if e in idx_clase]
    if len(validos) < len(etiquetas_test):
        print(f"⚠ se descartan {len(etiquetas_test) - len(validos)} imágenes de test con etiquetas que no están en train")
    X_test = X_test[validos]
    y_train = np.array([idx_clase[e] for e in etiquetas_train], dtype=np.int16)
    y_test = np.array([idx_clase[etiquetas_test[i]] for i in validos], dtype=np.int16)

    idx_tr, idx_val = train_test_split(np.arange(len(y_train)), test_size=FRAC_VALIDACION,
                                       stratify=y_train, random_state=SEMILLA)
    idx_tr, idx_val = np.sort(idx_tr), np.sort(idx_val)

    np.save(datos / "train_x.npy", X_train)
    np.save(datos / "train_y.npy", y_train)
    np.save(datos / "test_x.npy", X_test)
    np.save(datos / "test_y.npy", y_test)
    np.savez(datos / "split.npz", idx_tr=idx_tr, idx_val=idx_val)
    (datos / "clases.txt").write_text("\n".join(CLASES) + "\n")
    meta = {
        "dataset": DATASET, "modo_prueba": False, "tamano": TAMANO, "clases": CLASES,
        "n_train": int(len(y_train)), "n_val": int(len(idx_val)), "n_test": int(len(y_test)),
        "frac_validacion": FRAC_VALIDACION, "semilla": SEMILLA,
        "preprocesado": "exif_transpose, RGB, ImageOps.fit(BILINEAR, centrado), uint8",
        "fuente": "local/preparar_datos.py (parquet por parquet, fuera de Colab)",
        "pillow": Image.__version__, "numpy": np.__version__, "fecha": time.strftime("%Y-%m-%d %H:%M"),
    }
    (datos / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))   # al final: marca "terminado"
    shutil.rmtree(parciales)
    shutil.rmtree(tmp, ignore_errors=True)
    conteo = np.bincount(y_train, minlength=len(CLASES))
    print(f"\n✓ Listo en {(time.time() - t0) / 60:.0f} min: train {len(y_train):,} (val {len(idx_val):,}) · test {len(y_test):,}")
    print("  por clase en train:", ", ".join(f"{c} {n}" for c, n in zip(CLASES, conteo)))
