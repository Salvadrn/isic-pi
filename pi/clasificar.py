#!/usr/bin/env python3
"""Clasifica fotos de lesiones de piel con el modelo TFLite del proyecto isic-pi.

Corre igual en la Raspberry Pi y en la Mac. Desde la carpeta del repo:

  python pi/clasificar.py foto.jpg otra.jpg carpeta/   # clasifica archivos
  python pi/clasificar.py --webcam                     # una foto con la webcam (la de la Mac o una USB)
  python pi/clasificar.py --webcam --vivo              # ventana en vivo: espacio guarda la foto, q sale
  python pi/clasificar.py --camara                     # una foto con la cámara de la Pi (rpicam-still)
  python pi/clasificar.py --autoprueba                 # comprueba que da lo mismo que al exportar

`--modelo demo/objetos` usa un modelo de objetos comunes (ImageNet) para probar la cámara sin el modelo de piel.

Proyecto escolar: no es un dispositivo médico y no sirve para diagnosticar.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent.parent
MODELO_DEFECTO = RAIZ / "modelo"
CAPTURAS = RAIZ / "capturas"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
UMBRAL_CONFIANZA = 0.5
NOMBRES = {
    "AK": "Queratosis actínica", "BCC": "Carcinoma basocelular", "BKL": "Queratosis benigna",
    "DF": "Dermatofibroma", "MEL": "Melanoma", "NV": "Nevo melanocítico",
    "SCC": "Carcinoma escamoso", "VASC": "Lesión vascular",
}


class ErrorUsuario(Exception):
    """Error que se le explica al usuario sin traceback."""


def preparar_imagen(img, tamano):
    """EXIF → RGB → recorte central cuadrado → resize bilineal → uint8 (tamano, tamano, 3).
    Tiene que ser idéntica a la de notebooks/01_preparar_datos.ipynb; tests/probar_clasificar.py lo comprueba."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(img, (tamano, tamano), method=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def sin_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def cargar_interprete():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite-runtime"
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow"
    except ImportError:
        raise ErrorUsuario("Falta el intérprete de TFLite. Instálalo con: pip install ai-edge-litert") from None


class Clasificador:
    def __init__(self, carpeta, ligero=False, hilos=4):
        self.carpeta = Path(carpeta)
        self.ligero = ligero
        self.ruta = self.carpeta / ("modelo_ligero.tflite" if ligero else "modelo.tflite")
        if not self.ruta.exists():
            raise ErrorUsuario(
                f"No encuentro {self.ruta}.\n"
                "El modelo de piel viene en modelo/ del repo: bájalo con git pull (o descomprime ahí el\n"
                "isic_pi_modelo.zip del notebook 03), o prueba la cámara con el modelo de objetos: --modelo demo/objetos")
        ruta_clases = self.carpeta / "clases.txt"
        if not ruta_clases.exists():
            raise ErrorUsuario(f"No encuentro {ruta_clases}.")
        self.clases = ruta_clases.read_text(encoding="utf-8").split()
        Interpreter, self.runtime = cargar_interprete()
        self.it = Interpreter(model_path=str(self.ruta), num_threads=hilos)
        self.it.allocate_tensors()
        self.ent = self.it.get_input_details()[0]
        self.sal = self.it.get_output_details()[0]
        if self.ent["dtype"] != np.float32:
            raise ErrorUsuario(f"El modelo espera {self.ent['dtype']}; este programa solo maneja entrada float32.")
        self.tamano = int(self.ent["shape"][1])
        if int(self.sal["shape"][-1]) != len(self.clases):
            raise ErrorUsuario(f"El modelo tiene {self.sal['shape'][-1]} salidas, pero clases.txt tiene {len(self.clases)} clases.")
        self.es_isic = any(c in NOMBRES for c in self.clases)

    def probabilidades(self, x):
        """x: uint8 (tamano, tamano, 3) ya preparado. Devuelve (probabilidades, milisegundos)."""
        self.it.set_tensor(self.ent["index"], x[None].astype(np.float32))
        t0 = time.perf_counter()
        self.it.invoke()
        ms = (time.perf_counter() - t0) * 1000
        return self.it.get_tensor(self.sal["index"])[0].astype(np.float64), ms

    def clasificar(self, img):
        return self.probabilidades(preparar_imagen(img, self.tamano))

    def nombre(self, i):
        c = self.clases[i]
        return f"{NOMBRES[c]} ({c})" if c in NOMBRES else c


def resultado(c, prob, ms, origen, top):
    orden = np.argsort(prob)[::-1][:top]
    return {
        "origen": str(origen),
        "ms": round(ms, 1),
        "top": [{"clase": c.clases[i], "nombre": c.nombre(i), "prob": round(float(prob[i]), 4)} for i in orden],
    }


def imprimir(r):
    print(r["origen"])
    for k, t in enumerate(r["top"], 1):
        print(f"  {k}. {t['nombre']:<34s} {100 * t['prob']:5.1f} %")
    aviso = "  (confianza baja)" if r["top"][0]["prob"] < UMBRAL_CONFIANZA else ""
    print(f"  {r['ms']:.1f} ms{aviso}")


def expandir(rutas):
    archivos = []
    for r in map(Path, rutas):
        if r.is_dir():
            archivos += sorted(p for p in r.rglob("*") if p.suffix.lower() in EXTENSIONES)
        elif r.exists():
            archivos.append(r)
        else:
            raise ErrorUsuario(f"No existe: {r}")
    if not archivos:
        raise ErrorUsuario("No encontré imágenes (jpg, png, bmp, webp o tif).")
    return archivos


def ruta_captura():
    CAPTURAS.mkdir(parents=True, exist_ok=True)
    return CAPTURAS / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"


def abrir_webcam(indice):
    try:
        import cv2
    except ImportError:
        raise ErrorUsuario("Para usar la webcam instala OpenCV: pip install opencv-python") from None
    cam = cv2.VideoCapture(indice)
    if not cam.isOpened():
        raise ErrorUsuario(f"No pude abrir la webcam {indice}. En la Mac revisa el permiso en "
                           "Ajustes del Sistema → Privacidad y seguridad → Cámara.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cv2, cam


def foto_webcam(indice, calentar_s=1.0):
    """Lee cuadros durante calentar_s segundos (para que ajuste exposición y color) y se queda con el último."""
    _, cam = abrir_webcam(indice)
    try:
        cuadro, fin = None, time.time() + calentar_s
        while time.time() < fin or cuadro is None:
            ok, f = cam.read()
            if ok:
                cuadro = f
            elif time.time() > fin + 5:
                raise ErrorUsuario("La webcam no entrega imágenes (¿le diste permiso de cámara?).")
    finally:
        cam.release()
    return Image.fromarray(np.ascontiguousarray(cuadro[:, :, ::-1]))     # BGR → RGB


def foto_camara_pi(destino, espera_ms=1500):
    exe = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if not exe:
        raise ErrorUsuario("No encuentro rpicam-still: esto solo funciona en la Raspberry Pi con la cámara conectada.")
    r = subprocess.run([exe, "-n", "-t", str(espera_ms), "-o", str(destino)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not Path(destino).exists():
        raise ErrorUsuario("La cámara no tomó la foto:\n" + (r.stderr or r.stdout)[-600:])
    return Image.open(destino)


def en_vivo(c, indice, max_cuadros=None):
    cv2, cam = abrir_webcam(indice)
    print("Ventana en vivo: pon la lesión dentro del cuadro. Espacio = guardar foto · q = salir")
    cuadros = fallas = 0
    try:
        while True:
            ok, cuadro = cam.read()
            if not ok:
                fallas += 1
                if fallas > 100:
                    raise ErrorUsuario("La webcam dejó de entregar imágenes.")
                time.sleep(0.05)
                continue
            h, w = cuadro.shape[:2]
            lado = min(h, w)
            x0, y0 = (w - lado) // 2, (h - lado) // 2                       # lo que ve el modelo
            prob, ms = c.clasificar(Image.fromarray(np.ascontiguousarray(cuadro[:, :, ::-1])))
            i = int(prob.argmax())
            vista = cuadro.copy()
            cv2.rectangle(vista, (x0, y0), (x0 + lado - 1, y0 + lado - 1), (255, 255, 255), 2)
            cv2.rectangle(vista, (0, h - 56), (w, h), (0, 0, 0), -1)
            color = (255, 255, 255) if prob[i] >= UMBRAL_CONFIANZA else (0, 215, 255)   # amarillo = confianza baja
            texto = f"{sin_acentos(c.nombre(i))}  {100 * prob[i]:.0f}%  ({ms:.0f} ms)"
            cv2.putText(vista, texto, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            if not c.es_isic:
                cv2.putText(vista, "MODELO DE PRUEBA", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.imshow("isic-pi", vista)
            tecla = cv2.waitKey(1) & 0xFF
            cuadros += 1
            if tecla == ord(" "):
                ruta = ruta_captura()
                cv2.imwrite(str(ruta), cuadro)
                imprimir(resultado(c, prob, ms, ruta, 3))
            if tecla in (ord("q"), 27) or (max_cuadros and cuadros >= max_cuadros):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


# El float32 da lo mismo en cualquier CPU (medido hasta 3e-6). El ligero cuantiza las activaciones a int8 al vuelo, y no
# todas las combinaciones de CPU y versión del intérprete usan las mismas rutinas: la Pi (litert 2.2.0 y 2.3.0) y la Mac
# con 2.2.0 coinciden entre sí, pero la Mac M4 que exportó el primer modelo (2.3.0) difiere de ellas hasta 1.5e-2.
TOLERANCIA = {"prob_float32": 1e-3, "prob_ligero": 5e-2}


def autoprueba(c):
    ruta = c.carpeta / "muestras_pi.npz"
    if not ruta.exists():
        raise ErrorUsuario(f"No encuentro {ruta}; viene con el modelo en modelo/ del repo (git pull) "
                           "o en el isic_pi_modelo.zip del notebook 03.")
    mu = np.load(ruta)
    clave = "prob_ligero" if c.ligero else "prob_float32"
    if clave not in mu.files:
        raise ErrorUsuario(f"{ruta.name} no trae {clave}.")
    tol = TOLERANCIA[clave]
    difs, iguales, fallas, tiempos = [], 0, 0, []
    for x, esperado in zip(mu["x"], mu[clave]):
        p, ms = c.probabilidades(x)
        difs.append(float(np.abs(p - esperado).max()))
        igual = p.argmax() == esperado.argmax()
        segunda, primera = np.sort(esperado)[-2:]
        iguales += int(igual)
        fallas += int(not igual and primera - segunda > 2 * tol)   # casi empatada: se voltea sin pasarse de tol
        tiempos.append(ms)
    ok = max(difs) < tol and fallas == 0
    print(f"Autoprueba de {c.ruta} con {len(difs)} fotos de muestra ({c.runtime}):")
    print(f"  misma clase que al exportar: {iguales}/{len(difs)} · diferencia máxima en probabilidad: {max(difs):.1e}"
          f" · {np.median(tiempos):.1f} ms por foto")
    if iguales < len(difs) and not fallas:
        print("  (las que cambian de clase estaban casi empatadas al exportar: no cuentan como error)")
    print(f"  ✓ Da lo mismo que al exportar (tolerancia {tol:.0e})." if ok else
          f"  ✗ NO coincide con lo exportado (tolerancia {tol:.0e}): revisa que el modelo y las muestras"
          " vengan del mismo zip.")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("imagenes", nargs="*", help="fotos o carpetas a clasificar")
    p.add_argument("--webcam", action="store_true", help="tomar la foto con una webcam (Mac o USB)")
    p.add_argument("--indice", type=int, default=0, help="qué webcam usar (0 = la primera)")
    p.add_argument("--vivo", action="store_true", help="con --webcam: ventana en vivo")
    p.add_argument("--camara", action="store_true", help="tomar la foto con la cámara de la Pi")
    p.add_argument("--autoprueba", action="store_true", help="comparar con las muestras del notebook 03")
    p.add_argument("--modelo", type=Path, default=MODELO_DEFECTO, help="carpeta con modelo.tflite y clases.txt")
    p.add_argument("--ligero", action="store_true", help="usar modelo_ligero.tflite")
    p.add_argument("--top", type=int, default=3, help="cuántas clases mostrar")
    p.add_argument("--hilos", type=int, default=4, help="hilos del intérprete")
    p.add_argument("--json", action="store_true", help="salida en JSON")
    p.add_argument("--max-cuadros", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if sum([bool(a.imagenes), a.webcam, a.camara, a.autoprueba]) != 1:
        p.error("elige una sola cosa: fotos, --webcam, --camara o --autoprueba")
    if a.vivo and not a.webcam:
        p.error("--vivo va junto con --webcam")

    modelo = a.modelo
    if not modelo.is_absolute() and not modelo.exists() and (RAIZ / modelo).exists():
        modelo = RAIZ / modelo                       # p. ej. --modelo demo/objetos desde otra carpeta
    try:
        c = Clasificador(modelo, ligero=a.ligero, hilos=a.hilos)
        if a.autoprueba:
            return autoprueba(c)
        if a.vivo:
            en_vivo(c, a.indice, a.max_cuadros)
            return 0
        resultados = []
        if a.imagenes:
            for ruta in expandir(a.imagenes):
                with Image.open(ruta) as img:
                    prob, ms = c.clasificar(img)
                resultados.append(resultado(c, prob, ms, ruta, a.top))
        else:
            destino = ruta_captura()
            if a.webcam:
                img = foto_webcam(a.indice)
                img.save(destino, quality=95)
            else:
                img = foto_camara_pi(destino)
            prob, ms = c.clasificar(img)
            resultados.append(resultado(c, prob, ms, destino, a.top))
        if a.json:
            print(json.dumps(resultados, ensure_ascii=False, indent=2))
        else:
            for r in resultados:
                imprimir(r)
            if c.es_isic:
                print("\nProyecto escolar: no es un diagnóstico médico.")
        return 0
    except ErrorUsuario as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
