#!/usr/bin/env python3
"""App de pantalla táctil del proyecto isic-pi: imagen en vivo, botón para tomar la foto y probabilidades.

Sirve una página local (pi/app/) que Chromium muestra a pantalla completa en la Pi (ver pi/app_kiosco.sh).
No necesita nada más que lo que ya usa pi/clasificar.py (NumPy, Pillow y el intérprete de TFLite); OpenCV solo
para una webcam USB. Desde la carpeta del repo:

  python pi/app.py                      # cámara automática: la de la Pi; si no hay, una USB; si no, fotos de prueba
  python pi/app.py --camara prueba      # sin cámara: pasa fotos de test de ISIC, para probar la app
  python pi/app.py --camara usb --indice 1
  python pi/app.py --publico            # también se abre desde otra compu de la red (no pide contraseña)

Luego abre http://127.0.0.1:8000 en el navegador.

La foto que se clasifica es exactamente el cuadro guía que se ve en pantalla: el cuadrado central que mide GUIA
veces el lado corto de la imagen. Las fotos solo se guardan (en capturas/) si se toca «Guardar».

Proyecto escolar: no es un dispositivo médico y no sirve para diagnosticar.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clasificar  # noqa: E402  (pi/clasificar.py: el mismo modelo y el mismo preproceso)

WEB = Path(__file__).resolve().parent / "app"
GUIA = 0.8                       # lado del cuadro guía / lado corto de la imagen
FPS = 15
RIESGO = {"MEL", "BCC", "SCC", "AK"}
UMBRAL_RIESGO = 0.25              # se sugiere revisar si una de estas clases llega a 25 % entre las 3 primeras
VIEJA = 2.0                       # s: una imagen más vieja que esto ya no es «la que se ve»
COMUN = {
    "AK": "Lesión precancerosa", "BCC": "Tipo de cáncer de piel", "BKL": "Lesión benigna",
    "DF": "Lesión benigna", "MEL": "Tipo de cáncer de piel", "NV": "Lunar",
    "SCC": "Tipo de cáncer de piel", "VASC": "Lesión benigna de vasos",
}
TIPOS = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2", ".svg": "image/svg+xml"}
ID_VALIDO = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{3}$")


class Fotograma:
    """La última imagen de la cámara (JPEG), compartida entre el hilo de la cámara y los clientes."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.n = 0                                   # 0 = todavía no llega ninguna imagen
        self.t = 0.0                                 # time.monotonic() de la última imagen
        self.error = None

    def poner(self, jpeg):
        with self.cond:
            self.jpeg, self.n, self.t, self.error = jpeg, self.n + 1, time.monotonic(), None
            self.cond.notify_all()

    def fallar(self, mensaje):
        with self.cond:
            self.error = mensaje
            self.cond.notify_all()

    def esperar(self, ultimo, segundos):
        """Espera una imagen distinta de la número `ultimo`. Devuelve (n, jpeg)."""
        with self.cond:
            self.cond.wait_for(lambda: self.n != ultimo, timeout=segundos)
            return self.n, self.jpeg

    def actual(self):
        """(jpeg, error, segundos desde la última imagen), leídos juntos."""
        with self.cond:
            return self.jpeg, self.error, time.monotonic() - self.t


class Camara(threading.Thread):
    nombre = "Cámara"

    def __init__(self):
        super().__init__(daemon=True)
        self.salida = Fotograma()
        self.parar = threading.Event()

    def run(self):
        try:
            self.bucle()
        except clasificar.ErrorUsuario as e:
            self.salida.fallar(str(e))
        except Exception as e:                       # el hilo no puede tumbar la app: la pantalla muestra el error
            self.salida.fallar(f"{type(e).__name__}: {e}")

    def detener(self):
        self.parar.set()


class CamaraPi(Camara):
    """Camera Module de la Pi con rpicam-vid, que manda JPEGs uno tras otro por la salida estándar.
    Si rpicam-vid se cae (cable flojo, cámara ocupada), lo vuelve a abrir cada 2 s y mientras tanto avisa."""
    nombre = "Cámara de la Pi"

    def __init__(self, ancho=1280, alto=960):
        super().__init__()
        self.ancho, self.alto = ancho, alto
        self.proc = None
        self.ultimo_error = ""

    def comando(self, enfoque):
        cmd = ["rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg", "--quality", "90", "--width", str(self.ancho),
               "--height", str(self.alto), "--framerate", str(FPS), "-o", "-"]
        return cmd + (["--autofocus-mode", "continuous"] if enfoque else [])

    def bucle(self):
        while not self.parar.is_set():
            for enfoque in (True, False):          # sin enfoque automático si la cámara no lo tiene
                t0, imagenes = time.time(), self.leer(self.comando(enfoque))
                if self.parar.is_set() or imagenes or time.time() - t0 > 5:
                    break
            if self.parar.is_set():
                return
            self.salida.fallar("La cámara de la Pi no manda imagen; reintentando. " + self.ultimo_error)
            self.parar.wait(2)

    def leer(self, cmd):
        proc = self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errores = []
        lector = threading.Thread(target=lambda: errores.extend(proc.stderr.read().decode(errors="replace").splitlines()),
                                  daemon=True)
        lector.start()
        buf, imagenes = b"", 0
        while not self.parar.is_set():
            datos = proc.stdout.read1(1 << 16)
            if not datos:
                break
            buf += datos
            while True:
                i = buf.find(b"\xff\xd8")
                j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
                if i < 0:
                    buf = buf[-1:]
                    break
                if j < 0:
                    buf = buf[i:]
                    break
                self.salida.poner(buf[i:j + 2])
                imagenes += 1
                buf = buf[j + 2:]
        self.terminar()
        lector.join(timeout=1)
        self.ultimo_error = " ".join(errores[-2:])
        return imagenes

    def terminar(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def detener(self):
        super().detener()
        self.terminar()


class CamaraUSB(Camara):
    """Webcam USB (o la de la Mac) con OpenCV. Si deja de mandar imagen, la vuelve a abrir cada 2 s."""
    nombre = "Webcam"

    def __init__(self, indice=0):
        super().__init__()
        self.indice = indice

    def bucle(self):
        try:
            import cv2
        except ImportError:
            raise clasificar.ErrorUsuario("Para una webcam USB hace falta OpenCV: .venv/bin/pip install opencv-python-headless") from None
        while not self.parar.is_set():
            self.leer(cv2)
            if not self.parar.is_set():
                self.parar.wait(2)

    def leer(self, cv2):
        cam = cv2.VideoCapture(self.indice)
        try:
            if not cam.isOpened():
                self.salida.fallar(f"No pude abrir la webcam {self.indice}; reintentando.")
                return
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # sin esto, 720p va a 5-10 fps en USB 2.0
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cam.set(cv2.CAP_PROP_FPS, FPS)
            fallas = 0
            while not self.parar.is_set():
                t0 = time.time()
                ok, cuadro = cam.read()
                if not ok:
                    fallas += 1
                    if fallas >= 10:
                        self.salida.fallar("La webcam dejó de mandar imagen; reintentando.")
                        return
                    continue
                fallas = 0
                ok, jpg = cv2.imencode(".jpg", cuadro, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    self.salida.poner(jpg.tobytes())
                time.sleep(max(0.0, 1 / FPS - (time.time() - t0)))
        finally:
            cam.release()


class CamaraPrueba(Camara):
    """Sin cámara: pasa fotos (las de test que vienen con el modelo, o las de una carpeta), una cada `cada` s."""
    nombre = "Fotos de prueba"

    def __init__(self, fotos, cada=4.0, ancho=960, alto=720):
        super().__init__()
        if not fotos:
            raise clasificar.ErrorUsuario("No hay fotos de prueba: usa --fotos CARPETA o un modelo con muestras_pi.npz.")
        self.cada = cada
        self.jpegs = [self.cuadro(f, ancho, alto) for f in fotos]

    @staticmethod
    def cuadro(foto, ancho, alto):
        """La foto, cuadrada y del alto de la imagen, centrada sobre fondo oscuro, como si la viera una cámara."""
        lienzo = Image.new("RGB", (ancho, alto), (17, 17, 17))
        lado = min(ancho, alto)
        lienzo.paste(foto.convert("RGB").resize((lado, lado), Image.BILINEAR), ((ancho - lado) // 2, (alto - lado) // 2))
        out = io.BytesIO()
        lienzo.save(out, "JPEG", quality=92)
        return out.getvalue()

    def bucle(self):
        k = 0
        while not self.parar.is_set():
            t_cambio = time.time() + self.cada
            while time.time() < t_cambio and not self.parar.is_set():
                self.salida.poner(self.jpegs[k % len(self.jpegs)])
                time.sleep(1 / FPS)
            k += 1


def abrir_foto(ruta):
    try:
        with Image.open(ruta) as im:
            return im.convert("RGB")
    except OSError as e:
        raise clasificar.ErrorUsuario(f"No pude leer {ruta}: {e}") from None


def fotos_de_prueba(carpeta_modelo, carpeta_fotos=None):
    if carpeta_fotos:
        return [abrir_foto(p) for p in clasificar.expandir([carpeta_fotos])[:200]]
    ruta = Path(carpeta_modelo) / "muestras_pi.npz"
    if not ruta.exists():
        return []
    return [Image.fromarray(x) for x in np.load(ruta)["x"]]


def hay_camara_pi():
    if not shutil.which("rpicam-vid") or not shutil.which("rpicam-hello"):
        return False
    try:
        r = subprocess.run(["rpicam-hello", "--list-cameras"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(re.search(r"^\s*\d+\s*:", r.stdout + r.stderr, re.M))


def elegir_camara(a):
    if a.camara == "pi" or (a.camara == "auto" and hay_camara_pi()):
        return CamaraPi()
    if a.camara == "usb":
        return CamaraUSB(a.indice)
    if a.camara == "auto":
        try:
            import cv2
            cam = cv2.VideoCapture(a.indice)
            abierta = cam.isOpened()
            cam.release()
            if abierta:
                return CamaraUSB(a.indice)
        except ImportError:
            pass
        print("No encontré cámara: uso fotos de prueba.", file=sys.stderr)
    return CamaraPrueba(fotos_de_prueba(a.modelo, a.fotos))


class App:
    """Lo que la página le pide al servidor: estado, imagen en vivo, tomar foto y guardarla."""

    def __init__(self, camara, clasificador, capturas):
        self.camara = camara
        self.c = clasificador
        self.capturas = Path(capturas)
        self.lock = threading.Lock()                 # el intérprete de TFLite no es para varios hilos a la vez
        self.tomadas = OrderedDict()                 # id → (recorte JPEG, resultado); solo las últimas 10
        self.ms = None

    def estado(self):
        s = self.camara.salida
        return {
            "camara": {"nombre": self.camara.nombre, "imagenes": s.n, "error": s.error},
            "modelo": {"isic": self.c.es_isic, "clases": len(self.c.clases), "entrada": self.c.tamano,
                       "archivo": self.c.ruta.name, "interprete": self.c.runtime},
            "guia": GUIA,
            "ms": None if self.ms is None else round(self.ms, 1),
        }

    def tomar(self):
        jpeg, error, edad = self.camara.salida.actual()
        if error or jpeg is None:
            raise clasificar.ErrorUsuario(error or "La cámara todavía no manda imagen.")
        if edad > VIEJA:
            raise clasificar.ErrorUsuario("La cámara no está mandando imagen nueva; espera un momento.")
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = img.size
        lado = max(1, round(min(w, h) * GUIA))
        x0, y0 = (w - lado) // 2, (h - lado) // 2
        recorte = img.crop((x0, y0, x0 + lado, y0 + lado))
        with self.lock:
            prob, ms = self.c.clasificar(recorte)
        self.ms = ms
        orden = [int(i) for i in np.argsort(prob)[::-1]]
        top = [{"codigo": self.c.clases[i],
                "nombre": clasificar.NOMBRES.get(self.c.clases[i], self.c.clases[i].replace("_", " ")),
                "comun": COMUN.get(self.c.clases[i], ""),
                "prob": round(float(prob[i]), 4)} for i in orden[:3]]
        ident = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        r = {"id": ident, "ms": round(ms, 1), "top": top, "isic": self.c.es_isic,
             "revisar": self.c.es_isic and any(t["codigo"] in RIESGO and t["prob"] >= UMBRAL_RIESGO for t in top),
             "seguro": top[0]["prob"] >= clasificar.UMBRAL_CONFIANZA,
             "todas": {self.c.clases[i]: round(float(prob[i]), 4) for i in orden}}
        out = io.BytesIO()
        recorte.save(out, "JPEG", quality=92)
        self.tomadas[ident] = (out.getvalue(), r)
        while len(self.tomadas) > 10:
            self.tomadas.popitem(last=False)
        return r

    def guardar(self, ident):
        if not ID_VALIDO.match(ident or "") or ident not in self.tomadas:
            raise clasificar.ErrorUsuario("Esa foto ya no está; toma otra.")
        jpeg, r = self.tomadas[ident]
        self.capturas.mkdir(parents=True, exist_ok=True)
        (self.capturas / f"{ident}.jpg").write_bytes(jpeg)
        (self.capturas / f"{ident}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"guardada": f"capturas/{ident}.jpg"}


def archivos_web():
    """Lo único que se sirve como archivo: lo que hay en pi/app/ (sin subcarpetas salvo fuentes/)."""
    rutas = {}
    for p in list(WEB.glob("*")) + list((WEB / "fuentes").glob("*")):
        if p.is_file() and p.suffix in TIPOS:
            rutas["/" + p.relative_to(WEB).as_posix()] = p
    rutas["/"] = WEB / "index.html"
    return rutas


def crear_manejador(app, archivos, servidor_ref):
    class Manejador(BaseHTTPRequestHandler):
        server_version = "isic-pi"

        def log_message(self, formato, *args):       # silencioso: solo errores
            pass

        def responder(self, codigo, cuerpo, tipo, extra=None):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(cuerpo)

        def json(self, datos, codigo=HTTPStatus.OK):
            self.responder(codigo, json.dumps(datos, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            ruta = self.path.split("?", 1)[0]
            if ruta in archivos:
                p = archivos[ruta]
                return self.responder(HTTPStatus.OK, p.read_bytes(), TIPOS[p.suffix])
            if ruta == "/estado":
                return self.json(app.estado())
            if ruta == "/video":
                return self.video()
            m = re.fullmatch(r"/recorte/([0-9-]+)\.jpg", ruta)
            if m and m.group(1) in app.tomadas:
                return self.responder(HTTPStatus.OK, app.tomadas[m.group(1)][0], "image/jpeg")
            self.json({"error": "No existe."}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            ruta = self.path.split("?", 1)[0]
            try:
                largo = max(0, min(int(self.headers.get("Content-Length") or 0), 10_000))
                cuerpo = json.loads(self.rfile.read(largo) or b"{}")
            except ValueError:
                cuerpo = {}
            if not isinstance(cuerpo, dict):
                cuerpo = {}
            try:
                if ruta == "/tomar":
                    return self.json(app.tomar())
                if ruta == "/guardar":
                    return self.json(app.guardar(str(cuerpo.get("id", ""))))
                if ruta == "/salir":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        return self.json({"error": "Solo desde la propia Pi."}, HTTPStatus.FORBIDDEN)
                    self.json({"saliendo": True})
                    threading.Thread(target=servidor_ref[0].shutdown, daemon=True).start()
                    return None
            except clasificar.ErrorUsuario as e:
                return self.json({"error": str(e)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as e:                   # SD llena, sin permiso, imagen dañada: que la pantalla lo diga
                traceback.print_exc()
                return self.json({"error": f"Error interno ({type(e).__name__}): {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.json({"error": "No existe."}, HTTPStatus.NOT_FOUND)

        def video(self):
            """MJPEG: el navegador lo muestra en un <img> y se actualiza solo."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=cuadro")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            ultimo = 0                               # la imagen 0 es «todavía no hay»: espera la primera de verdad
            try:
                while not app.camara.parar.is_set():
                    n, jpeg = app.camara.salida.esperar(ultimo, 2.0)
                    if jpeg is None or n == ultimo:
                        continue
                    ultimo = n
                    self.wfile.write(b"--cuadro\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Manejador


def crear_servidor(app, puerto, publico=False):
    ref = []
    servidor = ThreadingHTTPServer(("0.0.0.0" if publico else "127.0.0.1", puerto),
                                   crear_manejador(app, archivos_web(), ref))
    servidor.daemon_threads = True
    ref.append(servidor)
    return servidor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camara", choices=["auto", "pi", "usb", "prueba"], default="auto")
    p.add_argument("--indice", type=int, default=0, help="número de la webcam USB")
    p.add_argument("--fotos", help="carpeta de fotos para --camara prueba (si no, las de test del modelo)")
    p.add_argument("--modelo", default=str(clasificar.MODELO_DEFECTO), help="carpeta del modelo")
    p.add_argument("--ligero", action="store_true", help="usar modelo_ligero.tflite")
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--puerto", type=int, default=8000)
    p.add_argument("--publico", action="store_true", help="aceptar conexiones de otras compus de la red")
    a = p.parse_args(argv)
    try:
        c = clasificar.Clasificador(a.modelo, ligero=a.ligero, hilos=a.hilos)
        cam = elegir_camara(a)
    except clasificar.ErrorUsuario as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    app = App(cam, c, clasificar.CAPTURAS)
    try:
        servidor = crear_servidor(app, a.puerto, a.publico)
    except OSError as e:
        print(f"Error: no pude abrir el puerto {a.puerto} ({e.strerror}). ¿Ya hay otra app corriendo?", file=sys.stderr)
        return 2
    cam.start()
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=servidor.shutdown, daemon=True).start())
    print(f"App lista en http://127.0.0.1:{a.puerto} · {cam.nombre} · {c.ruta.name}", flush=True)
    try:
        servidor.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cam.detener()
        servidor.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
