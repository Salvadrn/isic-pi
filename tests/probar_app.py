"""Prueba pi/app.py (la app de pantalla táctil) sin cámara ni navegador. Corre en la Mac y en la Pi: solo necesita
ai-edge-litert, NumPy y Pillow; la cámara de la Pi (rpicam-vid), la webcam (OpenCV) y las fotos se simulan.

Qué comprueba:
- que el servidor sirve la página, los estilos, el script y las fuentes, y nada fuera de pi/app/;
- el estado, la imagen en vivo (MJPEG) y que «Tomar» clasifica exactamente el cuadro guía: el cuadrado central
  de GUIA × el lado corto, con el mismo resultado que el modelo aplicado directo a ese recorte;
- que solo se guarda en capturas/ lo que se pide con «Guardar», y que «Salir» solo funciona desde la propia Pi;
- la cámara de la Pi con un rpicam-vid falso (también cuando no acepta --autofocus-mode, y que se vuelve a abrir
  si rpicam-vid se cierra), la detección con un rpicam-hello falso y la webcam con un cv2 falso;
- los errores: sin imagen todavía, imagen vieja o cámara caída (no clasifica una imagen congelada), cuerpos
  raros, una carpeta sin permiso para guardar, y el modelo de objetos (no es de piel, no pide revisar).

Uso:
    python tests/probar_app.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("app", RAIZ / "pi" / "app.py")
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)
cl = app_mod.clasificar


def pedir(base, ruta, datos=None):
    """GET (o POST con JSON si hay datos). Devuelve (código, cabeceras, cuerpo)."""
    req = urllib.request.Request(base + ruta, method="POST" if datos is not None else "GET",
                                 data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def arrancar(app):
    servidor = app_mod.crear_servidor(app, 0)
    hilo = threading.Thread(target=servidor.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    hilo.start()
    return servidor, hilo, f"http://127.0.0.1:{servidor.server_address[1]}"


def esperar(condicion, segundos=10):
    fin = time.time() + segundos
    while time.time() < fin:
        if condicion():
            return True
        time.sleep(0.05)
    return False


def jpeg(arr):
    out = io.BytesIO()
    Image.fromarray(arr).save(out, "JPEG", quality=90)
    return out.getvalue()


def probar_servidor(capturas):
    print("== servidor con fotos de prueba y el modelo de piel ==")
    c = cl.Clasificador(cl.MODELO_DEFECTO)
    cam = app_mod.CamaraPrueba(app_mod.fotos_de_prueba(cl.MODELO_DEFECTO), cada=1000)   # una sola foto fija
    app = app_mod.App(cam, c, capturas)
    servidor, hilo, base = arrancar(app)
    try:
        codigo, _, _ = pedir(base, "/tomar", {})
        assert codigo == 503, "tomar sin imagen debería avisar"
        cam.start()
        assert esperar(lambda: cam.salida.n > 0), "la cámara de prueba no mandó imagen"

        for ruta, tipo in [("/", "text/html"), ("/estilo.css", "text/css"), ("/app.js", "text/javascript"),
                           ("/fuentes/azeret-mono.woff2", "font/woff2"), ("/fuentes/barlow-condensed-500.woff2", "font/woff2")]:
            codigo, cab, cuerpo = pedir(base, ruta)
            assert codigo == 200 and cab["Content-Type"].startswith(tipo) and cuerpo, (ruta, codigo)
        for ruta in ["/../pi/app.py", "/fuentes/../app.py", "/app.py", "/fuentes/OFL-azeret-mono.txt", "/%2e%2e/README.md"]:
            assert pedir(base, ruta)[0] == 404, ruta
        print("  página, estilos, script y fuentes; nada fuera de pi/app/")

        e = json.loads(pedir(base, "/estado")[2])
        assert e["guia"] == app_mod.GUIA and e["modelo"]["isic"] and e["modelo"]["clases"] == 8, e
        assert e["camara"]["nombre"] == "Fotos de prueba" and e["camara"]["error"] is None, e
        print(f"  estado: {e['camara']['nombre']}, modelo de {e['modelo']['clases']} clases, guía {e['guia']}")

        with urllib.request.urlopen(base + "/video", timeout=10) as r:
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace"), r.headers
            for _ in range(2):
                assert r.readline().strip() == b"--cuadro"
                cabeceras = {}
                while (linea := r.readline().strip()):
                    k, v = linea.decode().split(": ", 1)
                    cabeceras[k] = v
                cuadro = r.read(int(cabeceras["Content-Length"]))
                assert cabeceras["Content-Type"] == "image/jpeg" and cuadro[:2] == b"\xff\xd8", cabeceras
                r.readline()
        print("  imagen en vivo: 2 cuadros JPEG por MJPEG")

        foto = Image.open(io.BytesIO(cam.salida.jpeg)).convert("RGB")
        w, h = foto.size
        lado = round(min(w, h) * app_mod.GUIA)
        x0, y0 = (w - lado) // 2, (h - lado) // 2
        prob, _ = c.clasificar(foto.crop((x0, y0, x0 + lado, y0 + lado)))
        codigo, _, cuerpo = pedir(base, "/tomar", {})
        r = json.loads(cuerpo)
        assert codigo == 200, r
        orden = np.argsort(prob)[::-1]
        assert [t["codigo"] for t in r["top"]] == [c.clases[i] for i in orden[:3]], r["top"]
        assert all(abs(t["prob"] - round(float(prob[i]), 4)) < 1e-9 for t, i in zip(r["top"], orden)), r["top"]
        assert r["isic"] and r["revisar"] == any(t["codigo"] in app_mod.RIESGO and t["prob"] >= app_mod.UMBRAL_RIESGO
                                                 for t in r["top"]), r
        assert r["top"][0]["nombre"] == cl.NOMBRES[r["top"][0]["codigo"]] and r["top"][0]["comun"], r["top"][0]
        recorte = Image.open(io.BytesIO(pedir(base, f"/recorte/{r['id']}.jpg")[2]))
        assert recorte.size == (lado, lado), recorte.size
        print(f"  tomar: clasifica el cuadro guía de {lado}×{lado} de una imagen de {w}×{h}, igual que el modelo directo "
              f"({r['top'][0]['codigo']} {100 * r['top'][0]['prob']:.0f} %)")

        assert not any(capturas.glob("*")), "no debería guardar nada sin que lo pidan"
        codigo, _, cuerpo = pedir(base, "/guardar", {"id": r["id"]})
        assert codigo == 200, cuerpo
        guardada = json.loads((capturas / f"{r['id']}.json").read_text(encoding="utf-8"))
        assert guardada == r and (capturas / f"{r['id']}.jpg").read_bytes()[:2] == b"\xff\xd8"
        for malo in ["../../etc/passwd", "20990101-000000-000", "", 7]:
            codigo, _, cuerpo = pedir(base, "/guardar", {"id": malo})
            assert codigo == 503 and "toma otra" in json.loads(cuerpo)["error"], (malo, cuerpo)
        assert pedir(base, "/recorte/20990101-000000-000.jpg")[0] == 404
        print("  guardar: solo cuando se pide, con la foto y el resultado; identificadores raros se rechazan")

        for cuerpo in (b"[]", b"null", b"no es json"):
            con = http.client.HTTPConnection("127.0.0.1", servidor.server_address[1], timeout=10)
            con.request("POST", "/guardar", body=cuerpo, headers={"Content-Type": "application/json"})
            resp = con.getresponse()
            assert resp.status == 503 and "error" in json.loads(resp.read()), cuerpo
            con.close()
        con = http.client.HTTPConnection("127.0.0.1", servidor.server_address[1], timeout=10)
        con.putrequest("POST", "/tomar")
        con.putheader("Content-Length", "abc")
        con.endheaders()
        resp = con.getresponse()
        assert resp.status == 200 and "top" in json.loads(resp.read()), resp.status
        con.close()
        capturas.chmod(0o500)
        try:
            ident = json.loads(pedir(base, "/tomar", {})[2])["id"]
            with contextlib.redirect_stderr(io.StringIO()):       # el servidor deja el traceback en su registro
                codigo, _, cuerpo = pedir(base, "/guardar", {"id": ident})
            if os.geteuid() != 0:                    # root sí puede escribir
                assert codigo == 500 and "PermissionError" in json.loads(cuerpo)["error"], cuerpo
        finally:
            capturas.chmod(0o700)
        print("  cuerpos raros y una carpeta sin permiso dan un error en JSON, no una conexión cortada")

        cam.detener()
        cam.join(5)
        vieja = app_mod.VIEJA
        app_mod.VIEJA = 0.05
        try:
            time.sleep(0.2)
            codigo, _, cuerpo = pedir(base, "/tomar", {})
            assert codigo == 503 and "imagen nueva" in json.loads(cuerpo)["error"], cuerpo
        finally:
            app_mod.VIEJA = vieja
        cam.salida.fallar("La cámara de la Pi no manda imagen; reintentando.")
        codigo, _, cuerpo = pedir(base, "/tomar", {})
        assert codigo == 503 and "reintentando" in json.loads(cuerpo)["error"], cuerpo
        print("  con la imagen vieja o la cámara caída, «Tomar» avisa en vez de clasificar una imagen congelada")

        assert pedir(base, "/salir", {})[0] == 200
        hilo.join(5)
        assert not hilo.is_alive(), "salir no apagó el servidor"
        print("  salir desde la propia máquina apaga el servidor")
    finally:
        cam.detener()
        servidor.server_close()


def probar_modelo_de_objetos(capturas):
    print("== con el modelo de objetos (no es de piel) ==")
    demo = RAIZ / "demo" / "objetos"
    c = cl.Clasificador(demo)
    cam = app_mod.CamaraPrueba(app_mod.fotos_de_prueba(demo), cada=1000)
    app = app_mod.App(cam, c, capturas)
    cam.start()
    try:
        assert esperar(lambda: cam.salida.n > 0)
        r = app.tomar()
        assert not r["isic"] and not r["revisar"] and r["top"][0]["comun"] == "", r
        assert "_" not in r["top"][0]["nombre"], r["top"][0]
        print(f"  {r['top'][0]['nombre']}: sin nombre común y sin «revisar»")
    finally:
        cam.detener()


def probar_camara_pi(d):
    print("== cámara de la Pi simulada (rpicam-vid y rpicam-hello falsos) ==")
    bin_falso = d / "bin"
    bin_falso.mkdir()
    cuadros = [jpeg(np.full((96, 128, 3), v, np.uint8)) for v in (40, 120, 200)]
    (d / "cuadros.bin").write_bytes(b"".join(cuadros) * 3)
    vid = bin_falso / "rpicam-vid"
    vid.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "if '--autofocus-mode' in sys.argv and open(sys.argv[0] + '.af').read() == 'no':\n"
        "    sys.stderr.write('ERROR: autofocus not supported\\n'); sys.exit(1)\n"
        f"sys.stdout.buffer.write(open({str(d / 'cuadros.bin')!r}, 'rb').read()); sys.stdout.flush()\n"
        "if open(sys.argv[0] + '.cierra').read() == 'si':\n"
        "    sys.stderr.write('ERROR: camera frontend has timed out\\n'); sys.exit(1)\n"
        "time.sleep(30)\n")
    vid.chmod(0o755)
    hello = bin_falso / "rpicam-hello"
    hello.write_text("#!/bin/sh\ncat \"$0.salida\"\n")
    hello.chmod(0o755)
    path_original = os.environ["PATH"]
    os.environ["PATH"] = f"{bin_falso}{os.pathsep}{path_original}"
    (bin_falso / "rpicam-vid.cierra").write_text("no")
    try:
        for af in ("si", "no"):
            (bin_falso / "rpicam-vid.af").write_text(af)
            cam = app_mod.CamaraPi()
            cam.start()
            try:
                assert esperar(lambda: cam.salida.n >= 9), (af, cam.salida.n, cam.salida.error)
                assert cam.salida.jpeg == cuadros[-1], "el último cuadro no llegó entero"
            finally:
                cam.detener()
                cam.join(5)
            assert cam.proc.poll() is not None, "rpicam-vid sigue corriendo después de detener"
        print("  9 cuadros JPEG separados bien del flujo, con y sin enfoque automático; al detener se cierra rpicam-vid")
        (bin_falso / "rpicam-vid.af").write_text("si")
        (bin_falso / "rpicam-vid.cierra").write_text("si")
        cam = app_mod.CamaraPi()
        cam.start()
        try:
            assert esperar(lambda: cam.salida.error is not None, 5), "no avisó que rpicam-vid se cerró"
            assert "reintentando" in cam.salida.error and "timed out" in cam.salida.error, cam.salida.error
            assert esperar(lambda: cam.salida.n >= 18, 10), cam.salida.n
        finally:
            cam.detener()
            cam.join(5)
        print("  si rpicam-vid se cierra, avisa con su motivo y lo vuelve a abrir")
        (bin_falso / "rpicam-hello.salida").write_text("Available cameras\n-----------------\n0 : imx708 [4608x2592 10-bit RGGB]\n")
        assert app_mod.hay_camara_pi()
        (bin_falso / "rpicam-hello.salida").write_text("No cameras available!\n")
        assert not app_mod.hay_camara_pi()
        print("  detecta si hay cámara conectada con rpicam-hello --list-cameras")
    finally:
        os.environ["PATH"] = path_original


def probar_webcam():
    print("== webcam simulada (cv2 falso) ==")
    cuadro_bgr = np.zeros((72, 96, 3), np.uint8)
    cuadro_bgr[:, :, 2] = 200                                       # rojo en BGR
    m = types.ModuleType("cv2")
    m.CAP_PROP_FRAME_WIDTH, m.CAP_PROP_FRAME_HEIGHT, m.IMWRITE_JPEG_QUALITY = 3, 4, 1
    m.CAP_PROP_FOURCC, m.CAP_PROP_FPS = 6, 5
    m.VideoWriter_fourcc = lambda *c: 1196444237
    puestas = []

    class Cam:
        def isOpened(self):
            return True

        def set(self, *a):
            puestas.append(a)
            return True

        def read(self):
            return True, cuadro_bgr.copy()

        def release(self):
            pass

    m.VideoCapture = lambda i: Cam()
    m.imencode = lambda ext, img, params: (True, np.frombuffer(jpeg(np.ascontiguousarray(img[:, :, ::-1])), np.uint8))
    sys.modules["cv2"] = m
    try:
        cam = app_mod.CamaraUSB(0)
        cam.start()
        try:
            assert esperar(lambda: cam.salida.n >= 3), cam.salida.error
            pixel = np.asarray(Image.open(io.BytesIO(cam.salida.jpeg)).convert("RGB"))[36, 48]
            assert pixel[0] > 150 and pixel[2] < 60, pixel
            assert (m.CAP_PROP_FOURCC, 1196444237) in puestas, puestas
        finally:
            cam.detener()
        print("  pide MJPG, manda cuadros y respeta los colores (BGR → RGB)")
        sys.modules["cv2"] = None
        cam = app_mod.CamaraUSB(0)
        cam.start()
        cam.join(5)
        assert "OpenCV" in (cam.salida.error or "") and "ErrorUsuario" not in cam.salida.error, cam.salida.error
        print("  sin OpenCV → error claro en la pantalla")
    finally:
        sys.modules.pop("cv2", None)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        probar_servidor(d / "capturas")
        probar_modelo_de_objetos(d / "capturas2")
        probar_camara_pi(d)
        probar_webcam()
    print("TODO OK")
