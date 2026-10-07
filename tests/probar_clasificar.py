"""Prueba pi/clasificar.py sin cámara ni modelo real. Corre en la Mac y en la Pi: solo necesita ai-edge-litert,
NumPy y Pillow (OpenCV y rpicam-still se simulan). Usa el modelo de objetos de demo/objetos.

Qué comprueba: que preparar_imagen es idéntica píxel a píxel a la del notebook 01 (foto grande de cámara, no
cuadradas, gris, con alfa y con rotación EXIF); la autoprueba y su tolerancia (más holgada para el ligero); y que cada modo (archivos y carpetas, foto con
webcam, ventana en vivo, foto con la cámara de la Pi) da exactamente lo mismo que el modelo aplicado directo a la
misma imagen. También, que los errores salen con mensaje y código 2.

Qué NO comprueba: una webcam o una cámara de verdad, ni el modelo real de lesiones.

Uso:
    python tests/probar_clasificar.py
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent.parent
DEMO = RAIZ / "demo" / "objetos"

spec = importlib.util.spec_from_file_location("clasificar", RAIZ / "pi" / "clasificar.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
rng = np.random.default_rng(3)
MODELO = cl.Clasificador(DEMO)


def correr(*args):
    """Ejecuta main() capturando la salida. Devuelve (código, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            codigo = cl.main([str(a) for a in args])
        except SystemExit as e:
            codigo = e.code
    return codigo, out.getvalue(), err.getvalue()


def foto_suave(alto, ancho):
    """Imagen sintética suave (ruido de baja frecuencia), más parecida a una foto que el ruido puro."""
    chica = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(chica).resize((ancho, alto), Image.BILINEAR))


def esperado(img):
    """Clase y probabilidad que da el modelo aplicado directo."""
    p, _ = MODELO.clasificar(img)
    i = int(p.argmax())
    return MODELO.clases[i], round(float(p[i]), 4)


def preparar_del_notebook():
    nb = json.loads((RAIZ / "notebooks" / "01_preparar_datos.ipynb").read_text())
    for celda in nb["cells"]:
        fuente = "".join(celda["source"])
        if celda["cell_type"] == "code" and "def preparar_imagen" in fuente:
            for nodo in ast.parse(fuente).body:
                if isinstance(nodo, ast.FunctionDef) and nodo.name == "preparar_imagen":
                    ns = {"np": np, "Image": Image, "ImageOps": ImageOps, "TAMANO": 224}
                    exec(compile(ast.Module([nodo], []), "<notebook 01>", "exec"), ns)
                    return ns["preparar_imagen"]
    raise AssertionError("no encontré preparar_imagen en el notebook 01")


def probar_paridad():
    nb_preparar = preparar_del_notebook()
    jpeg = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6                                   # orientación EXIF: girar 90°
    Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(jpeg, "JPEG", exif=exif.tobytes())
    fotos = {
        "cámara 4608×2592": lambda d=rng.integers(0, 256, (2592, 4608, 3), dtype=np.uint8): Image.fromarray(d),
        "640×480": lambda d=rng.integers(0, 256, (480, 640, 3), dtype=np.uint8): Image.fromarray(d),
        "480×640": lambda d=rng.integers(0, 256, (640, 480, 3), dtype=np.uint8): Image.fromarray(d),
        "gris": lambda d=rng.integers(0, 256, (200, 300), dtype=np.uint8): Image.fromarray(d, "L"),
        "con alfa": lambda d=rng.integers(0, 256, (500, 500, 4), dtype=np.uint8): Image.fromarray(d, "RGBA"),
        "JPEG con rotación EXIF": lambda: Image.open(io.BytesIO(jpeg.getvalue())),
    }
    assert ImageOps.exif_transpose(fotos["JPEG con rotación EXIF"]()).size == (300, 400), "el EXIF no rotó"
    for nombre, foto in fotos.items():
        a, b = nb_preparar(foto()), cl.preparar_imagen(foto(), 224)
        assert a.shape == b.shape == (224, 224, 3) and np.array_equal(a, b), f"preproceso distinto en «{nombre}»"
    print(f"  preparar_imagen idéntica al notebook 01 en {len(fotos)} casos: {', '.join(fotos)}")


def falso_cv2(cuadro_rgb, teclas=()):
    """Un cv2 mínimo: una webcam que siempre entrega el mismo cuadro, y ventanas que no se dibujan."""
    m = types.ModuleType("cv2")
    m.CAP_PROP_FRAME_WIDTH, m.CAP_PROP_FRAME_HEIGHT, m.FONT_HERSHEY_SIMPLEX, m.LINE_AA = 3, 4, 0, 16
    m.textos, m.mostrados, m.teclas = [], 0, list(teclas)
    bgr = np.ascontiguousarray(cuadro_rgb[:, :, ::-1])

    class Cam:
        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def read(self):
            return True, bgr.copy()

        def release(self):
            pass

    def imshow(nombre, img):
        m.mostrados += 1

    m.VideoCapture = lambda indice: Cam()
    m.putText = lambda img, texto, *a, **k: m.textos.append(texto)
    m.rectangle = lambda *a, **k: None
    m.imshow = imshow
    m.waitKey = lambda ms: m.teclas.pop(0) if m.teclas else -1
    m.imwrite = lambda ruta, img: Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])).save(ruta) or True
    m.destroyAllWindows = lambda: None
    return m


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cl.CAPTURAS = d / "capturas"                      # que las pruebas no dejen fotos en el repo

        print("== preproceso ==")
        probar_paridad()

        print("== autoprueba con el modelo de objetos ==")
        codigo, out, _ = correr("--autoprueba", "--modelo", DEMO)
        assert codigo == 0, out
        print("  " + out.strip().splitlines()[1].strip())

        print("== tolerancia de la autoprueba ==")
        tol = d / "modelo_tolerancia"                    # el modelo de objetos hace de float32 y de ligero
        tol.mkdir()
        shutil.copy(DEMO / "clases.txt", tol / "clases.txt")
        for nombre in ("modelo.tflite", "modelo_ligero.tflite"):
            shutil.copy(DEMO / "modelo.tflite", tol / nombre)
        mu = np.load(DEMO / "muestras_pi.npz")
        for desvio, pasa_float32, pasa_ligero in [(0.02, False, True), (0.08, False, False)]:
            prob = mu["prob_float32"].copy()
            prob[np.arange(len(prob)), prob.argmax(axis=1)] += desvio   # misma clase, probabilidad corrida
            np.savez(tol / "muestras_pi.npz", x=mu["x"], prob_float32=prob, prob_ligero=prob)
            for extra, pasa in [((), pasa_float32), (("--ligero",), pasa_ligero)]:
                codigo, out, _ = correr("--autoprueba", "--modelo", tol, *extra)
                assert codigo == (0 if pasa else 1), (desvio, extra, out)
        print("  desvío de 0.02: el float32 falla y el ligero pasa; desvío de 0.08: los dos fallan")

        print("== archivos y carpetas ==")
        fotos = d / "fotos"
        (fotos / "sub").mkdir(parents=True)
        Image.fromarray(foto_suave(480, 640)).save(fotos / "a.png")
        Image.fromarray(foto_suave(720, 1280)).save(fotos / "b.jpg", quality=95)
        Image.fromarray(foto_suave(640, 480)).save(fotos / "sub" / "c.png")
        codigo, out, _ = correr(fotos, "--modelo", DEMO, "--json")
        assert codigo == 0
        resultados = json.loads(out)
        assert len(resultados) == 3, resultados
        for r in resultados:
            with Image.open(r["origen"]) as img:
                clase, prob = esperado(img)
            assert (r["top"][0]["clase"], r["top"][0]["prob"]) == (clase, prob), (r, clase, prob)
            assert len(r["top"]) == 3 and r["top"][0]["prob"] >= r["top"][1]["prob"] >= r["top"][2]["prob"]
        print(f"  3 fotos (una en subcarpeta): mismo resultado que el modelo directo ({', '.join(r['top'][0]['clase'] for r in resultados)})")
        codigo, out, _ = correr(fotos / "a.png", "--modelo", DEMO)
        assert codigo == 0 and "ms" in out and " %" in out
        print("  salida de texto OK")

        print("== webcam simulada ==")
        cuadro = foto_suave(720, 1280)
        clase_cuadro, _ = esperado(Image.fromarray(cuadro))
        sys.modules["cv2"] = falso_cv2(cuadro)
        codigo, out, err = correr("--webcam", "--modelo", DEMO, "--json")
        assert codigo == 0, err
        r = json.loads(out)[0]
        assert r["top"][0]["clase"] == clase_cuadro and Path(r["origen"]).exists(), (r, clase_cuadro)
        print(f"  una foto: {r['top'][0]['clase']} (igual que el modelo directo), guardada en capturas/")
        cv2 = falso_cv2(cuadro, teclas=[ord(" ")])
        sys.modules["cv2"] = cv2
        codigo, out, err = correr("--webcam", "--vivo", "--max-cuadros", 5, "--modelo", DEMO)
        assert codigo == 0, err
        assert cv2.mostrados == 5, cv2.mostrados
        assert any(t.startswith(cl.sin_acentos(clase_cuadro)) for t in cv2.textos), cv2.textos
        assert "MODELO DE PRUEBA" in cv2.textos and clase_cuadro in out
        print(f"  en vivo: {cv2.mostrados} cuadros, texto «{cv2.textos[0]}», espacio guardó una foto")
        sys.modules["cv2"] = None                         # como si OpenCV no estuviera instalado
        codigo, _, err = correr("--webcam", "--modelo", DEMO)
        assert codigo == 2 and "opencv" in err.lower(), err
        print("  sin OpenCV → mensaje claro")
        del sys.modules["cv2"]

        print("== cámara de la Pi simulada (rpicam-still falso) ==")
        bin_falso = d / "bin"
        bin_falso.mkdir()
        np.save(d / "cuadro_pi.npy", foto_suave(1296, 2304))
        rpicam = bin_falso / "rpicam-still"
        rpicam.write_text(
            f"#!{sys.executable}\n"
            "import sys, numpy as np\nfrom PIL import Image\n"
            "salida = sys.argv[sys.argv.index('-o') + 1]\n"
            f"Image.fromarray(np.load({str(d / 'cuadro_pi.npy')!r})).save(salida)\n")
        rpicam.chmod(0o755)
        path_original = os.environ["PATH"]
        os.environ["PATH"] = f"{bin_falso}{os.pathsep}{path_original}"
        try:
            codigo, out, err = correr("--camara", "--modelo", DEMO, "--json")
        finally:
            os.environ["PATH"] = path_original
        assert codigo == 0, err
        r = json.loads(out)[0]
        with Image.open(r["origen"]) as img:
            assert r["top"][0]["clase"] == esperado(img)[0], r
        print(f"  foto con rpicam-still → {r['top'][0]['clase']} (igual que el modelo directo)")

        print("== errores ==")
        vacia = d / "vacia"
        vacia.mkdir()
        casos = [
            (["--autoprueba", "--modelo", d / "no-existe"], "No encuentro"),
            ([vacia, "--modelo", DEMO], "No encontré imágenes"),
            (["--autoprueba", "--ligero", "--modelo", DEMO], "modelo_ligero.tflite"),
            (["--webcam", "--camara", "--modelo", DEMO], None),
            (["--vivo", "--modelo", DEMO], None),
        ]
        for args, texto in casos:
            codigo, _, err = correr(*args)
            assert codigo == 2, (args, codigo, err)
            assert texto is None or texto in err, (args, err)
        print(f"  {len(casos)} casos devuelven código 2 con mensaje")
    print("TODO OK")
