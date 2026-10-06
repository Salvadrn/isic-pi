"""Prueba pi/clasificar.py sin cámara ni modelo real. Corre en la Mac y en la Pi: solo necesita ai-edge-litert,
NumPy y Pillow (OpenCV y rpicam-still se simulan).

Qué comprueba: que preparar_imagen es idéntica píxel a píxel a la del notebook 01 (foto grande de cámara, no
cuadradas, gris, con alfa y con rotación EXIF); la autoprueba con el modelo de colores; clasificar archivos y
carpetas; la foto con webcam y la ventana en vivo (webcam simulada); la foto con la cámara de la Pi (rpicam-still
simulado); y los mensajes de error.

Qué NO comprueba: una webcam o una cámara de verdad, ni el modelo real de lesiones.

Uso:
    python tests/probar_clasificar.py
"""
import ast
import colorsys
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent.parent
DEMO = RAIZ / "demo" / "colores"

spec = importlib.util.spec_from_file_location("clasificar", RAIZ / "pi" / "clasificar.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
rng = np.random.default_rng(3)


def correr(*args):
    """Ejecuta main() capturando la salida. Devuelve (código, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            codigo = cl.main([str(a) for a in args])
        except SystemExit as e:
            codigo = e.code
    return codigo, out.getvalue(), err.getvalue()


def color_rgb(tono):
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb(tono / 360, 0.9, 0.9))


def imagen_color(tono, alto=480, ancho=640):
    base = np.array(color_rgb(tono), dtype=np.float64)
    return np.clip(base + rng.normal(0, 20, (alto, ancho, 3)), 0, 255).astype(np.uint8)


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


def falso_cv2(tono, teclas=()):
    """Un cv2 mínimo: una webcam que entrega cuadros de un color, y ventanas que no se dibujan."""
    m = types.ModuleType("cv2")
    m.CAP_PROP_FRAME_WIDTH, m.CAP_PROP_FRAME_HEIGHT, m.FONT_HERSHEY_SIMPLEX, m.LINE_AA = 3, 4, 0, 16
    m.textos, m.mostrados, m.teclas = [], 0, list(teclas)

    class Cam:
        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def read(self):
            return True, np.ascontiguousarray(imagen_color(tono, 720, 1280)[:, :, ::-1])    # BGR

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

        print("== autoprueba con el modelo de colores ==")
        codigo, out, _ = correr("--autoprueba", "--modelo", DEMO)
        assert codigo == 0, out
        print("  " + out.strip().splitlines()[1].strip())

        print("== archivos y carpetas ==")
        fotos = d / "fotos"
        (fotos / "sub").mkdir(parents=True)
        Image.fromarray(imagen_color(0)).save(fotos / "rojo.png")
        Image.fromarray(imagen_color(225)).save(fotos / "azul.jpg", quality=95)
        Image.fromarray(imagen_color(120, 640, 480)).save(fotos / "sub" / "verde.png")
        codigo, out, _ = correr(fotos, "--modelo", DEMO, "--json")
        assert codigo == 0
        top = {Path(r["origen"]).stem: r["top"][0]["clase"] for r in json.loads(out)}
        assert top == {"rojo": "rojo", "azul": "azul", "verde": "verde"}, top
        print(f"  3 fotos (una en subcarpeta) → {top}")
        codigo, out, _ = correr(fotos / "rojo.png", "--modelo", DEMO)
        assert codigo == 0 and "rojo" in out and "ms" in out
        print("  salida de texto OK")

        print("== webcam simulada ==")
        sys.modules["cv2"] = falso_cv2(225)
        codigo, out, err = correr("--webcam", "--modelo", DEMO, "--json")
        assert codigo == 0, err
        r = json.loads(out)[0]
        assert r["top"][0]["clase"] == "azul" and Path(r["origen"]).exists(), r
        print(f"  una foto: {r['top'][0]['clase']} {100 * r['top'][0]['prob']:.0f} %, guardada en capturas/")
        cv2 = falso_cv2(0, teclas=[ord(" ")])
        sys.modules["cv2"] = cv2
        codigo, out, err = correr("--webcam", "--vivo", "--max-cuadros", 5, "--modelo", DEMO)
        assert codigo == 0, err
        assert cv2.mostrados == 5 and any(t.startswith("rojo") for t in cv2.textos), cv2.textos
        assert "MODELO DE PRUEBA" in cv2.textos and "rojo" in out
        print(f"  en vivo: {cv2.mostrados} cuadros, texto «{cv2.textos[0]}», espacio guardó una foto")
        sys.modules["cv2"] = None                         # como si OpenCV no estuviera instalado
        codigo, _, err = correr("--webcam", "--modelo", DEMO)
        assert codigo == 2 and "opencv" in err.lower(), err
        print("  sin OpenCV → mensaje claro")
        del sys.modules["cv2"]

        print("== cámara de la Pi simulada (rpicam-still falso) ==")
        bin_falso = d / "bin"
        bin_falso.mkdir()
        rpicam = bin_falso / "rpicam-still"
        rpicam.write_text(
            f"#!{sys.executable}\n"
            "import sys, numpy as np\nfrom PIL import Image\n"
            "salida = sys.argv[sys.argv.index('-o') + 1]\n"
            f"Image.fromarray(np.full((1296, 2304, 3), {color_rgb(0)}, np.uint8)).save(salida)\n")
        rpicam.chmod(0o755)
        path_original = os.environ["PATH"]
        os.environ["PATH"] = f"{bin_falso}{os.pathsep}{path_original}"
        try:
            codigo, out, err = correr("--camara", "--modelo", DEMO, "--json")
        finally:
            os.environ["PATH"] = path_original
        assert codigo == 0, err
        assert json.loads(out)[0]["top"][0]["clase"] == "rojo", out
        print("  foto con rpicam-still → rojo")

        print("== errores ==")
        casos = [
            (("--autoprueba", "--modelo", d / "no-existe"), "No encuentro"),
            ((d / "capturas" / "vacia",), None),
            (("--autoprueba", "--ligero", "--modelo", DEMO), "modelo_ligero.tflite"),
            (("--webcam", "--camara", "--modelo", DEMO), None),
        ]
        (d / "capturas" / "vacia").mkdir()
        for args, texto in casos:
            codigo, _, err = correr(*(list(args) + (["--modelo", DEMO] if args[0] == d / "capturas" / "vacia" else [])))
            assert codigo == 2, (args, codigo, err)
            assert texto is None or texto in err, (args, err)
        print(f"  {len(casos)} casos devuelven código 2 con mensaje")
    print("TODO OK")
