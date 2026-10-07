"""Ejecuta las celdas de código de un notebook del proyecto fuera de Colab (por ejemplo, en la Mac).

El notebook busca sus datos en ./isic-pi/ de la carpeta desde donde lo corres, así que córrelo desde la
misma carpeta que usaste con local/preparar_datos.py. Los VARIABLE=valor reemplazan los de la celda de
configuración (cada uno tiene que existir ahí, si no, se detiene).

Uso (Python 3.12 con TensorFlow; ver el README):
    cd ~/isic-pi-datos
    python ~/Desktop/isic-pi/local/correr_notebook.py ~/Desktop/isic-pi/notebooks/02_entrenar.ipynb PERMITIR_CPU=True
"""
import builtins
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # sin ventanas: las gráficas se guardan, plt.show() no hace nada


def celdas(ruta):
    nb = json.loads(Path(ruta).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def main(argv):
    if len(argv) < 1:
        raise SystemExit(__doc__)
    ruta, cambios = argv[0], dict(a.split("=", 1) for a in argv[1:])
    builtins.display = print                # en un notebook `display` ya existe
    ns = {"__name__": "__main__"}
    pendientes = set(cambios)
    for n, fuente in enumerate(celdas(ruta)):
        fuente = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
        for var, valor in cambios.items():
            fuente, cuantas = re.subn(rf"^{var} = [^\s#]+", f"{var} = {valor}", fuente, flags=re.M)
            if cuantas:
                pendientes.discard(var)
        if n == 0 and pendientes:
            raise SystemExit(f"No encontré en la configuración: {', '.join(sorted(pendientes))}")
        t0 = time.time()
        print(f"\n### celda {n + 1} ###", flush=True)
        exec(compile(fuente, f"<{Path(ruta).name} celda {n + 1}>", "exec"), ns)
        print(f"### celda {n + 1} terminó en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
