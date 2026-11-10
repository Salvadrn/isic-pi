"""Genera las figuras del artículo a partir de los datos medidos (articulo/datos/) y de las fotos de prueba.

Uso (Python con matplotlib, NumPy y Pillow):
    python articulo/figuras/hacer_figuras.py

Salida, en articulo/figuras/: fig_sistema.pdf, fig_clases.pdf, fig_entrenamiento.pdf, fig_confusion.pdf.
fig_app.png se arma aparte, con capturas de la aplicación (ver componer_capturas al final).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "datos"
FOTOS = RAIZ / "demo" / "fotos_prueba"

for ttf in (Path.home() / "Library" / "Fonts").glob("OpenSans-*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

TINTA, SEC, TENUE, FILETE = "#0b0b0b", "#52514e", "#8a8984", "#dddcd7"
AZUL, NARANJA = "#2a78d6", "#eb6834"                     # paleta validada (dos series)
RAMPA = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # secuencial de un solo tono
plt.rcParams.update({
    "font.family": ["Open Sans", "DejaVu Sans"], "font.size": 9, "text.color": TINTA,
    "axes.edgecolor": FILETE, "axes.labelcolor": SEC, "axes.linewidth": 0.8,
    "xtick.color": SEC, "ytick.color": SEC, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
NOMBRES = {
    "AK": "Queratosis actínica", "BCC": "Carcinoma basocelular", "BKL": "Queratosis benigna",
    "DF": "Dermatofibroma", "MEL": "Melanoma", "NV": "Nevo melanocítico",
    "SCC": "Carcinoma escamoso", "VASC": "Lesión vascular",
}
ANCHO = 6.3                                              # pulgadas: el ancho del texto de la plantilla


def fig_sistema():
    """Del conjunto de datos a la pantalla: arriba lo que pasa en la laptop, abajo lo que pasa en la Raspberry Pi."""
    fig, ax = plt.subplots(figsize=(ANCHO, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def caja(x, y, titulo, detalle, w=20.5, h=13, fuerte=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=0.9,
                                    edgecolor=TINTA if fuerte else SEC, facecolor="#f4f3ef" if fuerte else "white"))
        ax.text(x + w / 2, y + h - 2.9, titulo, ha="center", va="center", fontsize=8.6, fontweight="bold")
        ax.text(x + w / 2, y + 4.3, detalle, ha="center", va="center", fontsize=7.4, color=SEC, linespacing=1.3)
        return x, y, w, h

    def flecha(a, b, texto=None, vertical=False):
        if vertical:
            p0, p1 = (a[0] + a[2] / 2, a[1]), (b[0] + b[2] / 2, b[1] + b[3])
        else:
            p0, p1 = (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=1, color=TINTA,
                                     shrinkA=1.5, shrinkB=1.5))
        if texto:
            ax.text(p0[0] + 1.2, (p0[1] + p1[1]) / 2, texto, ha="left", va="center", fontsize=7.4, color=SEC)

    xs, y1, y2 = [1, 27, 53, 79], 31, 3
    ax.text(1, 46.4, "Entrenamiento · laptop (CPU)", fontsize=8, color=SEC, va="center")
    ax.text(1, 18.4, "Uso · Raspberry Pi 5", fontsize=8, color=SEC, va="center")
    a1 = caja(xs[0], y1, "ISIC 2019", "25,331 imágenes\n8 clases")
    a2 = caja(xs[1], y1, "Preparación", "recorte cuadrado\n224 × 224 px")
    a3 = caja(xs[2], y1, "MobileNetV2", "cabeza nueva y\najuste fino (Keras)")
    a4 = caja(xs[3], y1, "Archivo .tflite", "float32\n8.9 MB", fuerte=True)
    b1 = caja(xs[0], y2, "Cámara", "imagen en vivo\ny cuadro guía")
    b2 = caja(xs[1], y2, "Preparación", "la misma función\nque al entrenar")
    b3 = caja(xs[2], y2, "LiteRT", "inferencia en CPU\n≈ 10 ms", fuerte=True)
    b4 = caja(xs[3], y2, "Pantalla táctil", "3 clases y\nsus probabilidades")
    for a, b in [(a1, a2), (a2, a3), (a3, a4), (b1, b2), (b2, b3), (b3, b4)]:
        flecha(a, b)
    # el modelo baja de la laptop a la Pi por el repositorio
    x0, x1 = a4[0] + a4[2] / 2, b3[0] + b3[2] / 2
    ax.plot([x0, x0, x1], [y1, 23.5, 23.5], color=TINTA, linewidth=1, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x1, 23.5), (x1, y2 + 13), arrowstyle="-|>", mutation_scale=9, linewidth=1,
                                 color=TINTA, shrinkA=0, shrinkB=1.5))
    ax.text((x0 + x1) / 2, 24.8, "repositorio (git)", ha="center", va="bottom", fontsize=7.4, color=SEC)
    fig.savefig(AQUI / "fig_sistema.pdf")
    plt.close(fig)


EJEMPLOS = ["AK-ISIC_0062738", "BCC-ISIC_0029546", "BKL-ISIC_0012323", "DF-ISIC_0071844",
            "MEL-ISIC_0012425", "NV-ISIC_0014977", "SCC-ISIC_0071727", "VASC-ISIC_0029439"]


def cuadrada(ruta, lado):
    im = Image.open(ruta).convert("RGB")
    m = min(im.size)
    x0, y0 = (im.width - m) // 2, (im.height - m) // 2
    return im.crop((x0, y0, x0 + m, y0 + m)).resize((lado, lado), Image.LANCZOS)


def fig_clases():
    """Un ejemplo de test por clase, con el mismo recorte cuadrado que ve la red."""
    fig, ejes = plt.subplots(2, 4, figsize=(ANCHO, 3.75))
    for ax, nombre in zip(ejes.ravel(), EJEMPLOS):
        clase = nombre.split("-")[0]
        ax.imshow(cuadrada(FOTOS / f"{nombre}.jpg", 360))
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_xlabel(f"{clase}\n{NOMBRES[clase]}", fontsize=8, color=TINTA, labelpad=4, linespacing=1.25)
    fig.subplots_adjust(wspace=0.06, hspace=0.34)
    fig.savefig(AQUI / "fig_clases.pdf", dpi=220)
    plt.close(fig)


def fig_entrenamiento():
    """Dos paneles con el mismo eje de épocas (nunca dos escalas en un panel): recall de validación y pérdidas."""
    h = json.loads((DATOS / "entrenamiento.json").read_text())["historial"]
    ep = np.arange(1, len(h) + 1)
    n1 = sum(1 for e in h if e["fase"] == "cabeza")
    rec = np.array([e["val_recall_macro"] for e in h])
    fig, (a, b) = plt.subplots(2, 1, figsize=(ANCHO, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    for ax in (a, b):
        ax.axvline(n1 + 0.5, color=TENUE, linewidth=0.8)
        ax.grid(axis="y", color=FILETE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    a.plot(ep, rec, color=TINTA, linewidth=1.8, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1)
    mejor = int(rec.argmax())
    a.plot(ep[mejor], rec[mejor], marker="o", markersize=7.5, color=AZUL, markeredgecolor="white", markeredgewidth=1.2)
    a.annotate(f"mejor época: {rec[mejor]:.3f}", (ep[mejor], rec[mejor]), xytext=(-9, 7), textcoords="offset points",
               ha="right", fontsize=8, color=TINTA)
    a.set_ylabel("Recall promedio por clase\n(validación)")
    a.set_ylim(0.3, 0.72)
    a.spines["top"].set_visible(False)
    a.text(n1 / 2 + 0.5, 0.728, "Fase 1 · solo la capa final", ha="center", va="bottom", fontsize=8, color=SEC)
    a.text(n1 + 0.5 + (len(h) - n1) / 2, 0.728, "Fase 2 · ajuste fino de las últimas 40 capas", ha="center",
           va="bottom", fontsize=8, color=SEC)
    lr = [e["learning_rate"] for e in h]
    baja = next((i for i in range(n1 + 1, len(h)) if lr[i] < lr[i - 1] * 0.75), None)
    if baja is not None:
        a.annotate("el paso de aprendizaje\nbaja a la mitad", (ep[baja], rec[baja] - 0.012), xytext=(ep[baja] + 1.3, 0.47),
                   textcoords="data", ha="center", va="top", fontsize=7.6, color=SEC, linespacing=1.25,
                   arrowprops={"arrowstyle": "-", "color": TENUE, "linewidth": 0.8, "shrinkB": 3})
    b.plot(ep, [e["loss"] for e in h], color=NARANJA, linewidth=1.8, linestyle=(0, (4, 2)), label="entrenamiento")
    b.plot(ep, [e["val_loss"] for e in h], color=AZUL, linewidth=1.8, label="validación")
    b.text(ep[-1] + 0.25, h[-1]["loss"], "entrenamiento", va="center", fontsize=8)
    b.text(ep[-1] + 0.25, h[-1]["val_loss"], "validación", va="center", fontsize=8)
    b.set_ylabel("Pérdida\n(entropía cruzada)")
    b.set_xlabel("Época")
    b.set_xticks(ep[::1])
    b.set_xlim(0.5, len(h) + 3.6)
    b.legend(loc="upper right", frameon=False, fontsize=8, handlelength=2.6)
    fig.align_ylabels((a, b))
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(AQUI / "fig_entrenamiento.pdf")
    plt.close(fig)


def fig_confusion():
    """Matriz de confusión en test, por filas (cada fila suma 100 %). Un solo tono, de claro a oscuro."""
    d = json.loads((DATOS / "confusion_test.json").read_text())
    clases, M = d["clases"], np.array(d["matriz"], dtype=float)
    P = 100 * M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(ANCHO * 0.78, ANCHO * 0.66))
    ax.imshow(P, cmap=LinearSegmentedColormap.from_list("azul", RAMPA), vmin=0, vmax=100)
    for i in range(len(clases)):
        for j in range(len(clases)):
            v = P[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 55 else TINTA, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(clases)), clases)
    ax.set_yticks(range(len(clases)), [f"{c}  (n = {int(n):,})" for c, n in zip(clases, M.sum(axis=1))])
    ax.set_xlabel("Clase que dijo el modelo", labelpad=6)
    ax.set_ylabel("Clase real", labelpad=6)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=8.5, colors=TINTA)
    ax.set_xticks(np.arange(-0.5, len(clases)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clases)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)        # separación con el color del fondo, sin bordes
    ax.tick_params(which="minor", length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.savefig(AQUI / "fig_confusion.pdf")
    plt.close(fig)


def componer_capturas(rutas, salida=AQUI / "fig_app.png", ancho=1024):
    """Arma en una rejilla de 2 × 2 las capturas de la aplicación, cada una con su letra (a, b, c, d) arriba."""
    from PIL import ImageDraw, ImageFont

    fuente = ImageFont.truetype(str(Path.home() / "Library" / "Fonts" / "OpenSans-Bold.ttf"), 30)
    caps = [Image.open(r).convert("RGB") for r in rutas]
    caps = [c.resize((ancho, round(c.height * ancho / c.width)), Image.LANCZOS) for c in caps]
    alto, sep, banda = max(c.height for c in caps), 22, 46
    lienzo = Image.new("RGB", (2 * ancho + sep, 2 * (alto + banda) + sep), "white")
    d = ImageDraw.Draw(lienzo)
    for k, c in enumerate(caps):
        x, y = (k % 2) * (ancho + sep), (k // 2) * (alto + banda + sep)
        d.text((x + 2, y + banda / 2 - 3), f"({'abcd'[k]})", font=fuente, fill="#0b0b0b", anchor="lm")
        lienzo.paste(c, (x, y + banda))
    lienzo.save(salida, optimize=True)
    return salida


if __name__ == "__main__":
    fig_sistema()
    fig_clases()
    fig_entrenamiento()
    fig_confusion()
    print("figuras listas en", AQUI)
