# isic-pi

Clasificador de lesiones de piel (dataset **ISIC 2019**, 8 clases) que se **entrena en Google Colab** y se **ejecuta en una Raspberry Pi 5**.

> Proyecto escolar. No es un dispositivo médico y no sirve para diagnosticar nada.

## Cómo encaja

```
         Google Colab                                         Raspberry Pi 5
         ────────────                                         ──────────────
 01 preparar datos ─▶ Drive: arreglos 224×224
 02 entrenar       ─▶ Drive: pesos (MobileNetV2)
 03 exportar       ─▶ modelo.tflite ──── rsync ────▶  pi/clasificar.py ◀── cámara o fotos
```

## Plan por sesiones

Una pieza por sesión, cada una se prueba antes de seguir.

- [x] **Sesión 1:** repo + `notebooks/01_preparar_datos.ipynb`
- [ ] **Sesión 2:** `02_entrenar.ipynb` (MobileNetV2 con transfer learning y pesos por clase)
- [ ] **Sesión 3:** `03_exportar_tflite.ipynb` (métricas por clase y exportación a `.tflite`)
- [ ] **Sesión 4:** `pi/clasificar.py` (clasificar fotos y tomar fotos con la cámara)
- [ ] **Sesión 5:** scripts para instalar y enviar todo a la Pi, y este README completo

## Sesión 1 · Preparar datos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salvadrn/isic-pi/blob/main/notebooks/01_preparar_datos.ipynb)

1. Abre el notebook en Colab con un entorno de ejecución **CPU** (no hace falta GPU).
2. La primera vez pon `MODO_PRUEBA = True`: usa ~30 imágenes por clase y guarda en `datos_prueba/`, para comprobar que todo corre.
3. Corrida completa con `MODO_PRUEBA = False`. Descarga el dataset (≈ 9.8 GB) y deja ≈ 4 GB en `Mi unidad/isic-pi/datos/`.

Lo que deja en Drive: `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy`, `split.npz`, `clases.txt` y `meta.json`. Los notebooks siguientes leen de ahí, así que cada sesión empieza sin repetir la descarga.

## Estado: qué está probado y qué no

- **Notebook 01:** probado en Python 3.12 con un dataset falso del mismo esquema (`image` + `label`), en los dos modos, incluyendo etiquetas desconocidas en test, imágenes no cuadradas, en escala de grises y con canal alfa. **Todavía no lo he corrido en Colab con el dataset real:** la descarga desde Hugging Face, el montaje de Drive, el tiempo y el espacio ocupado son estimados, no medidos.
- **Todo lo demás:** aún sin escribir.

## Decisiones

- **MobileNetV2 + TFLite:** el modelo se entrena en Colab y a la Pi solo llega un archivo `.tflite` de unos pocos MB. La Pi 5 no entrena.
- **Preproceso compartido:** entrenamiento y Raspberry reducen las fotos con la misma función (recorte central cuadrado + resize bilineal a 224×224) para que el modelo vea lo mismo en los dos lados.
- **Datos en Drive, no en git:** los arreglos pesan ≈ 4 GB; el repo solo guarda código (y, más adelante, el modelo ya exportado).

## Limitaciones conocidas

- Las fotos del dataset son **dermatoscópicas** (lupa con luz polarizada). Con una cámara normal la precisión baja bastante.
- La partición train/test viene hecha en Hugging Face. Si fotos de la misma lesión quedaron en ambos lados, las métricas saldrán más optimistas de lo real; no pude verificar cómo se hizo.
