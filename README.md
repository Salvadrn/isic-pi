# isic-pi

Clasificador de lesiones de piel (dataset **ISIC 2019**, 8 clases) que se **entrena en Google Colab** y se **ejecuta en una Raspberry Pi 5**.

> Proyecto escolar. No es un dispositivo médico y no sirve para diagnosticar nada.

## Cómo encaja

```
         Google Colab                                         Raspberry Pi 5
         ────────────                                         ──────────────
 01 preparar datos ─▶ Drive: arreglos 224×224
 02 entrenar       ─▶ Drive: modelo.keras (MobileNetV2)
 03 exportar       ─▶ modelo.tflite ──── rsync ────▶  pi/clasificar.py ◀── cámara o fotos
```

## Plan por sesiones

Una pieza por sesión, cada una se prueba antes de seguir.

- [x] **Sesión 1:** repo + `notebooks/01_preparar_datos.ipynb`
- [x] **Sesión 2:** `notebooks/02_entrenar.ipynb` (MobileNetV2 con transfer learning y pesos por clase)
- [ ] **Sesión 3:** `03_exportar_tflite.ipynb` (métricas por clase en test y exportación a `.tflite`)
- [ ] **Sesión 4:** `pi/clasificar.py` (clasificar fotos y tomar fotos con la cámara)
- [ ] **Sesión 5:** scripts para instalar y enviar todo a la Pi, y este README completo

## Sesión 1 · Preparar datos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salvadrn/isic-pi/blob/main/notebooks/01_preparar_datos.ipynb)

1. Abre el notebook en Colab con un entorno de ejecución **CPU** (no hace falta GPU).
2. La primera vez pon `MODO_PRUEBA = True`: usa ~30 imágenes por clase y guarda en `datos_prueba/`, para comprobar que todo corre.
3. Corrida completa con `MODO_PRUEBA = False`. Descarga el dataset (≈ 9.8 GB) y deja ≈ 4 GB en `Mi unidad/isic-pi/datos/`.

Lo que deja en Drive: `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy`, `split.npz`, `clases.txt` y `meta.json`. Los notebooks siguientes leen de ahí, así que cada sesión empieza sin repetir la descarga.

## Sesión 2 · Entrenar

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salvadrn/isic-pi/blob/main/notebooks/02_entrenar.ipynb)

1. Necesita haber corrido el notebook 01 (lee `Mi unidad/isic-pi/datos/`).
2. Entorno de ejecución **GPU (T4)**.
3. La primera vez pon `MODO_PRUEBA = True`: usa `datos_prueba/` del notebook 01, una época por fase, y corre incluso sin GPU. Sirve para comprobar que todo funciona antes de esperar el entrenamiento completo.
4. Corrida completa con `MODO_PRUEBA = False`.

Cómo entrena:
- **Fase 1:** MobileNetV2 (preentrenada en ImageNet) congelada; solo aprende la capa final de 8 clases.
- **Fase 2:** se descongelan las últimas 40 capas de MobileNetV2 (menos las BatchNorm) y se afina con un paso de aprendizaje 10 veces menor. Si afinar no mejora lo que ya logró la fase 1, se queda con la fase 1.
- **Desbalance:** cada foto pesa `(N / (K · n_clase)) ** 0.5` en la pérdida, y se decide cuándo parar y qué pesos conservar con el **recall promedio por clase** en validación, no con el accuracy (NV domina y lo infla).
- **Aumento de datos** (giros, rotaciones, zoom, brillo, contraste) solo durante el entrenamiento; no forma parte del modelo exportado.
- El escalado de píxeles a [-1, 1] va **dentro** del modelo, así la Raspberry manda los píxeles 0–255 tal cual.
- El conjunto de **test no se toca** aquí; se reserva para la sesión 3.

Deja en `Mi unidad/isic-pi/modelo/`: `modelo.keras`, `clases.txt`, `entrenamiento.json` (hiperparámetros, pesos por clase, historial, recall por clase), `reporte_val.txt`, `curvas.png` y `confusion_val.png`.

Si el recall de alguna clase sale muy bajo, las perillas están en la celda de configuración: `POTENCIA_PESOS`, `EPOCAS_AJUSTE` y `CAPAS_A_AJUSTAR` (cada cambio requiere `FORZAR = True`).

## Probar los notebooks sin Colab

Las pruebas ejecutan las celdas de cada notebook sobre datos falsos, sin Hugging Face, Drive ni GPU. Probado con Python 3.12:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install datasets pandas scikit-learn matplotlib tqdm pillow numpy tensorflow
```

```bash
python tests/probar_nb01_datos.py
```

```bash
python tests/probar_nb02_entrenar.py
```

La segunda descarga los pesos de ImageNet (≈ 10 MB) y tarda unos minutos en CPU.

## Estado: qué está probado y qué no

- **Notebook 01:** probado en Python 3.12 con un dataset falso del mismo esquema (`image` + `label`), en los dos modos, incluyendo etiquetas desconocidas en test, imágenes no cuadradas, en escala de grises y con canal alfa. **Todavía no lo he corrido en Colab con el dataset real:** la descarga desde Hugging Face, el montaje de Drive, el tiempo y el espacio ocupado son estimados, no medidos.
- **Notebook 02:** probado en CPU (Python 3.12, TensorFlow 2.21, Keras 3.15) con datos falsos que fabrica el propio notebook 01 y con 8 clases de colores sólidos. Verificado: imágenes y etiquetas llegan alineadas a Keras, `class_weight` se aplica con `tf.data`, las guardas fallan con mensaje claro (sin datos del 01, sin GPU, modelo ya entrenado), el modelo guardado reproduce las predicciones, el test no se copia y el modelo exportado no lleva capas de aumento. En colores sólidos aprende (recall promedio 0.91 en validación con la configuración por defecto) y un ajuste fino que destruye el modelo se descarta. **No probado:** GPU, Drive ni ISIC real. El tiempo de entrenamiento y la calidad por clase salen de la corrida en Colab, no de estas pruebas.
- **Todo lo demás:** aún sin escribir.

## Decisiones

- **MobileNetV2 + TFLite:** el modelo se entrena en Colab y a la Pi solo llega un archivo `.tflite` de unos pocos MB. La Pi 5 no entrena.
- **Preproceso compartido:** entrenamiento y Raspberry reducen las fotos con la misma función (recorte central cuadrado + resize bilineal a 224×224) para que el modelo vea lo mismo en los dos lados.
- **Datos en Drive, no en git:** los arreglos pesan ≈ 4 GB; el repo solo guarda código (y, más adelante, el modelo ya exportado).
- **Mejores pesos en archivo, no en memoria:** `get_weights()` / `set_weights()` dependen del orden de las variables, que en otras versiones de Keras cambia al congelar y descongelar capas y mezclaría en silencio los vectores de BatchNorm. En Keras 3.15 el orden no cambia, pero guardar y cargar desde archivo es seguro en cualquier versión.

## Limitaciones conocidas

- Las fotos del dataset son **dermatoscópicas** (lupa con luz polarizada). Con una cámara normal la precisión baja bastante.
- La partición train/test viene hecha en Hugging Face. Si fotos de la misma lesión quedaron en ambos lados, las métricas saldrán más optimistas de lo real; no pude verificar cómo se hizo.
