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
- [x] **Sesión 3:** `notebooks/03_exportar_tflite.ipynb` (métricas por clase en test y exportación a `.tflite`)
- [ ] **Sesión 4:** `pi/clasificar.py` (clasificar fotos y tomar fotos con la cámara)
- [ ] **Sesión 5:** scripts para instalar y enviar todo a la Pi, y este README completo

## Hardware

Lista de materiales comprando la Raspberry, con precios de una tienda en México: [docs/BOM.md](docs/BOM.md) (también en [CSV](docs/BOM.csv)). Lo mínimo para correr el modelo (Pi 5 de 8 GB, fuente y microSD) son $4,874 MXN; con carcasa, cámara y cable, $5,846 MXN. Con la Pi 5 de 4 GB, que alcanza para este proyecto, se ahorran $1,556. Precios del 2026-10-01, sin envío. Nada de eso está probado en físico todavía.

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

## Sesión 3 · Evaluar en test y exportar a TFLite

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salvadrn/isic-pi/blob/main/notebooks/03_exportar_tflite.ipynb)

1. Necesita haber corrido los notebooks 01 y 02.
2. Entorno de ejecución **CPU** (el intérprete de TFLite no usa la GPU).
3. La primera vez pon `MODO_PRUEBA = True` (usa `datos_prueba/` y `modelo_prueba/`).
4. Corrida completa con `MODO_PRUEBA = False`.

Qué hace:
- Mide el modelo **por primera vez en test**: accuracy, recall por clase y su promedio, F1 y matriz de confusión. Esa es la cifra para el reporte.
- Convierte a `modelo.tflite` (float32, ≈ 9 MB, mismas probabilidades que Keras) y `modelo_ligero.tflite` (pesos en int8, ≈ 2.5 MB).
- Vuelve a medir los dos `.tflite` en test con `ai-edge-litert`, el mismo intérprete que usará la Pi, y se detiene si el float32 no reproduce a Keras.
- Recomienda el ligero solo si pierde 1 punto de recall promedio o menos.

Deja en `Mi unidad/isic-pi/exportado/` los dos modelos, `clases.txt`, `muestras_pi.npz` (fotos de test con las probabilidades esperadas, para comprobar en la Pi), el reporte, la matriz de confusión, `metricas_test.json` y un `isic_pi_modelo.zip` con todo, que se descarga solo.

**En tu Mac:** descomprime el zip dentro de la carpeta `modelo/` de este repo. Esa carpeta es la que se copiará a la Pi en la sesión 4.

## Instalar en la Pi

La Pi tiene su propia copia del repo en `~/isic-pi`, conectada a GitHub con la sesión de `gh` de Adrián (desde el 2026-10-06). Los cambios se hacen en la Mac, se suben a GitHub y la Pi los baja:

```
Mac (editamos) ──push──▶ GitHub ──git pull──▶ Pi
```

Desde la Mac (`isic-pi` es el alias de la Pi en `~/.ssh/config`):

```bash
ssh isic-pi 'cd ~/isic-pi && git pull'
```

O en la terminal de la Pi:

```bash
cd ~/isic-pi && git pull
```

**En una Pi nueva** (Raspberry Pi OS 64-bit). La nuestra se instaló copiando el repo desde la Mac y después se conectó a GitHub; estos pasos son el equivalente desde cero, sin probar tal cual:

```bash
sudo apt update && sudo apt install -y git gh python3-venv
```

```bash
gh auth login
```

```bash
gh repo clone Salvadrn/isic-pi ~/isic-pi && cd ~/isic-pi && python3 -m venv .venv && .venv/bin/pip install --upgrade pip ai-edge-litert pillow numpy
```

La sesión 5 convierte esto en scripts.

## Probar los notebooks sin Colab

Las pruebas ejecutan las celdas de cada notebook sobre datos falsos, sin Hugging Face, Drive ni GPU. Probado con Python 3.12:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install datasets pandas scikit-learn matplotlib tqdm pillow numpy tensorflow ai-edge-litert
```

```bash
python tests/probar_nb01_datos.py
```

```bash
python tests/probar_nb02_entrenar.py
```

```bash
python tests/probar_nb03_exportar.py
```

Las pruebas 02 y 03 descargan los pesos de ImageNet (≈ 10 MB) y tardan unos minutos en CPU.

## Estado: qué está probado y qué no

- **Notebook 01:** probado en Python 3.12 con un dataset falso del mismo esquema (`image` + `label`), en los dos modos, incluyendo etiquetas desconocidas en test, imágenes no cuadradas, en escala de grises y con canal alfa. **Todavía no lo he corrido en Colab con el dataset real:** la descarga desde Hugging Face, el montaje de Drive, el tiempo y el espacio ocupado son estimados, no medidos.
- **Notebook 02:** probado en CPU (Python 3.12, TensorFlow 2.21, Keras 3.15) con datos falsos que fabrica el propio notebook 01 y con 8 clases de colores sólidos. Verificado: imágenes y etiquetas llegan alineadas a Keras, `class_weight` se aplica con `tf.data`, las guardas fallan con mensaje claro (sin datos del 01, sin GPU, modelo ya entrenado), el modelo guardado reproduce las predicciones, el test no se copia y el modelo exportado no lleva capas de aumento. En colores sólidos aprende (recall promedio 0.91 en validación con la configuración por defecto) y un ajuste fino que destruye el modelo se descarta. **No probado:** GPU, Drive ni ISIC real. El tiempo de entrenamiento y la calidad por clase salen de la corrida en Colab, no de estas pruebas.
- **Notebook 03:** probado en CPU (Python 3.12, TensorFlow 2.21, Keras 3.15, ai-edge-litert 2.2.0), encadenado con el 01 y el 02 sobre datos falsos. Verificado: la conversión directa desde Keras funciona en esta versión de TensorFlow, el float32 reproduce a Keras (diferencia máxima de 7 × 10⁻⁶ en las probabilidades), el `.tflite` leído de disco reproduce las muestras, el zip trae lo mismo que la carpeta y las guardas fallan con mensaje claro (sin 01, sin 02, ya exportado). Con 8 clases de colores sólidos el float32 sacó recall promedio 0.969 en test y el ligero 0.875. **No probado:** Colab, Drive, ISIC real y el intérprete de la Pi, que es ARM (eso lo comprueban las muestras en la sesión 4).
- **Raspberry Pi 5 (8 GB, Raspberry Pi OS Trixie, Python 3.13):** entorno instalado el 2026-10-06 en `~/isic-pi/.venv` (`ai-edge-litert` 2.2.0, NumPy 2.5.3, Pillow 12.3.0). **Medido en la Pi** con un modelo de juguete de la misma arquitectura que el real: da las mismas probabilidades que la Mac (diferencia máxima 2.4 × 10⁻⁷, misma clase en 16 de 16 muestras) y tarda 9.8 ms por foto con `modelo.tflite` y 13.0 ms con el ligero (4 hilos, sin contar abrir y reducir la foto). El modelo real todavía no existe.
- **Programa de la Pi (sesión 4) y scripts (sesión 5):** aún sin escribir.

## Decisiones

- **MobileNetV2 + TFLite:** el modelo se entrena en Colab y a la Pi solo llega un archivo `.tflite` de unos pocos MB. La Pi 5 no entrena.
- **Preproceso compartido:** entrenamiento y Raspberry reducen las fotos con la misma función (recorte central cuadrado + resize bilineal a 224×224) para que el modelo vea lo mismo en los dos lados.
- **Datos en Drive, no en git:** los arreglos pesan ≈ 4 GB; el repo solo guarda código (y, más adelante, el modelo ya exportado).
- **Mejores pesos en archivo, no en memoria:** `get_weights()` / `set_weights()` dependen del orden de las variables, que en otras versiones de Keras cambia al congelar y descongelar capas y mezclaría en silencio los vectores de BatchNorm. En Keras 3.15 el orden no cambia, pero guardar y cargar desde archivo es seguro en cualquier versión.

- **float32 por defecto en la Pi:** el ligero pesa ≈ 2.5 MB en vez de ≈ 9 MB, pero en la prueba con datos falsos perdió 9 puntos de recall promedio y en la Pi fue más lento (13.0 ms contra 9.8 ms por foto). El notebook 03 lo mide con ISIC real y solo lo recomienda si pierde 1 punto o menos.
- **Muestras de comprobación:** con el modelo viajan unas fotos de test y las probabilidades que dio en Colab, para que la Pi verifique que su intérprete da lo mismo.

## Limitaciones conocidas

- Las fotos del dataset son **dermatoscópicas** (lupa con luz polarizada). Con una cámara normal la precisión baja bastante.
- La partición train/test viene hecha en Hugging Face. Si fotos de la misma lesión quedaron en ambos lados, las métricas saldrán más optimistas de lo real; no pude verificar cómo se hizo.
