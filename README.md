# isic-pi

Clasificador de lesiones de piel (dataset **ISIC 2019**, 8 clases) que se **entrena en Google Colab o en una Mac** y se **ejecuta en una Raspberry Pi 5**.

> Proyecto escolar. No es un dispositivo médico y no sirve para diagnosticar nada.

## Cómo encaja

```
   Google Colab o la Mac                                            Raspberry Pi 5
   ─────────────────────                                            ──────────────
 01 preparar datos ─▶ arreglos 224×224
 02 entrenar       ─▶ modelo.keras (MobileNetV2)
 03 exportar       ─▶ modelo/ del repo ── GitHub ── git pull ──▶  pi/clasificar.py ◀── cámara o fotos
```

## Plan por sesiones

Una pieza por sesión, cada una se prueba antes de seguir.

- [x] **Sesión 1:** repo + `notebooks/01_preparar_datos.ipynb`
- [x] **Sesión 2:** `notebooks/02_entrenar.ipynb` (MobileNetV2 con transfer learning y pesos por clase)
- [x] **Sesión 3:** `notebooks/03_exportar_tflite.ipynb` (métricas por clase en test y exportación a `.tflite`)
- [x] **Sesión 4:** `pi/clasificar.py` (clasificar fotos, webcam, cámara de la Pi y autoprueba; falta probarlo con una cámara de verdad)
- [x] **Primer modelo real (2026-10-07):** entrenado en la Mac con los notebooks 02 y 03; está en `modelo/` y ya corre en la Pi
- [ ] **Sesión 5:** scripts para instalar y enviar todo a la Pi, y este README completo

## Resultados del primer modelo (2026-10-07)

Entrenado en la Mac (Apple M4, en CPU porque TensorFlow no usa su GPU, 45 min) con todo ISIC 2019: 18,237 fotos para entrenar y 2,027 para validar. Medido en las **5,067 fotos de test**, que el modelo no vio al entrenar ni al elegir la mejor época:

| Clase | Fotos en test | Recall | Precisión |
|---|---:|---:|---:|
| AK · queratosis actínica | 182 | 0.45 | 0.42 |
| BCC · carcinoma basocelular | 606 | 0.79 | 0.66 |
| BKL · queratosis benigna | 507 | 0.62 | 0.48 |
| DF · dermatofibroma | 42 | 0.60 | 0.23 |
| **MEL · melanoma** | 949 | **0.47** | 0.76 |
| NV · nevo melanocítico (lunar) | 2,606 | 0.85 | 0.85 |
| SCC · carcinoma escamoso | 117 | 0.42 | 0.42 |
| VASC · lesión vascular | 58 | 0.86 | 0.67 |

- **Recall promedio por clase: 0.632** (en validación había dado 0.670). El accuracy es 0.725, pero lo infla NV, que es la mitad del test.
- **Melanoma, la debilidad principal:** encuentra 447 de 949. De los 502 que se le escapan, 298 los toma por lunar (NV) y 111 por queratosis benigna (BKL).
- Si se juntan las tres malignas (MEL, BCC y SCC) contra el resto, detecta el 66 % de las malignas y acierta el 90 % de las benignas.
- **El ligero** pierde 1.4 puntos de recall promedio en la Mac que lo exportó (0.619) y 1.0 en la Pi (0.623, medido en la Pi con las 5,067 fotos). Como además con 4 núcleos es más lento (11.5 ms contra 9.7 ms por foto), la Pi usa `modelo.tflite`.
- La mejor época en validación fue la 13 de 14 del ajuste (la 14 bajó de 0.670 a 0.652): el entrenamiento terminó por el límite de épocas, no porque dejara de mejorar. Más épocas de ajuste podrían mejorar el modelo; no está probado.
- **Medido en la Pi:** clasificando las 5,067 fotos de test da exactamente lo mismo que la Mac (misma clase en todas, con los dos modelos). Tarda 9.7 ms por foto con 4 núcleos (11.9 ms cuando ya se calentó), 12.7 ms con 2 y 21.7 ms con 1. La autoprueba da la misma clase que al exportar en 16 de 16 fotos, con diferencia máxima de 2.9 × 10⁻⁶ en las probabilidades.

Matriz de confusión: [modelo/confusion_test.png](modelo/confusion_test.png). Todas las cifras: [modelo/metricas_test.json](modelo/metricas_test.json) y [modelo/reporte_test.txt](modelo/reporte_test.txt).

## Hardware

Lista de materiales comprando la Raspberry, con precios de una tienda en México: [docs/BOM.md](docs/BOM.md) (también en [CSV](docs/BOM.csv)). Lo mínimo para correr el modelo (Pi 5 de 8 GB, fuente y microSD) son $4,874 MXN; con carcasa, cámara y cable, $5,846 MXN. Con la Pi 5 de 4 GB, que alcanza para este proyecto, se ahorran $1,556. Precios del 2026-10-01, sin envío.

**Corriente y calor (medido el 2026-10-07 con la Pi 5 de 8 GB, sin ventilador, clasificando el test completo):**

- **Conectada al USB-C de una laptop** (la Pi detecta 3 A) se apagó en cuanto usó los 4 núcleos: el 5 V bajó de 5.05 a 4.69 V, el kernel anotó `Undervoltage detected!` y 2 segundos después quedó apagada con el foco rojo. Con 1 y 2 núcleos aguantó. Así se explican sus dos caídas anteriores.
- **Con un cargador de pared UGREEN Nexode de 65 W** (también 3 A a 5 V) aguantó 4 minutos seguidos con los 4 núcleos, pero con 7 avisos breves de bajo voltaje (el 5 V bajó hasta 4.76 V). Funciona, pero va justo.
- **Sin ventilador** llega a 87 °C y se frena sola a 1.5 GHz: la misma clasificación pasa de 9.7 a 11.9 ms por foto.
- Para la presentación conviene la fuente oficial de 27 W (5 A) y el Active Cooler o la carcasa con ventilador (ver el BOM). Mientras tanto, `--hilos 2` gasta menos corriente y sigue siendo rápido (12.7 ms por foto).

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
- Vuelve a medir los dos `.tflite` en test con `ai-edge-litert`, el mismo intérprete que usa la Pi, y se detiene si el float32 no reproduce a Keras. Las cifras del float32 valen en cualquier máquina; las del ligero pueden cambiar un poco según el procesador (ver [Decisiones](#decisiones)).
- Recomienda el ligero solo si pierde 1 punto de recall promedio o menos.

Deja en `Mi unidad/isic-pi/exportado/` los dos modelos, `clases.txt`, `muestras_pi.npz` (fotos de test con las probabilidades esperadas, para comprobar en la Pi), el reporte, la matriz de confusión, `metricas_test.json` y un `isic_pi_modelo.zip` con todo, que se descarga solo.

**En tu Mac:** descomprime el zip dentro de la carpeta `modelo/` de este repo y súbelo a GitHub. La Pi lo baja con `git pull` (ver [Instalar en la Pi](#instalar-en-la-pi)).

## Entrenar en la Mac, sin Colab

Así se entrenó el primer modelo: es lo mismo que los notebooks 01 a 03, pero los datos quedan en `~/isic-pi-datos` en vez de Drive. Hace falta Python 3.12 con los mismos paquetes que [Probar los notebooks sin Colab](#probar-los-notebooks-sin-colab), en un entorno fuera del Escritorio (iCloud puede vaciar los archivos de un entorno que esté ahí). Homebrew trae Python 3.14; [uv](https://docs.astral.sh/uv/) baja su propio 3.12:

```bash
brew install uv
```

```bash
uv venv --python 3.12 ~/isic-pi-datos/venv && uv pip install --python ~/isic-pi-datos/venv/bin/python datasets pandas scikit-learn matplotlib tqdm pillow numpy tensorflow ai-edge-litert
```

```bash
cd ~/isic-pi-datos && venv/bin/python ~/Desktop/isic-pi/local/preparar_datos.py --destino ~/isic-pi-datos
```

```bash
cd ~/isic-pi-datos && venv/bin/python ~/Desktop/isic-pi/local/correr_notebook.py ~/Desktop/isic-pi/notebooks/02_entrenar.ipynb PERMITIR_CPU=True
```

```bash
cd ~/isic-pi-datos && venv/bin/python ~/Desktop/isic-pi/local/correr_notebook.py ~/Desktop/isic-pi/notebooks/03_exportar_tflite.ipynb
```

- `local/preparar_datos.py` reemplaza al notebook 01: baja los 20 archivos parquet del dataset uno por uno, reduce cada foto con la misma función que usa la Pi y borra el parquet. Así no necesita los ≈ 20 GB de la caché de Hugging Face. Si se corta, retoma desde el último archivo terminado.
- `local/correr_notebook.py` corre las celdas de un notebook fuera de Colab. `PERMITIR_CPU=True` deja entrenar sin GPU.
- Tiempos medidos en una MacBook con M4 y 16 GB: cada parquet tardó de 40 s a 3.4 min en bajar según la red, entrenar 45 min y exportar 1.4 min. Al final quedan ≈ 9.5 GB en `~/isic-pi-datos`: 3.9 GB de datos, 3.9 GB de copia de trabajo (`trabajo/`) y 1.8 GB del entorno.
- El resultado queda en `~/isic-pi-datos/isic-pi/exportado/isic_pi_modelo.zip`, que se descomprime en `modelo/` igual que el de Colab.
- La Mac tiene que estar conectada a la corriente, con la tapa abierta. Para que no se duerma, conviene correrlo con `caffeinate -i -s` antes del comando.

## Sesión 4 · El programa: `pi/clasificar.py`

Clasifica fotos con el modelo exportado. Corre igual en la Pi y en la Mac. Desde la carpeta del repo:

```bash
.venv/bin/python pi/clasificar.py foto.jpg carpeta/
```

```bash
.venv/bin/python pi/clasificar.py --webcam --vivo
```

```bash
.venv/bin/python pi/clasificar.py --camara
```

```bash
.venv/bin/python pi/clasificar.py --autoprueba
```

- `--webcam`: una foto con la webcam (la de la Mac o una USB). Con `--vivo` abre una ventana en vivo con un cuadro guía, que es lo que ve el modelo; espacio guarda la foto en `capturas/` y q sale.
- `--camara`: una foto con la cámara de la Pi (Camera Module 3, con `rpicam-still`).
- `--autoprueba`: clasifica las fotos de muestra que vienen con el modelo y comprueba que da lo mismo que al exportar: la misma clase (salvo casi empates) y hasta 0.001 de diferencia en las probabilidades (0.05 con `--ligero`, ver [Decisiones](#decisiones)).
- Otras opciones: `--modelo CARPETA`, `--ligero`, `--top N` y `--json`.

Sin `--modelo` usa el modelo de piel de `modelo/`.

**Modelo de objetos:** `--modelo demo/objetos` usa MobileNetV2 con sus pesos públicos de ImageNet (1,000 objetos comunes; no sabe nada de piel). Sirve para probar una cámara sin el modelo de piel: enséñale una taza y debe decir algo como `coffee_mug`. Se genera con `demo/generar_modelo_objetos.py`.

**En la Mac**, una sola vez:

```bash
cd ~/Desktop/isic-pi && python3 -m venv .venv && .venv/bin/pip install ai-edge-litert pillow numpy opencv-python
```

La primera vez, macOS pide permiso de cámara para la Terminal. Si el Escritorio se sincroniza con iCloud, el primer arranque puede tardar minutos mientras iCloud vuelve a bajar los archivos del `.venv`.

**En la Pi** el `.venv` ya tiene lo necesario para fotos, `--autoprueba` y `--camara`. Para una webcam USB en la Pi hace falta OpenCV (`.venv/bin/pip install opencv-python-headless`); `--vivo` además necesita un monitor conectado y `opencv-python`.

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

```bash
python tests/probar_clasificar.py
```

La última también corre en la Pi (`.venv/bin/python tests/probar_clasificar.py`): solo necesita `ai-edge-litert`, NumPy y Pillow, porque simula la webcam y `rpicam-still`.

Las pruebas 02 y 03 descargan los pesos de ImageNet (≈ 10 MB) y tardan unos minutos en CPU.

## Estado: qué está probado y qué no

- **Notebook 01:** probado en Python 3.12 con un dataset falso del mismo esquema (`image` + `label`), en los dos modos, incluyendo etiquetas desconocidas en test, imágenes no cuadradas, en escala de grises y con canal alfa. **Con el dataset real** lo reemplazó `local/preparar_datos.py` en la Mac (2026-10-07), con la misma función de preproceso: 20,264 fotos de train y 5,067 de test. **El notebook 01 en sí no se ha corrido en Colab con el dataset real:** el montaje de Drive, el tiempo y el espacio en Colab son estimados.
- **Notebook 02:** probado en CPU (Python 3.12, TensorFlow 2.21, Keras 3.15) con datos falsos que fabrica el propio notebook 01 y con 8 clases de colores sólidos. Verificado: imágenes y etiquetas llegan alineadas a Keras, `class_weight` se aplica con `tf.data`, las guardas fallan con mensaje claro (sin datos del 01, sin GPU, modelo ya entrenado), el modelo guardado reproduce las predicciones, el test no se copia y el modelo exportado no lleva capas de aumento. En colores sólidos aprende (recall promedio 0.91 en validación con la configuración por defecto) y un ajuste fino que destruye el modelo se descarta. **Corrido con ISIC real** el 2026-10-07 en la Mac, en CPU, con `local/correr_notebook.py` (45 min, recall promedio 0.670 en validación). **No probado:** GPU, Colab ni Drive.
- **Notebook 03:** probado en CPU (Python 3.12, TensorFlow 2.21, Keras 3.15, ai-edge-litert 2.2.0), encadenado con el 01 y el 02 sobre datos falsos. Verificado: la conversión directa desde Keras funciona en esta versión de TensorFlow, el float32 reproduce a Keras (diferencia máxima de 7 × 10⁻⁶ en las probabilidades), el `.tflite` leído de disco reproduce las muestras, el zip trae lo mismo que la carpeta y las guardas fallan con mensaje claro (sin 01, sin 02, ya exportado). Con 8 clases de colores sólidos el float32 sacó recall promedio 0.969 en test y el ligero 0.875. **Corrido con ISIC real** el 2026-10-07 en la Mac (ver [Resultados](#resultados-del-primer-modelo-2026-10-07)). **No probado:** Colab ni Drive.
- **Raspberry Pi 5 (8 GB, Raspberry Pi OS Trixie, Python 3.13):** entorno instalado el 2026-10-06 en `~/isic-pi/.venv` (`ai-edge-litert` 2.2.0, NumPy 2.5.3, Pillow 12.3.0). **Medido en la Pi** con un modelo de juguete de la misma arquitectura que el real: da las mismas probabilidades que la Mac (diferencia máxima 2.4 × 10⁻⁷, misma clase en 16 de 16 muestras) y tarda 9.8 ms por foto con `modelo.tflite` y 13.0 ms con el ligero (4 hilos, sin contar abrir y reducir la foto). **Con el modelo real** (2026-10-07, bajado con `git pull`): misma clase que al exportar en 16 de 16 muestras, con diferencia máxima de 2.9 × 10⁻⁶ (el ligero difiere hasta 1.5 × 10⁻², ver [Decisiones](#decisiones)). En las 5,067 fotos de test da la misma clase que la Mac en todas, con los dos modelos: recall promedio 0.632 el float32 y 0.623 el ligero. Con alimentación de laptop se apaga al usar los 4 núcleos (ver [Hardware](#hardware)).
- **Programa (`pi/clasificar.py`):** probado en la Mac y en la Pi con `tests/probar_clasificar.py`. El preproceso es idéntico píxel a píxel al del notebook 01 (6 casos, incluida una foto de 12 MP y una con rotación EXIF). La autoprueba coincide: en la Pi, 8 de 8 muestras, diferencia máxima de 8.2 × 10⁻⁷ y 10.1 ms por foto con el modelo de objetos. Cada modo (archivos, webcam, en vivo y cámara de la Pi) da lo mismo que el modelo aplicado directo. **La webcam y la cámara de la Pi están simuladas en la prueba:** con la webcam real de la Mac la ventana en vivo sí abre (2026-10-06, con permiso de cámara para la Terminal; desde la línea de comandos de Claude macOS no da la cámara), y la cámara de la Pi todavía no se compra.
- **Scripts (sesión 5):** aún sin escribir.

## Decisiones

- **MobileNetV2 + TFLite:** el modelo se entrena en Colab o en la Mac y a la Pi solo llega un archivo `.tflite` de unos pocos MB. La Pi 5 no entrena.
- **Preproceso compartido:** entrenamiento y Raspberry reducen las fotos con la misma función (recorte central cuadrado + resize bilineal a 224×224) para que el modelo vea lo mismo en los dos lados.
- **Datos fuera de git:** los arreglos pesan ≈ 4 GB y viven en Drive (con Colab) o en `~/isic-pi-datos` (con la Mac); el repo solo guarda código y el modelo exportado (≈ 13 MB en `modelo/`).
- **Mejores pesos en archivo, no en memoria:** `get_weights()` / `set_weights()` dependen del orden de las variables, que en otras versiones de Keras cambia al congelar y descongelar capas y mezclaría en silencio los vectores de BatchNorm. En Keras 3.15 el orden no cambia, pero guardar y cargar desde archivo es seguro en cualquier versión.

- **float32 por defecto en la Pi:** el ligero pesa ≈ 2.5 MB en vez de ≈ 9 MB, pero con ISIC real pierde recall promedio (1.4 puntos en la Mac que lo exportó, 0.632 → 0.619; 1.0 en la Pi, 0.632 → 0.623) y con 4 núcleos es más lento en la Pi (11.5 ms contra 9.7 ms por foto; con 1 núcleo es al revés, 19.0 contra 21.7 ms). El notebook 03 lo recomienda si pierde 1 punto o menos; en la Pi queda justo en el límite, así que decide la velocidad.
- **Autoprueba más holgada para el ligero:** el ligero convierte a int8 los cálculos intermedios al vuelo, y no todas las combinaciones de procesador y versión del intérprete usan las mismas rutinas para eso. Medido: la Pi (con `ai-edge-litert` 2.2.0 y 2.3.0) y la Mac con 2.2.0 dan lo mismo entre sí, pero en las 16 muestras difieren hasta 1.5 × 10⁻² de la Mac M4 con 2.3.0 que exportó, con la misma clase en todas; en todo el test sí cambia la clase de algunas fotos (por eso 0.619 contra 0.623). El float32 coincide hasta 3 × 10⁻⁶ en todos. Por eso el ligero se acepta hasta 0.05 de diferencia. La clase tiene que coincidir, salvo en las muestras que al exportar estaban casi empatadas (las dos primeras a menos del doble de la tolerancia).
- **Muestras de comprobación:** con el modelo viajan unas fotos de test y las probabilidades que dio al exportar (en Colab o en la Mac), para que la Pi verifique que su intérprete da lo mismo.

## Limitaciones conocidas

- Las fotos del dataset son **dermatoscópicas** (lupa con luz polarizada). Con una cámara normal la precisión baja bastante.
- **Se le escapa más de la mitad de los melanomas** (recall 0.47 en test). Sirve para mostrar cómo funciona el flujo completo, no para opinar sobre una lesión real.
- La partición train/test viene hecha en Hugging Face. Si fotos de la misma lesión quedaron en ambos lados, las métricas saldrán más optimistas de lo real; no pude verificar cómo se hizo.
