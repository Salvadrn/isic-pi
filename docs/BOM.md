# BOM · isic-pi (comprando la Raspberry Pi)

Lista de materiales para correr el clasificador de lesiones de piel en una **Raspberry Pi 5**, comprando todo nuevo. Entrenar no requiere hardware: se hace en Google Colab.

> Proyecto escolar, no es un dispositivo médico. **Nada de esto se ha probado en físico**: la elección de cámara y de distancia de enfoque sale de hojas de datos y de las fichas de la tienda, no de pruebas con la Pi ni con lesiones.

**Precios.** Publicados en [330ohms](https://www.330ohms.com/shop/category/raspberry-pi-raspberry-pi-5-2), tienda en México que tiene todo lo de esta lista, consultados el 2026-10-01. **No incluyen envío**; revisa en el carrito si el IVA ya está incluido. Raspberry Pi subió precios varias veces en 2026 por el costo de la memoria ([Tom's Hardware](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-price-increases-drastically-as-ai-shortage-bites-16gb-version-now-usd205-second-price-increase-in-three-months-over-70-percent-more-expensive-than-original-msrp)), así que revisa antes de comprar.

## A. Para correr el modelo

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | MXN | Dónde |
|---|---|---|---|---|---|---|---|
| A1 | Raspberry Pi 5 | 1 | 8 GB | La computadora que corre el modelo TFLite. Para este proyecto alcanza la de 4 GB ($2,636): ver «¿8 GB o 4 GB?» abajo. | Comprar | $4,192 | [330ohms RP-01112](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/rp-01112-raspberry-pi-5-8gb-1378) |
| A2 | Fuente oficial USB-C 27 W | 1 | 5.1 V / 5 A | Alimenta la Pi 5 y lo que le conectes por USB. Con una de 3 A funciona, pero limita la corriente de los puertos USB. | Comprar | $344 | [330ohms RP-01153](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/rp-01153-fuente-oficial-raspberry-pi-5-27w-1379) |
| A3 | microSD 64 GB | 1 | ADATA, UHS-I clase 10, hasta 80 MB/s, con adaptador SD | Raspberry Pi OS 64-bit, el proyecto y las fotos que tomes. La de 32 GB cuesta $316. | Comprar | $338 | [330ohms CN-00006](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/cn-00006-memoria-microsd-64gb-clase-10-adata-1369) |
| A4 | Carcasa oficial Raspberry Pi 5 | 1 | Con ventilador que se prende según la temperatura | La protege para llevarla a la escuela y evita que se caliente. Alternativa más barata: Active Cooler oficial, $140 (enfría, pero la placa queda descubierta). | Recomendado | $296 | [330ohms RP-01159](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/rp-01159-carcasa-oficial-raspberry-pi-5-1380) |

## B. Cámara (demo en vivo)

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | MXN | Dónde |
|---|---|---|---|---|---|---|---|
| B1 | Raspberry Pi Cámara 3 (Estándar) | 1 | 12 MP, enfoque automático, 75° diagonal, enfoca desde 10 cm | Tomar la foto de la lesión y clasificarla en vivo. La versión Wide (enfoca desde 5 cm) no la busqué en esta tienda. | Comprar si quieres demo en vivo | $644 | [330ohms RP-00872](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/rp-00872-raspberry-pi-camara-3-1324) |
| B2 | Cable de cámara para Pi 5 | 1 | 200 mm, 22 pines (0.5 mm) → 15 pines (1 mm) | La Pi 5 tiene un conector de cámara más pequeño que el de la cámara; sin este cable no se pueden conectar. La ficha confirma que es el de estándar a mini para Pi 5. | Comprar si llevas cámara | $32 | [330ohms RP-01128](https://www.330ohms.com/shop/raspberry-pi-raspberry-pi-5-2/rp-01128-cable-para-camara-raspberry-pi-200mm-1442) |

**Estándar o Wide.** La Estándar tiene menos distorsión, pero no enfoca a menos de 10 cm. La Wide enfoca desde 5 cm, con 120° de campo y más distorsión en los bordes. Con 12 MP se puede recortar el centro de la foto (el preproceso del proyecto recorta al centro cuadrado, igual en el entrenamiento y en la Pi), así que la Estándar es el punto de partida razonable.

## C. Opcional

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | MXN | Dónde |
|---|---|---|---|---|---|---|---|
| C1 | Lector de microSD | 1 | USB-C o USB-A, según tu Mac | Grabar el sistema desde la Mac con Raspberry Pi Imager. Sin precio verificado. | Si tu Mac no tiene lector | — | — |
| C2 | Cable micro-HDMI a HDMI | 1 | — | Solo si conectas monitor. Si trabajas por SSH desde la Mac no hace falta. Sin precio verificado. | Opcional | — | — |
| C3 | Soporte o cono impreso en 3D | 1 | PLA negro mate | Mantiene la cámara a distancia fija y bloquea la luz ambiente. Lo imprimes tú; diseño pendiente. | Opcional | — | — |
| C4 | Luz difusa constante (anillo LED o lámpara) | 1 | — | Sombras y brillos distintos cambian lo que ve el modelo. Idea, no probada; sin precio verificado. | Opcional | — | — |
| C5 | Lámina polarizadora lineal | 2 | — | Polarización cruzada para reducir reflejos, como en la dermatoscopia. Experimental, no probado; sin precio verificado. | Opcional | — | — |

## Cuánto cuesta

| Escenario | Con Pi 5 de 8 GB | Con Pi 5 de 4 GB |
|---|---|---|
| Lo mínimo para correr el modelo (Pi + fuente + microSD) | **$4,874** | $3,318 |
| Mínimo + carcasa | **$5,170** | $3,614 |
| Completo: con carcasa, cámara y cable | **$5,846** | $4,290 |

Precios de 330ohms del 2026-10-01, sin envío. Lo opcional sin precio no entra en los totales.

## ¿8 GB o 4 GB?

Para este proyecto **alcanza la Pi 5 de 4 GB** y ahorras **$1,556**. El modelo pesa ≈ 9 MB (2.27 millones de parámetros en float32) y la Pi solo lo usa para clasificar, no para entrenar; la memoria que ocupa al correr es del orden de cientos de MB (estimado, no medido en la Pi). La de 8 GB solo conviene si la vas a usar para otras cosas además de este proyecto.

## Otras referencias de precio

- **Pi 5 de 8 GB en otras tiendas de México:** entre ≈ $4,170 y $4,860 según resultados de búsqueda del 2026-10-01 (UNIT Electronics, Microbot, Amazon México). No revisé esas fichas una por una.
- **En Estados Unidos** (Adafruit, 2026-10-01): Pi 5 de 8 GB a 200 USD y de 4 GB a 130 USD. Sale parecido o más caro una vez que sumas envío e impuestos de importación.

## Software y servicios (sin costo)

| Qué | Para qué | Costo |
|---|---|---|
| Raspberry Pi OS 64-bit y Raspberry Pi Imager | Sistema de la Pi y grabarlo en la microSD | Gratis |
| Google Colab | Entrenar el modelo (GPU T4 con la cuota gratuita; no está garantizada) | Gratis |
| Google Drive | Relevo entre notebooks: el proyecto usa ≈ 4 GB de los 15 GB gratuitos | Gratis |
| Dataset `Atoany/ISIC2019` en Hugging Face | Imágenes de entrenamiento (≈ 9.8 GB de descarga) | Gratis |
| `ai-edge-litert`, Pillow, NumPy (y OpenCV si usas cámara USB) | Correr el modelo y leer imágenes en la Pi | Gratis |

## Qué conviene saber antes de comprar

- **Puedes empezar sin cámara.** Con la Pi puedes validar el modelo clasificando fotos del conjunto de test, y comprar la cámara y su cable ($676) solo si hace falta la demo en vivo.
- **El dataset es dermatoscópico** (lupa con luz polarizada, pegada a la piel). Con una cámara normal a 10 cm la precisión baja bastante, y no hay forma de saber cuánto hasta medirlo.
- **Un dermatoscopio de verdad cuesta cientos de dólares** (por ejemplo, el Dino-Lite MEDL4DW se anuncia en ≈ €580), fuera de un presupuesto escolar. Si algún día usas uno USB, verifica que sea compatible con Linux (UVC) antes de comprarlo.
