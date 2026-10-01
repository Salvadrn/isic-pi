# BOM · isic-pi

Lista de materiales para correr el clasificador de lesiones de piel en una **Raspberry Pi 5 (8 GB)**. Entrenar no requiere hardware: se hace en Google Colab.

> Proyecto escolar, no es un dispositivo médico. **Nada de esto se ha probado en físico**: la elección de cámara y de distancia de enfoque sale de hojas de datos y de lo que reportan las tiendas, no de pruebas con la Pi ni con lesiones.

**Precios.** Referencia en USD de Adafruit, consultada el 2026-10-01. MXN a 18.07 por dólar (FIX de Banxico del 1 de octubre, según prensa), **sin IVA ni envío**: en México suele costar más. Raspberry Pi subió precios varias veces en 2026 por el costo de la memoria ([Tom's Hardware](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-price-increases-drastically-as-ai-shortage-bites-16gb-version-now-usd205-second-price-increase-in-three-months-over-70-percent-more-expensive-than-original-msrp)), así que revisa el precio antes de comprar.

**Estado.** *Tengo*: ya lo tienes. *Verificar*: mira si ya lo tienes antes de comprarlo. *Comprar*: necesario para esa opción. *Opcional*: mejora, no es indispensable.

## A. Núcleo (clasificar fotos guardadas)

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | USD | MXN aprox. | Fuente |
|---|---|---|---|---|---|---|---|---|
| A1 | Raspberry Pi 5 | 1 | 8 GB | Corre el modelo TFLite (inferencia). | Tengo | — | — | — |
| A2 | Fuente oficial USB-C 27 W | 1 | 5.1 V / 5 A | Alimenta la Pi 5 y lo que le conectes por USB. Con una de 3 A funciona, pero limita la corriente de los puertos USB. Entre $12 y $15 según tienda. | Verificar | 14.04 | 254 | [Adafruit 5814](https://www.adafruit.com/product/5814) |
| A3 | microSD | 1 | ≥ 32 GB, clase A2 (recomendada 64 GB) | Raspberry Pi OS 64-bit, el proyecto y las capturas. Agotada en Adafruit; cualquier microSD A2 de marca sirve. Precio de la oficial de 64 GB. | Verificar | 12.79 | 231 | [Adafruit 6011](https://www.adafruit.com/product/6011) |
| A4 | Caja oficial con ventilador | 1 | Raspberry Pi Case para Pi 5 (ventilador con control de temperatura) | Protege la placa y evita que se caliente bajo carga sostenida. Alternativa: Active Cooler suelto, $11.95 a $13.50. | Opcional | 12.00 | 217 | [Adafruit 5816](https://www.adafruit.com/product/5816) |

## B. Cámara (demo en vivo)

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | USD | MXN aprox. | Fuente |
|---|---|---|---|---|---|---|---|---|
| B1 | Camera Module 3 (Estándar) | 1 | 12 MP, enfoque automático, 75° diagonal, enfoca desde 10 cm | Captura la lesión para clasificarla en vivo. Alternativa Wide (120°, enfoca desde 5 cm): $38.50, Adafruit 5658. | Comprar | 29.25 | 529 | [Adafruit 5657](https://www.adafruit.com/product/5657) |
| B2 | Cable de cámara para Pi 5, 22 pines → 15 pines | 1 | 200 mm, FPC 0.5 mm → 1 mm | Conecta el módulo (conector de 15 pines) a la Pi 5, cuyo conector de cámara es más pequeño (22 pines): el cable estándar de 15 pines no entra. | Comprar | 2.70 | 49 | [Adafruit 5818](https://www.adafruit.com/product/5818) |

**Estándar o Wide.** La Estándar tiene menos distorsión y cuesta menos, pero no enfoca a menos de 10 cm. La Wide enfoca desde 5 cm, con 120° de campo y más distorsión en los bordes. Con 12 MP se puede recortar el centro de la foto (el preproceso del proyecto recorta al centro cuadrado, igual en el entrenamiento y en la Pi), así que la Estándar es el punto de partida razonable.

## C. Opcional y experimental

| # | Componente | Cant. | Especificación | Para qué / notas | Estado | USD | MXN aprox. | Fuente |
|---|---|---|---|---|---|---|---|---|
| C1 | Lector de microSD (USB-C) | 1 | — | Grabar el sistema desde la Mac con Raspberry Pi Imager. Sin precio verificado. | Verificar | — | — | — |
| C2 | Cable micro-HDMI a HDMI | 1 | — | Solo si conectas monitor. Si trabajas por SSH no hace falta. Sin precio verificado. | Opcional | — | — | — |
| C3 | Soporte o cono impreso en 3D | 1 | PLA negro mate | Mantiene la cámara a distancia fija y bloquea la luz ambiente. Lo imprimes tú; diseño pendiente. | Opcional | — | — | — |
| C4 | Luz difusa constante (anillo LED o lámpara) | 1 | — | Sombras y brillos distintos cambian lo que ve el modelo. Idea, no probada; sin precio verificado. | Opcional | — | — | — |
| C5 | Lámina polarizadora lineal | 2 | — | Polarización cruzada para reducir reflejos, como en la dermatoscopia. Experimental, no probado; sin precio verificado. | Opcional | — | — | — |

## Cuánto cuesta cada escenario

| Escenario | USD | MXN aprox. |
|---|---|---|
| Solo clasificar fotos guardadas (sin cámara), si ya tienes fuente y microSD | 0.00 | 0 |
| Solo fotos guardadas, comprando fuente y microSD | 26.83 | 485 |
| Con cámara, si ya tienes fuente y microSD | 31.95 | 577 |
| Con cámara y caja, si ya tienes fuente y microSD | 43.95 | 794 |
| Todo nuevo (menos la Pi 5), con cámara y caja | 70.78 | 1,279 |

Los totales suman solo las filas con precio verificado; lo opcional sin precio no entra.

## Software y servicios (sin costo)

| Qué | Para qué | Costo |
|---|---|---|
| Raspberry Pi OS 64-bit y Raspberry Pi Imager | Sistema de la Pi y grabarlo en la microSD | Gratis |
| Google Colab | Entrenar el modelo (GPU T4 con la cuota gratuita; no está garantizada) | Gratis |
| Google Drive | Relevo entre notebooks: el proyecto usa ≈ 4 GB de los 15 GB gratuitos | Gratis |
| Dataset `Atoany/ISIC2019` en Hugging Face | Imágenes de entrenamiento (≈ 9.8 GB de descarga) | Gratis |
| `ai-edge-litert`, Pillow, NumPy (y OpenCV si usas cámara USB) | Correr el modelo y leer imágenes en la Pi | Gratis |

## Qué conviene saber antes de comprar

- **Empieza sin cámara.** Con la Pi puedes validar el modelo clasificando fotos del conjunto de test; compra la cámara solo si hace falta la demo en vivo.
- **El dataset es dermatoscópico** (lupa con luz polarizada, pegada a la piel). Con una cámara normal a 10 cm la precisión baja bastante, y no hay forma de saber cuánto hasta medirlo.
- **Un dermatoscopio de verdad cuesta cientos de dólares** (por ejemplo, el Dino-Lite MEDL4DW se anuncia en ≈ €580), fuera de un presupuesto escolar. Si algún día usas uno USB, verifica que sea compatible con Linux (UVC) antes de comprarlo.
- **Pi 5 y cámara:** el conector es de 22 pines, por eso el cable B2.
