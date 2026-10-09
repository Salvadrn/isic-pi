#!/bin/bash
# Abre la app de pantalla táctil a pantalla completa en la Pi: arranca pi/app.py y Chromium en modo kiosco.
# Para salir: mantén presionado el estado de la cámara (arriba a la izquierda) y toca «Salir»; se cierran los dos.
#
# Uso:  pi/app_kiosco.sh [opciones de pi/app.py]      por ejemplo: pi/app_kiosco.sh --camara usb
# El registro queda en ~/.cache/isic-pi/app.log. Para ponerla en el menú o al encender: pi/instalar_app.sh

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PUERTO=8000
CACHE="$HOME/.cache/isic-pi"
mkdir -p "$CACHE"

if curl -sf "http://127.0.0.1:$PUERTO/estado" > /dev/null; then
  echo "La app ya está corriendo en el puerto $PUERTO." >&2
  exit 1
fi

cd "$RAIZ" || exit 1
echo "--- $(date '+%F %T') arranca la app" >> "$CACHE/app.log"
"$RAIZ/.venv/bin/python" -u pi/app.py --puerto "$PUERTO" "$@" >> "$CACHE/app.log" 2>&1 &
APP=$!

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PUERTO/estado" > /dev/null && break
  if ! kill -0 "$APP" 2> /dev/null; then
    echo "La app no arrancó; mira $CACHE/app.log" >&2
    exit 1
  fi
  sleep 0.5
done

NAVEGADOR="$(command -v chromium || command -v chromium-browser)"
if [ -z "$NAVEGADOR" ]; then
  echo "No encuentro Chromium. La app sigue en http://127.0.0.1:$PUERTO" >&2
  wait "$APP"
  exit 0
fi

# Perfil propio: no toca el Chromium de siempre ni pregunta por contraseñas o por restaurar pestañas.
"$NAVEGADOR" --kiosk --noerrdialogs --disable-infobars --no-first-run --disable-session-crashed-bubble \
  --disable-features=Translate --password-store=basic --user-data-dir="$CACHE/chromium" \
  "http://127.0.0.1:$PUERTO/?kiosco" > /dev/null 2>&1 &
NAV=$!

wait -n "$APP" "$NAV"           # cuando uno de los dos termina, se cierra el otro
kill "$APP" "$NAV" 2> /dev/null
wait 2> /dev/null
