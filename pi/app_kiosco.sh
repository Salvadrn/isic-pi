#!/bin/bash
# Abre la app de pantalla táctil a pantalla completa en la Pi: arranca pi/app.py y Chromium en modo kiosco.
# Para salir: mantén presionado el estado de la cámara (arriba a la izquierda) y toca «Salir»; se cierran los dos.
#
# Uso:  pi/app_kiosco.sh [opciones de pi/app.py]      por ejemplo: pi/app_kiosco.sh --camara usb
# Otro puerto: ISIC_PUERTO=8001 pi/app_kiosco.sh
# Registros: ~/.cache/isic-pi/app.log y chromium.log. Para ponerla en el menú o al encender: pi/instalar_app.sh

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PUERTO="${ISIC_PUERTO:-8000}"
CACHE="$HOME/.cache/isic-pi"
mkdir -p "$CACHE"

# Una sola a la vez: dos apps peleándose la cámara dejan a una con error. El candado lo heredan la app y Chromium.
exec 9> "$CACHE/kiosco.lock"
if ! flock -n 9; then
  echo "La app ya se está abriendo." >&2
  exit 1
fi

for a in "$@"; do
  case "$a" in
    --puerto|--puerto=*) echo "Para cambiar el puerto usa: ISIC_PUERTO=8001 pi/app_kiosco.sh" >&2; exit 2 ;;
  esac
done

if curl -sf --max-time 2 "http://127.0.0.1:$PUERTO/estado" > /dev/null; then
  echo "La app ya está corriendo en el puerto $PUERTO." >&2
  exit 1
fi

cd "$RAIZ" || exit 1
echo "--- $(date '+%F %T') arranca la app" >> "$CACHE/app.log"
"$RAIZ/.venv/bin/python" -u pi/app.py --puerto "$PUERTO" "$@" >> "$CACHE/app.log" 2>&1 &
APP=$!

listo=0
for _ in $(seq 1 120); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PUERTO/estado" > /dev/null; then
    listo=1
    break
  fi
  if ! kill -0 "$APP" 2> /dev/null; then
    echo "La app no arrancó; mira $CACHE/app.log" >&2
    exit 1
  fi
  sleep 0.5
done
if [ "$listo" != 1 ]; then
  echo "La app no respondió en 60 s; mira $CACHE/app.log" >&2
  kill "$APP" 2> /dev/null
  exit 1
fi

NAVEGADOR="$(command -v chromium || command -v chromium-browser)"
if [ -z "$NAVEGADOR" ]; then
  echo "No encuentro Chromium. La app sigue en http://127.0.0.1:$PUERTO" >&2
  wait "$APP"
  exit 0
fi

# Por SSH no hay pantalla: se usa la del escritorio de la Pi.
if [ -z "$WAYLAND_DISPLAY$DISPLAY" ]; then
  RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SOCKET="$(ls "$RUNTIME"/wayland-[0-9] 2> /dev/null | head -n 1)"
  [ -n "$SOCKET" ] && export XDG_RUNTIME_DIR="$RUNTIME" WAYLAND_DISPLAY="${SOCKET##*/}"
fi

# Perfil propio: no toca el Chromium de siempre ni pregunta por contraseñas o por restaurar pestañas.
echo "--- $(date '+%F %T') abre Chromium" >> "$CACHE/chromium.log"
"$NAVEGADOR" --kiosk --noerrdialogs --disable-infobars --no-first-run --disable-session-crashed-bubble \
  --password-store=basic --user-data-dir="$CACHE/chromium" \
  "http://127.0.0.1:$PUERTO/?kiosco" >> "$CACHE/chromium.log" 2>&1 &
NAV=$!

wait -n "$APP" "$NAV"           # cuando uno de los dos termina, se cierra el otro
if kill -0 "$APP" 2> /dev/null; then
  echo "Chromium se cerró; mira $CACHE/chromium.log" >&2
fi
kill "$APP" "$NAV" 2> /dev/null
wait 2> /dev/null
