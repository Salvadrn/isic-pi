#!/bin/bash
# Pone la app de pantalla táctil en el menú de la Pi y, si quieres, hace que se abra sola al encender.
# No necesita sudo: todo queda en tu usuario.
#
# Uso:
#   pi/instalar_app.sh                 # ícono «Clasificador de piel» en el menú (y en el escritorio)
#   pi/instalar_app.sh --al-encender   # además, se abre sola al prender la Pi
#   pi/instalar_app.sh --quitar        # quita el ícono y el arranque automático

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
MENU="$HOME/.local/share/applications/isic-pi.desktop"
ESCRITORIO="$HOME/Desktop/isic-pi.desktop"
AUTOSTART="$HOME/.config/labwc/autostart"
MARCA="# isic-pi: app de pantalla táctil"

if [ "$1" = "--quitar" ]; then
  rm -f "$MENU" "$ESCRITORIO"
  [ -f "$AUTOSTART" ] && sed -i "\|$MARCA|d" "$AUTOSTART"
  echo "Listo: quité el ícono y el arranque automático."
  exit 0
fi

[ -x "$RAIZ/.venv/bin/python" ] || { echo "Falta el entorno $RAIZ/.venv (ver el README, «Instalar en la Pi»)." >&2; exit 1; }
command -v chromium > /dev/null || command -v chromium-browser > /dev/null \
  || echo "Aviso: no encontré Chromium; la app abrirá, pero sin pantalla completa." >&2
chmod +x "$RAIZ/pi/app_kiosco.sh"

mkdir -p "$(dirname "$MENU")"
cat > "$MENU" << EOF
[Desktop Entry]
Type=Application
Name=Clasificador de piel
Comment=isic-pi: toma una foto y muestra las probabilidades (no es un diagnóstico)
Exec=$RAIZ/pi/app_kiosco.sh
Icon=camera-photo
Terminal=false
Categories=Education;Science;
EOF
if [ -d "$HOME/Desktop" ]; then
  cp "$MENU" "$ESCRITORIO"
  chmod +x "$ESCRITORIO"
fi
echo "Listo: «Clasificador de piel» está en el menú (Educación) y en el escritorio."

if [ "$1" = "--al-encender" ]; then
  mkdir -p "$(dirname "$AUTOSTART")"
  if ! grep -qF "$MARCA" "$AUTOSTART" 2> /dev/null; then
    echo "$RAIZ/pi/app_kiosco.sh &  $MARCA" >> "$AUTOSTART"
  fi
  echo "Listo: se abre sola al encender. Para quitarlo: pi/instalar_app.sh --quitar"
fi
