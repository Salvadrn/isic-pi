"use strict";
// Lógica de la app táctil. Escala el escenario de 1024×600 a la pantalla, acomoda el cuadro guía sobre la imagen
// en vivo, toma la foto y muestra las probabilidades. Habla con pi/app.py.

const $ = (id) => document.getElementById(id);
const ANCHO = 1024, ALTO = 600;
let guia = 0.8, enVivo = false, ocupado = false, ultimaId = null, imagenes = -1, temporizadorAviso = null, presion = null;

function escalar() {
  const s = Math.min(innerWidth / ANCHO, innerHeight / ALTO);
  $("escenario").style.transform = `translate(${(innerWidth - ANCHO * s) / 2}px, ${(innerHeight - ALTO * s) / 2}px) scale(${s})`;
}

// ---------- Imagen en vivo y cuadro guía ----------
// El cuadro guía es exactamente lo que se clasifica: el cuadrado central de lado `guia` × el lado corto de la imagen.
function acomodar() {
  const v = $("video");
  const w = v.naturalWidth, h = v.naturalHeight;
  if (!w || !h) return;
  const s = Math.min(ANCHO / w, ALTO / h);
  const bw = w * s, bh = h * s, bx = (ANCHO - bw) / 2, by = (ALTO - bh) / 2;
  Object.assign(v.style, { left: `${bx}px`, top: `${by}px`, width: `${bw}px`, height: `${bh}px` });
  const lado = guia * Math.min(bw, bh), gx = (ANCHO - lado) / 2, gy = (ALTO - lado) / 2;
  $("sombra").setAttribute("d", `M0 0H${ANCHO}V${ALTO}H0Z M${gx} ${gy}V${gy + lado}H${gx + lado}V${gy}Z`);
  $("reticula").setAttribute("transform", `translate(${gx} ${gy}) scale(${lado / 480})`);
  const d = $("destello");
  d.setAttribute("x", gx); d.setAttribute("y", gy); d.setAttribute("width", lado); d.setAttribute("height", lado);
  $("instruccion").style.top = `${Math.max(4, gy - 46)}px`;
  $("pie-guia").style.top = `${Math.min(ALTO - 40, gy + lado + 12)}px`;
}

function encenderVideo() {
  enVivo = true;
  $("video").src = `/video?t=${Date.now()}`;
}

function apagarVideo() {
  enVivo = false;
  $("video").removeAttribute("src");
}

// ---------- Estado de la cámara y del modelo ----------
async function leerEstado() {
  try {
    const r = await fetch("/estado", { cache: "no-store" });
    const d = await r.json();
    guia = d.guia;
    $("nombre-camara").textContent = d.camara.nombre;
    const llegan = d.camara.imagenes > imagenes && imagenes >= 0;
    $("punto-camara").className = "punto " + (d.camara.error ? "error" : llegan || (imagenes < 0 && d.camara.imagenes > 0) ? "ok" : "alerta");
    imagenes = d.camara.imagenes;
    const sin = $("sin-imagen");
    sin.hidden = !(d.camara.error || d.camara.imagenes === 0);
    sin.textContent = d.camara.error || "Esperando la imagen de la cámara…";
    $("modelo-nombre").textContent = d.modelo.isic ? "ISIC 2019" : "Objetos";
    $("modelo-clases").textContent = d.modelo.clases.toLocaleString("es-MX");
    $("modelo-entrada").textContent = `${d.modelo.entrada} × ${d.modelo.entrada}`;
    if (d.ms != null) $("inferencia").textContent = `${d.ms.toFixed(1)} ms`;
  } catch {
    $("punto-camara").className = "punto error";
  }
}

// ---------- Tomar la foto ----------
function destello() {
  const d = $("destello");
  d.style.transition = "none";
  d.style.opacity = "0.45";
  requestAnimationFrame(() => requestAnimationFrame(() => { d.style.transition = ""; d.style.opacity = "0"; }));
}

async function tomar() {
  if (ocupado) return;
  ocupado = true;
  $("tomar").classList.add("ocupado");
  destello();
  try {
    const r = await fetch("/tomar", { method: "POST" });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "No se pudo tomar la foto.");
    await mostrar(d);
  } catch (e) {
    avisar(e.message, true);
  } finally {
    ocupado = false;
    $("tomar").classList.remove("ocupado");
  }
}

function pct(p) {
  const v = Math.round(p * 100);
  return v === 0 && p > 0 ? "<1" : String(v);
}

function ajustar(el, maximo, minimo) {
  let t = maximo;
  el.style.fontSize = `${t}px`;
  while (el.scrollWidth > el.clientWidth && t > minimo) {
    t -= 2;
    el.style.fontSize = `${t}px`;
  }
}

function fila(rango, o) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="num rango">${rango}</span><span class="nombre"></span><span class="num prob">${pct(o.prob)} %</span>`;
  li.querySelector(".nombre").textContent = o.nombre;
  return li;
}

// Escala de 0 a 100 con una aguja por cada una de las 3 clases más probables (la 1 blanca y más larga).
function dibujarEscala(top) {
  const x0 = 16, x1 = 332, base = 74, X = (p) => x0 + (x1 - x0) * p;
  let s = `<line x1="${x0}" y1="${base}" x2="${x1}" y2="${base}" stroke="#666666" stroke-width="1.5"/>`;
  for (let k = 0; k <= 10; k++) {
    const mayor = k % 5 === 0, x = X(k / 10);
    s += `<line x1="${x}" y1="${base - (mayor ? 14 : 8)}" x2="${x}" y2="${base}" stroke="${mayor ? "#9A9A9A" : "#666666"}" stroke-width="2"/>`;
  }
  for (const [p, txt] of [[0, "0"], [0.5, "50"], [1, "100"]]) {
    s += `<text x="${X(p)}" y="104" text-anchor="middle" class="num" fill="#666666" font-size="20">${txt}</text>`;
  }
  // Etiquetas 1, 2, 3 sin encimarse: se separan al menos 22 unidades, de izquierda a derecha.
  const agujas = top.map((o, i) => ({ i, x: X(Math.min(1, Math.max(0, o.prob))) }));
  const etiquetas = [...agujas].sort((a, b) => a.x - b.x);
  etiquetas.forEach((e, k) => { e.lx = k === 0 ? e.x : Math.max(e.x, etiquetas[k - 1].lx + 22); });
  const exceso = Math.max(0, etiquetas[etiquetas.length - 1].lx - (x1 + 8));
  etiquetas.forEach((e) => { e.lx -= exceso; });
  for (const a of agujas.sort((p, q) => q.i - p.i)) {
    const primera = a.i === 0, color = primera ? "#F0F0F0" : "#9A9A9A";
    s += `<line x1="${a.x}" y1="${primera ? 26 : 38}" x2="${a.x}" y2="${base + (primera ? 10 : 0)}" stroke="${color}" stroke-width="${primera ? 3 : 2}"/>`;
    s += `<text x="${a.lx}" y="${primera ? 18 : 30}" text-anchor="middle" class="num" fill="${color}" font-size="20">${a.i + 1}</text>`;
  }
  $("escala").innerHTML = s;
}

async function mostrar(d) {
  ultimaId = d.id;
  const img = $("recorte");
  img.src = `/recorte/${d.id}.jpg`;
  await img.decode().catch(() => {});
  const t = d.top[0];
  $("codigo").textContent = d.isic ? t.codigo : "";
  $("clase").textContent = t.nombre;
  $("comun").textContent = t.comun;
  const cifra = pct(t.prob);
  $("porcentaje").textContent = cifra;
  $("porcentaje").style.fontSize = cifra.length > 2 ? "80px" : "";
  dibujarEscala(d.top);
  $("otras").replaceChildren(...d.top.slice(1).map((o, k) => fila(k + 2, o)));
  $("revisar").hidden = !d.revisar;
  $("poco-seguro").hidden = d.seguro;
  $("de-prueba").hidden = d.isic;
  $("guardar").classList.remove("guardada");
  $("guardar-texto").textContent = "Guardar";
  $("inferencia").textContent = `${d.ms.toFixed(1)} ms`;
  cambiar("resultado");
  ajustar($("clase"), 60, 36);
}

function cambiar(nombre) {
  for (const p of document.querySelectorAll(".pantalla")) p.classList.toggle("activa", p.id === nombre);
  if (nombre === "camara") encenderVideo();
  else apagarVideo();
}

async function guardar() {
  const g = $("guardar");
  if (g.classList.contains("guardada") || !ultimaId) return;
  try {
    const r = await fetch("/guardar", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: ultimaId }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "No se pudo guardar.");
    g.classList.add("guardada");
    $("guardar-texto").textContent = "Guardada";
  } catch (e) {
    avisar(e.message, true);
  }
}

function avisar(texto, error = false) {
  const a = $("aviso");
  a.textContent = texto;
  a.classList.toggle("error", error);
  a.hidden = false;
  clearTimeout(temporizadorAviso);
  temporizadorAviso = setTimeout(() => { a.hidden = true; }, 4000);
}

// ---------- Salir (mantén presionado el estado de la cámara, o Esc con teclado) ----------
function abrirSalir() { $("dialogo-salir").hidden = false; }

async function salir() {
  $("dialogo-salir").hidden = true;
  try {
    const r = await fetch("/salir", { method: "POST" });
    if (!r.ok) throw new Error((await r.json()).error || "No se pudo salir.");
    apagarVideo();
    avisar("La app se cerró.");
  } catch (e) {
    avisar(e.message, true);
  }
}

// ---------- Arranque ----------
if (new URLSearchParams(location.search).has("kiosco")) document.body.classList.add("kiosco");
escalar();
addEventListener("resize", escalar);
document.addEventListener("contextmenu", (e) => e.preventDefault());
$("video").addEventListener("load", acomodar);
$("video").addEventListener("error", () => { if (enVivo) setTimeout(() => { if (enVivo) encenderVideo(); }, 1500); });
setInterval(() => { if (enVivo) acomodar(); }, 1000);
$("tomar").addEventListener("click", tomar);
$("otra").addEventListener("click", () => cambiar("camara"));
$("guardar").addEventListener("click", guardar);
$("salir-no").addEventListener("click", () => { $("dialogo-salir").hidden = true; });
$("salir-si").addEventListener("click", salir);
const estado = $("estado-camara");
estado.addEventListener("pointerdown", () => { presion = setTimeout(abrirSalir, 1200); });
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) estado.addEventListener(ev, () => clearTimeout(presion));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") abrirSalir();
  else if (e.key === " " && $("camara").classList.contains("activa")) { e.preventDefault(); tomar(); }
});
leerEstado();
setInterval(leerEstado, 2000);
encenderVideo();
