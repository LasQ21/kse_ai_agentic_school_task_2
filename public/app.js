// Сирена: мапа, картка району, рейтинг, графік точності. Без залежностей.
const NS = "http://www.w3.org/2000/svg";
const REFRESH_MS = 60_000;
const STOPS = ["#1a3550", "#1f4d6b", "#5d8f94", "#d9c78f", "#f2a43a"];   // шкала ризику, t = √p
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent?.append(n);
  return n;
};

const S = { map: null, data: null, byId: new Map(), meta: new Map(), hi: 0, selected: null, hovered: null, first: true };

// ---------- формат (те саме правило округлення, що й у боті: src.bot.fmt_p) ----------
function fmtP(p) {
  if (p == null || !Number.isFinite(p)) return "—";
  const pct = 100 * p;
  if (pct < 1) return "<1%";
  if (pct > 99) return ">99%";
  if (pct < 10) return `${Math.round(pct)}%`;
  return `${Math.min(95, 5 * Math.round(pct / 5))}%`;
}
function fmtDur(h) {
  const m = Math.max(0, Math.round(h * 60)), hh = Math.floor(m / 60), mm = m % 60;
  return hh ? `${hh} год ${mm} хв` : `${mm} хв`;
}
function fmtAge(s) {
  if (s == null) return "";
  return s < 90 ? `${Math.round(s)} с тому` : s < 5400 ? `${Math.round(s / 60)} хв тому` : `${(s / 3600).toFixed(1).replace(".", ",")} год тому`;
}
const clock = new Intl.DateTimeFormat("uk-UA", { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Kyiv" });
const num3 = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3).replace(".", ",");

// ---------- колір ----------
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const RGB = STOPS.map(hex);
function ramp(p) {
  const t = Math.sqrt(Math.min(1, Math.max(0, p))) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
  return `rgb(${RGB[i].map((c, k) => Math.round(c + (RGB[i + 1][k] - c) * f)).join(",")})`;
}

const E = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
function parseId(id) {
  const [o, r] = id.split("|");
  if (r === "*") return { name: o, sub: o === "м. Київ" ? "столиця" : "уся область" };
  return { name: r, sub: o };
}
const hLabel = (i) => `${S.data.horizons[i]} год`;

// ---------- мапа ----------
function buildMap(map) {
  const svg = $("#map");
  svg.setAttribute("viewBox", `0 0 ${map.w} ${map.h}`);
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  const pat = el("pattern", { id: "hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  el("rect", { width: 7, height: 7, fill: "#141f2b" }, pat);
  el("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: "#2a3949", "stroke-width": 2.4 }, pat);
  const clip = el("clipPath", { id: "land" }, defs);
  for (const u of map.units) el("path", { d: u.d }, clip);

  const other = el("g", { class: "other-g" }, svg);
  for (const o of map.other) {
    const p = el("path", { d: o.d, class: "other" }, other);
    p.addEventListener("pointermove", (e) => showTip(e, `<b>${E(o.name)}</b>Без прогнозу: ${E(o.reason)}.`));
    p.addEventListener("pointerleave", hideTip);
  }

  const units = el("g", { class: "units" }, svg);
  const sorted = [...map.units].sort((a, b) => a.cx - b.cx);
  sorted.forEach((u, i) => {
    const wrap = el("g", { class: "uw in", style: `--d:${(u.cx / map.w) * 900 + (i % 5) * 20}ms` }, units);
    const p = el("path", { d: u.d, class: "unit", tabindex: 0, role: "button", "data-id": u.id }, wrap);
    const { name, sub } = parseId(u.id);
    p.setAttribute("aria-label", `${name}, ${sub}`);
    p.addEventListener("pointerenter", () => hover(u.id));
    p.addEventListener("pointermove", (e) => showTip(e, tipHtml(u.id)));
    p.addEventListener("pointerleave", () => { hover(null); hideTip(); });
    p.addEventListener("focus", () => hover(u.id));
    p.addEventListener("blur", () => hover(null));
    p.addEventListener("click", () => select(u.id));
    p.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(u.id); } });
    S.meta.set(u.id, { ...u, node: p });
  });

  // радарний промінь: тонкий віяр, що повільно обертається над сушею
  const cx = map.w * 0.5, cy = map.h * 0.5, R = map.w, sweep = el("g", { "clip-path": "url(#land)", "pointer-events": "none" }, svg);
  const g = el("g", { class: "sweep", style: `transform-origin:${cx}px ${cy}px` }, sweep);
  const N = 16, step = 2.2;
  for (let i = 0; i < N; i++) {
    const a1 = (i * step * Math.PI) / 180, a2 = ((i + 1) * step * Math.PI) / 180;
    el("path", {
      d: `M${cx} ${cy}L${cx + R * Math.cos(a1)} ${cy + R * Math.sin(a1)}A${R} ${R} 0 0 1 ${cx + R * Math.cos(a2)} ${cy + R * Math.sin(a2)}Z`,
      fill: "#e8eef4", "fill-opacity": ((i + 1) / N) * 0.085,
    }, g);
  }

  el("path", { d: map.unitBorders, class: "b-unit" }, svg);
  el("path", { d: map.oblastBorders, class: "b-oblast" }, svg);
  el("path", { d: map.outline, class: "outline", pathLength: 1 }, svg);
  S.hoverEl = el("path", { class: "hover" }, svg);
  S.selEl = el("path", { class: "sel" }, svg);
  S.pulses = el("g", { class: "pulses" }, svg);
}

function paintMap() {
  const { units, horizons } = S.data;
  let active = 0;
  S.pulses.innerHTML = "";
  for (const u of units) {
    const m = S.meta.get(u.id);
    if (!m) continue;
    const on = u.state === "B";
    m.node.classList.toggle("active", on);
    m.node.style.fill = ramp(u.p[S.hi]);
    if (on) {
      active++;
      if (!REDUCED) {
        const c = el("circle", { cx: m.cx, cy: m.cy, r: 18, class: "ripple", style: `--rd:${((m.cx * 7 + m.cy * 3) % 36) / 10}s` }, S.pulses);
        c.dataset.id = u.id;
      }
    }
  }
  setActive(active, units.length);
  void horizons;
}

function setActive(n, total) {
  $("#n-total").textContent = total;
  const out = $("#n-active");
  if (S.first && !REDUCED) {
    const t0 = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / 1100);
      out.textContent = Math.round(n * (1 - (1 - k) ** 3));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } else out.textContent = n;
}

function hover(id) {
  S.hovered = id;
  S.hoverEl.setAttribute("d", id ? S.meta.get(id).d : "");
}

function tipHtml(id) {
  const u = S.byId.get(id), { name, sub } = parseId(id);
  if (!u) return `<b>${E(name)}</b>${E(sub)}`;
  const p = u.p[S.hi];
  const head = `<b>${E(name)}</b><span class="muted">${E(sub)}</span>`;
  const state = u.state === "B" ? `<div class="t-red">Тривога триває ${fmtDur(u.elapsed_h ?? 0)}</div>` : "";
  return `${head}${state}<div><span class="t-p">${fmtP(p)}</span> нова за ${hLabel(S.hi)}</div>`;
}
function showTip(e, html) {
  const tip = $("#tip"), st = $("#stage").getBoundingClientRect();
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX - st.left + 16, y = e.clientY - st.top + 16;
  if (x + w > st.width) x = e.clientX - st.left - w - 16;
  if (y + h > st.height) y = e.clientY - st.top - h - 16;
  tip.style.left = `${Math.max(0, x)}px`;
  tip.style.top = `${Math.max(0, y)}px`;
}
function hideTip() { $("#tip").hidden = true; }

// паралакс: мапа легко нахиляється за курсором
function tilt() {
  if (REDUCED) return;
  const stage = $("#stage"), svg = $("#map");
  stage.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    const r = stage.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
    svg.style.transform = `rotateX(${(-y * 4).toFixed(2)}deg) rotateY(${(x * 5).toFixed(2)}deg)`;
  });
  stage.addEventListener("pointerleave", () => { svg.style.transform = ""; });
}

// ---------- шкала ----------
function buildLegend() {
  $("#legend-bar").style.background = `linear-gradient(90deg, ${STOPS.join(",")})`;
  const t = $("#legend-ticks");
  t.innerHTML = "";
  for (const p of [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]) {
    const s = document.createElement("span");
    s.style.left = `${Math.sqrt(p) * 100}%`;
    s.textContent = `${Math.round(p * 100)}%`;
    t.append(s);
  }
}

// ---------- перемикач горизонту ----------
function buildSeg(horizons) {
  const seg = $("#seg");
  seg.innerHTML = '<span class="thumb"></span>';
  horizons.forEach((h, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.role = "radio";
    b.textContent = `${h} год`;
    b.addEventListener("click", () => setHorizon(i));
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); setHorizon((S.hi + d + horizons.length) % horizons.length); seg.children[S.hi + 1].focus(); }
    });
    seg.append(b);
  });
  syncSeg();
}
function syncSeg() {
  const seg = $("#seg");
  [...seg.querySelectorAll("button")].forEach((b, i) => {
    b.setAttribute("aria-checked", i === S.hi);
    b.tabIndex = i === S.hi ? 0 : -1;
  });
  seg.querySelector(".thumb").style.transform = `translateX(${S.hi * 100}%)`;
}
function setHorizon(i) {
  if (i === S.hi) return;
  S.hi = i;
  syncSeg();
  paintMap();
  renderRank();
  if (S.selected) renderCard(false);
}

// ---------- картка району ----------
function select(id, scroll = false) {
  S.selected = id;
  S.selEl.setAttribute("d", S.meta.get(id).d);
  renderCard(true);
  if (scroll) $("#map-section").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" });
}

function renderCard(animate) {
  const id = S.selected, u = S.byId.get(id), card = $("#card"), { name, sub } = parseId(id);
  if (!u) return;
  const on = u.state === "B";
  const bars = S.data.horizons.map((h, i) => {
    const p = u.p[i];
    return `<li class="${i === S.hi ? "cur" : ""}"><span>${h} год</span><span class="track"><span class="fill" style="--c:${ramp(p)}" data-w="${Math.max(2, Math.sqrt(p) * 100).toFixed(1)}"></span></span><b>${fmtP(p)}</b></li>`;
  }).join("");
  let counts = "";
  if (u.counts) {
    const [a, b, c] = [u.counts.p0[S.hi], u.counts.p1[S.hi], u.counts.p2plus[S.hi]];
    counts = `<p class="cnt">Скільки нових тривог за ${hLabel(S.hi)}
      <span class="cnt-bar"><i style="flex-grow:${a};background:#2f4a63"></i><i style="flex-grow:${b};background:#d9c78f"></i><i style="flex-grow:${c};background:#f2a43a"></i></span>
      <span class="cnt-key"><span><i style="background:#2f4a63"></i>жодної ${fmtP(a)}</span><span><i style="background:#d9c78f"></i>одна ${fmtP(b)}</span><span><i style="background:#f2a43a"></i>дві й більше ${fmtP(c)}</span></span></p>`;
  }
  const notes = [];
  if (on) notes.push(`Тривога триває ${fmtDur(u.elapsed_h ?? 0)}. Нова може початися лише після відбою — ймовірності стосуються її.`);
  if (u.climate) notes.push("Захід: модель тут не перевершила середній рівень, тож показано кліматичну оцінку.");
  card.innerHTML = `<div class="card-head"><div><h3>${E(name)}</h3><p class="card-sub">${E(sub)}</p></div>
    <span class="badge ${on ? "on" : ""}">${on ? "Тривога" : "Тривоги немає"}</span></div>
    <p class="cnt" style="margin-top:14px">Нова тривога почнеться протягом</p>
    <ul class="bars" style="margin-top:8px">${bars}</ul>${counts}
    ${notes.map((n) => `<p class="card-note">${n}</p>`).join("")}`;
  if (animate) { card.classList.remove("in"); void card.offsetWidth; card.classList.add("in"); }
  requestAnimationFrame(() => requestAnimationFrame(() => {
    card.querySelectorAll(".fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; });
  }));
}

// ---------- пошук ----------
function initSearch() {
  const q = $("#q"), list = $("#q-list");
  const norm = (s) => s.toLowerCase().replace(/[’`ʼ]/g, "'");
  let items = [], cur = -1;
  const close = () => { list.hidden = true; q.setAttribute("aria-expanded", "false"); cur = -1; };
  const paint = () => [...list.children].forEach((li, i) => li.setAttribute("aria-selected", i === cur));
  const pick = (id) => { close(); q.value = ""; select(id); };
  q.addEventListener("input", () => {
    const terms = norm(q.value).split(/\s+/).filter(Boolean);
    items = terms.length ? [...S.meta.keys()].filter((id) => { const { name, sub } = parseId(id); const t = norm(`${name} ${sub}`); return terms.every((w) => t.includes(w)); }).slice(0, 8) : [];
    list.innerHTML = "";
    for (const id of items) {
      const { name, sub } = parseId(id), li = document.createElement("li");
      li.role = "option";
      li.innerHTML = `${E(name)}<span>${E(sub)}</span>`;
      li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(id); });
      list.append(li);
    }
    list.hidden = !items.length;
    q.setAttribute("aria-expanded", String(!!items.length));
    cur = -1;
  });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!items.length) return;
      e.preventDefault();
      cur = (cur + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      paint();
    } else if (e.key === "Enter" && items.length) { e.preventDefault(); pick(items[Math.max(cur, 0)]); }
    else if (e.key === "Escape") close();
  });
  q.addEventListener("blur", close);
}

// ---------- рейтинг ----------
function renderRank() {
  const list = $("#rank-list"), hi = S.hi;
  $("#rank-h").textContent = hLabel(hi);
  const top = [...S.data.units].sort((a, b) => b.p[hi] - a.p[hi]).slice(0, 12);
  const old = new Map([...list.children].map((li) => [li.dataset.id, li]));
  const first = new Map([...old].map(([id, li]) => [id, li.getBoundingClientRect().top]));
  const frag = document.createDocumentFragment();
  top.forEach((u, i) => {
    let li = old.get(u.id);
    const { name, sub } = parseId(u.id), on = u.state === "B", isNew = !li;
    if (isNew) {
      li = document.createElement("li");
      li.dataset.id = u.id;
      li.innerHTML = '<button class="rank-item" type="button"><span class="r-pos"></span><span class="r-name"></span><span class="r-val"></span></button><span class="r-bar"></span>';
      li.firstChild.addEventListener("click", () => select(u.id, true));
    }
    li.querySelector(".r-pos").textContent = i + 1;
    li.querySelector(".r-name").innerHTML = `${on ? '<i class="dot" title="Тривога триває"></i>' : ""}${E(name)}<small>${E(sub)}</small>`;
    li.querySelector(".r-val").textContent = fmtP(u.p[hi]);
    const bar = li.querySelector(".r-bar");
    bar.style.setProperty("--c", ramp(u.p[hi]));
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${Math.sqrt(u.p[hi]) * 100}%`; }));
    if (isNew && !S.first && !REDUCED) li.animate([{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], { duration: 450, easing: "ease-out" });
    frag.append(li);
  });
  list.replaceChildren(frag);
  if (REDUCED) return;
  top.forEach((u) => {
    const li = [...list.children].find((n) => n.dataset.id === u.id);
    if (first.has(u.id)) {
      const dy = first.get(u.id) - li.getBoundingClientRect().top;
      if (dy) li.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 650, easing: "cubic-bezier(.2,.8,.2,1)" });
    }
  });
}

// ---------- дані ----------
async function load() {
  const r = await fetch("/api/forecast", { cache: "no-store" });
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

function applyData(d) {
  S.data = d;
  S.byId = new Map(d.units.map((u) => [u.id, u]));
  if (S.first) buildSeg(d.horizons);
  paintMap();
  renderRank();
  if (S.selected) renderCard(false);
  updateStatus();
  $("#map-msg").hidden = true;
  S.first = false;
}

function updateStatus() {
  const d = S.data, ban = $("#banner");
  if (!d) return;
  const asOf = d.as_of ? clock.format(new Date(d.as_of)) : "—";
  const age = d.data_last_update ? (Date.now() - new Date(d.data_last_update)) / 1000 : null;
  $("#live-time").textContent = d.stale
    ? `Дані застаріли, прогноз станом на ${asOf} (Київ)`
    : `Оновлено о ${asOf} (Київ)${age != null ? ` · дані ${fmtAge(age)}` : ""}`;
  ban.hidden = !d.stale && !S.failed;
  if (d.stale) ban.textContent = "Живе джерело не відповідає понад 10 хвилин. Ми не знаємо, що сталося після останнього оновлення, тому показуємо прогноз станом на нього.";
  else if (S.failed) ban.textContent = "Не вдалося оновити прогноз. Показано останній отриманий; спробуємо ще раз за хвилину.";
}

async function refresh() {
  try {
    applyData(await load());
    S.failed = false;
    updateStatus();
  } catch {
    S.failed = true;
    if (!S.data) {
      const m = $("#map-msg");
      m.hidden = false;
      m.innerHTML = '<p>Прогноз тимчасово недоступний.<br>Мапа з’явиться, щойно сервіс відповість.</p><button type="button">Спробувати ще раз</button>';
      m.querySelector("button").addEventListener("click", refresh);
      $("#live-time").textContent = "Не вдалося завантажити прогноз";
    } else updateStatus();
  }
}

// ---------- точність ----------
const CHART = {
  series: [
    { key: "val", label: "Validation (15.06–27.07.2026)", c: "#5d8f94" },
    { key: "ext", label: "Зовнішня: alerts.in.ua, інше визначення тривоги", c: "#f2a43a" },
    { key: "test", label: "Test (01–28.08.2026)", c: "#d9c78f" },
  ],
  rows: [
    { h: 1, val: [0.093, 0.082, 0.102], ext: [0.160, 0.145, 0.173], test: [0.101, 0.087, 0.115] },
    { h: 3, val: [0.095, 0.077, 0.112], ext: [0.261, 0.227, 0.292], test: [0.113, 0.092, 0.135] },
    { h: 6, val: [0.101, 0.079, 0.122], ext: [0.328, 0.283, 0.372], test: [0.115, 0.082, 0.148] },
    { h: 24, val: [0.061, 0.019, 0.102], ext: [0.439, 0.369, 0.505], test: [0.091, 0.020, 0.170] },
  ],
};
function buildChart() {
  const W = 1000, L = 76, R = 40, X0 = L, X1 = W - R, max = 0.55, rowH = 96, top = 8, H = top + rowH * 4 + 34;
  const x = (v) => X0 + (v / max) * (X1 - X0);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  for (let v = 0; v <= max + 1e-9; v += 0.1) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 30, class: v === 0 ? "zero" : "grid" }, svg);
    const t = el("text", { x: x(v), y: H - 10, "text-anchor": "middle" }, svg);
    t.textContent = v.toFixed(1).replace(".", ",");
  }
  const zl = el("text", { x: x(0) + 6, y: top + 11 }, svg);
  zl.textContent = "як базовий прогноз";
  CHART.rows.forEach((row, ri) => {
    const cy = top + ri * rowH + rowH / 2;
    el("text", { x: 0, y: cy + 5, class: "row-label" }, svg).textContent = `${row.h} год`;
    CHART.series.forEach((s, si) => {
      const [v, lo, hi] = row[s.key], y = cy + (si - 1) * 24;
      const g = el("g", { style: `--c:${s.c};--cd:${(ri * 3 + si) * 90}ms` }, svg);
      el("line", { x1: x(lo), x2: x(hi), y1: y, y2: y, class: "wh" }, g);
      el("circle", { cx: x(v), cy: y, r: 6.5, class: "dot" }, g);
      el("text", { x: x(hi) + 10, y: y + 4 }, g).textContent = num3(v);
    });
  });
  $("#chart").append(svg);
  $("#chart-legend").innerHTML = CHART.series.map((s) => `<span style="--c:${s.c}"><i></i>${s.label}</span>`).join("");
  const fmt = (a) => `${num3(a[0])} [${num3(a[1])}; ${num3(a[2])}]`;
  $("#tbl").innerHTML = `<table><thead><tr><th>Горизонт</th>${CHART.series.map((s) => `<th>${s.label}</th>`).join("")}</tr></thead><tbody>${
    CHART.rows.map((r) => `<tr><th>${r.h} год</th>${CHART.series.map((s) => `<td>${fmt(r[s.key])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function reveals() {
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("seen"); io.unobserve(e.target); } }), { threshold: 0.35 });
  io.observe($("#steps"));
  io.observe($("#chart"));
}

// ---------- старт ----------
async function boot() {
  buildLegend();
  buildChart();
  reveals();
  tilt();
  try {
    S.map = await (await fetch("geo/map.json")).json();
  } catch {
    $("#map-msg").hidden = false;
    $("#map-msg").textContent = "Не вдалося завантажити мапу районів. Оновіть сторінку.";
    return;
  }
  buildMap(S.map);
  initSearch();
  await refresh();
  setInterval(refresh, REFRESH_MS);
  setInterval(updateStatus, 15_000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
}
boot();
