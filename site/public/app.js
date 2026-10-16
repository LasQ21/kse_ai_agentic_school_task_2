// Сайт лише читає публічний API прогнозу (GET). Адреса API — з config.js (збирається зі змінної
// середовища API_BASE_URL на Vercel). Токенів тут немає.
(function () {
  "use strict";
  const CFG = window.APP_CONFIG || {};
  const MODE = CFG.mode || (CFG.apiBase ? "live" : "none");   // live | snapshot | none
  const API = (CFG.apiBase || "").replace(/\/+$/, "");
  const REFRESH_MS = 2 * 60 * 1000;
  const TIMEOUT_MS = 12000;
  const KYIV = new Intl.DateTimeFormat("uk-UA", { timeZone: "Europe/Kyiv", day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" });
  const $ = (id) => document.getElementById(id);
  const collator = new Intl.Collator("uk");

  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  }

  function ago(seconds) {
    if (seconds == null) return "невідомо";
    const s = Math.max(0, Math.round(seconds));
    if (s < 90) return `${s} с тому`;
    const m = Math.round(s / 60);
    if (m < 90) return `${m} хв тому`;
    const h = Math.round(m / 60);
    return h < 48 ? `${h} год тому` : `${Math.round(h / 24)} дн тому`;
  }

  function bucket(p) {
    return p < 0.05 ? 0 : p < 0.15 ? 1 : p < 0.3 ? 2 : p < 0.5 ? 3 : p < 0.75 ? 4 : 5;
  }

  async function getJSON(path) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      const r = await fetch(API + path, { signal: ctl.signal, headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally { clearTimeout(t); }
  }

  function unavailable(reason) {
    // Ніяких чисел, якщо даних немає: сайт не вдає, що він живий.
    $("forecast").hidden = true;
    $("quality").hidden = true;
    $("stale").hidden = true;
    const s = $("state");
    s.className = "state down";
    s.replaceChildren(
      el("p", { class: "state-line", text: "Дані недоступні." }),
      el("p", { class: "state-sub", text: reason }),
      el("p", { class: "state-sub", text: "Прогноз не показуємо, поки сервер прогнозу не відповідає. Орієнтуйтесь на офіційні сигнали." })
    );
    $("updated").textContent = `Остання спроба: ${KYIV.format(new Date())}.`;
  }

  function renderState(d) {
    const active = d.items.filter((i) => i.active_now).length;
    const s = $("state");
    s.className = "state";
    s.replaceChildren(
      MODE === "snapshot"
        ? el("p", { class: "state-line" },
            "На момент знімка тривога була у ", el("b", { text: String(active) }),
            ` з ${d.items.length} районів і міст. Дані тоді були оновлені `, el("b", { text: ago(d.data_age_s).replace(" тому", " до знімка") }), ".")
        : d.stale
        ? el("p", { class: "state-line" },
            `Станом на ${KYIV.format(new Date(d.as_of))} тривога була у `, el("b", { text: String(active) }),
            ` з ${d.items.length} районів і міст. Що зараз — невідомо: дані оновлено `, el("b", { text: ago(d.data_age_s) }), ".")
        : el("p", { class: "state-line" },
            "Тривога зараз у ", el("b", { text: String(active) }), ` з ${d.items.length} районів і міст. `,
            "Дані оновлено ", el("b", { text: ago(d.data_age_s) }), "."),
      el("p", { class: "state-sub", text:
        `Прогноз станом на ${KYIV.format(new Date(d.as_of))} (Київ). Джерело даних — alerts.in.ua; ` +
        "визначення тривоги там інше, ніж у навчанні моделі." })
    );
    const b = $("stale");
    if (d.stale) {
      b.replaceChildren(
        el("span", { class: "icon", "aria-hidden": "true" }),
        el("p", {}, el("b", { text: "Дані застаріли. " }),
          `Останнє оновлення — ${ago(d.data_age_s)} (${d.data_last_update ? KYIV.format(new Date(d.data_last_update)) : "невідомо"}). ` +
          "Ймовірності нижче пораховано станом на той момент, а не на зараз. Не покладайтеся на них.")
      );
      // Сталий SVG без даних з API — innerHTML тут безпечний.
      b.querySelector(".icon").innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M12 2 1 21h22L12 2z" fill="#fab219" stroke="#4a3500" stroke-width="1.5"/><path d="M12 9v5" stroke="#4a3500" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17.3" r="1.2" fill="#4a3500"/></svg>';
      b.hidden = false;
    } else {
      b.hidden = true;
    }
  }

  function riskCell(p) {
    return el("td", { class: `r r${bucket(p)}`, text: `${Math.round(p * 100)}%` });
  }

  let staleNow = false;

  function renderList(items, query) {
    const q = query.trim().toLocaleLowerCase("uk");
    const byOblast = new Map();
    for (const it of items) {
      if (q && !(`${it.oblast} ${it.raion || ""}`.toLocaleLowerCase("uk").includes(q))) continue;
      if (!byOblast.has(it.oblast)) byOblast.set(it.oblast, []);
      byOblast.get(it.oblast).push(it);
    }
    const list = $("list");
    list.replaceChildren();
    if (!byOblast.size) { list.append(el("p", { class: "empty", text: "Нічого не знайдено. Спробуйте іншу назву." })); return; }
    for (const oblast of [...byOblast.keys()].sort(collator.compare)) {
      const rows = byOblast.get(oblast).sort((a, b) => collator.compare(a.raion || "", b.raion || ""));
      const climate = rows[0].model === "climate";
      const cap = el("caption", {}, oblast,
        climate ? el("span", { class: "tag", text: "Базовий кліматичний рівень: частота тривог для цієї години тижня, а не прогноз поточної ситуації" }) : null);
      const head = el("thead", {}, el("tr", {},
        el("th", { scope: "col", text: "Район" }), el("th", { scope: "col", class: "h", text: "1 год" }),
        el("th", { scope: "col", class: "h", text: "3 год" }), el("th", { scope: "col", class: "h", text: "6 год" })));
      const body = el("tbody");
      for (const it of rows) {
        const name = el("th", { scope: "row" }, it.raion || "вся область",
          it.active_now ? el("span", { class: "now", text: MODE === "snapshot" ? "тривога на момент знімка"
            : staleNow ? "тривога на момент оновлення" : "тривога зараз" }) : null);
        body.append(el("tr", {}, name, riskCell(it.probabilities["1"]), riskCell(it.probabilities["3"]), riskCell(it.probabilities["6"])));
      }
      list.append(el("div", { class: "oblast" + (climate ? " climate" : "") }, el("table", {}, cap, head, body)));
    }
  }

  function renderExcluded(excluded) {
    $("excluded").replaceChildren(...excluded.map((e) =>
      el("li", {}, el("b", { text: e.oblast }), ` — без прогнозу: ${e.reason.replace(/^прогнозу немає:\s*/, "")}.`)));
  }

  function renderQuality(q) {
    const parts = [
      ["test", "Test: 1–28 серпня 2026, офіційний канал. Цей період ми переглядали двічі під час розробки."],
      ["external", "Зовнішня перевірка: 22 вересня – 15 жовтня 2026, дані alerts.in.ua з іншим визначенням тривоги. Це не test."],
    ];
    const box = $("quality-tables");
    box.replaceChildren();
    for (const [key, title] of parts) {
      const rows = q[key] || [];
      if (!rows.length) continue;
      const t = el("table", {}, el("caption", { text: title }),
        el("thead", {}, el("tr", {}, ...["Група", "H", "Brier прогнозу", "Brier клімату", "ROC-AUC", "Частка тривог"].map((h) => el("th", { scope: "col", text: h })))));
      const body = el("tbody");
      for (const r of rows) {
        const g = r.group === "western" ? "Захід (базовий кліматичний рівень)" : "Решта (модель GBM)";
        body.append(el("tr", {}, el("td", { text: g }), el("td", { text: `${r.horizon_h} год` }),
          el("td", { text: r.brier.toFixed(4) }),
          el("td", { text: r.group === "western" ? "те саме" : r.brier_climate.toFixed(4) }),
          el("td", { text: r.roc_auc == null ? "—" : r.roc_auc.toFixed(2) }),
          el("td", { text: `${(r.base_rate * 100).toFixed(1)}%` })));
      }
      t.append(body);
      box.append(t);
    }
    box.append(el("p", { text:
      "На заході кліматичний рівень ледве відрізняє години з тривогою від годин без неї (ROC-AUC близько 0.5–0.6). " +
      "Решта: модель стабільно краща за клімат, але ймовірності для alerts.in.ua не відкалібровані." }));
    $("quality").hidden = false;
  }

  let lastForecast = null;

  function showSnapshotBanner(snap) {
    const utc = new Intl.DateTimeFormat("uk-UA", { timeZone: "UTC", day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" });
    const kyiv = new Intl.DateTimeFormat("uk-UA", { timeZone: "Europe/Kyiv", day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" });
    const t = new Date(snap.snapshot_at_utc);
    const b = $("snapshot");
    b.replaceChildren(el("p", {}, `Знімок на ${kyiv.format(t)} (Київ). Не живі дані.`,
      el("span", { text: `${utc.format(t)} UTC. Сторінка не оновлюється: це збережений стан прогнозу на той момент, ` +
        "а не поточна ситуація. Орієнтуйтесь на офіційні сигнали." })));
    b.hidden = false;
  }

  async function loadSnapshot() {
    try {
      const r = await fetch(CFG.snapshot || "snapshot.json", { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const snap = await r.json();
      if (snap.kind !== "snapshot") throw new Error("невідомий формат");
      showSnapshotBanner(snap);
      lastForecast = snap;
      staleNow = !!snap.stale;
      renderState(snap);
      renderList(snap.items, $("q").value);
      renderExcluded(snap.excluded);
      $("forecast").hidden = false;
      renderQuality(snap.quality);
      $("updated").textContent = `Знімок зроблено ${KYIV.format(new Date(snap.snapshot_at_utc))} (Київ). Сторінка не оновлюється автоматично.`;
    } catch (e) {
      unavailable(`Не вдалося прочитати знімок прогнозу (${e.message}).`);
    }
  }

  async function refresh() {
    if (!API) { unavailable("Сервер прогнозу ще не розгорнуто: адресу API не налаштовано."); return; }
    try {
      const [forecast, regions] = await Promise.all([getJSON("/forecast"), getJSON("/regions")]);
      lastForecast = forecast;
      staleNow = !!forecast.stale;
      renderState(forecast);
      renderList(forecast.items, $("q").value);
      renderExcluded(regions.excluded);
      $("forecast").hidden = false;
      getJSON("/quality").then(renderQuality).catch(() => { $("quality").hidden = true; });
      $("updated").textContent = `Сторінку оновлено: ${KYIV.format(new Date())}. Автооновлення раз на 2 хв.`;
    } catch (e) {
      unavailable(`Сервер прогнозу не відповідає (${e.name === "AbortError" ? "час очікування вичерпано" : e.message}).`);
    }
  }

  $("q").addEventListener("input", () => { if (lastForecast) renderList(lastForecast.items, $("q").value); });
  if (MODE === "snapshot") {
    loadSnapshot();
  } else {
    refresh();
    setInterval(refresh, REFRESH_MS);
  }
})();
