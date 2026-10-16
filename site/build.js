// Збирання для Vercel: копіює public/ і пише public/config.js з адресою API зі змінної середовища.
// У фронтенд потрапляє ЛИШЕ адреса API (без токенів). Якщо API_BASE_URL не задано — сайт
// показує «дані недоступні».
const fs = require("fs");
const path = require("path");
const raw = (process.env.API_BASE_URL || "").trim().replace(/\/+$/, "");
if (raw && !/^https?:\/\//.test(raw)) {
  console.error("API_BASE_URL має починатися з http:// або https://");
  process.exit(1);
}
const out = path.join(__dirname, "public", "config.js");
fs.writeFileSync(out, `window.APP_CONFIG = ${JSON.stringify({ apiBase: raw })};\n`);
console.log(raw ? `config.js: apiBase = ${raw}` : "config.js: API_BASE_URL не задано -> «дані недоступні»");
