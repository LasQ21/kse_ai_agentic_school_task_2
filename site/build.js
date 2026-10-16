// Збирання для Vercel: пише public/config.js. Режими:
//   API_BASE_URL задано          -> "live": сайт читає живий API (лише адреса, без токенів);
//   інакше є public/snapshot.json -> "snapshot": сайт читає статичний знімок;
//   інакше                        -> "none": «дані недоступні».
const fs = require("fs");
const path = require("path");
const raw = (process.env.API_BASE_URL || "").trim().replace(/\/+$/, "");
if (raw && !/^https?:\/\//.test(raw)) {
  console.error("API_BASE_URL має починатися з http:// або https://");
  process.exit(1);
}
const hasSnapshot = fs.existsSync(path.join(__dirname, "public", "snapshot.json"));
const cfg = raw ? { mode: "live", apiBase: raw } : hasSnapshot ? { mode: "snapshot", snapshot: "snapshot.json" } : { mode: "none" };
fs.writeFileSync(path.join(__dirname, "public", "config.js"), `window.APP_CONFIG = ${JSON.stringify(cfg)};\n`);
console.log("config.js:", JSON.stringify(cfg));
