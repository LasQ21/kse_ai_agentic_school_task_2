"""Векторна мапа районів для сайту: UNOSAT (шар «Raion») -> public/geo/map.json.

Межі завантажуються з публічного ArcGIS-сервісу UNOSAT (стан 2023, 136 нових районів) і приводяться до
одиниць моделі (models/service_model_meta.json: 105 одиниць). Назви районів у моделі новіші за межі
(перейменування 2024), тому є таблиця RENAMED. Запорізька, Тернопільська області й м. Київ — одна
одиниця на кілька районів: їхні поліґони склеюються в один <path>, а внутрішні межі не малюються.

Результат: проєкція (рівнопроміжна з cos φ), спрощення Дугласа–Пекера, три шари ліній:
межі одиниць, межі областей, зовнішній контур. Шлях до кешу: --cache (щоб не качати щоразу).
Запуск: python scripts/build_map.py [--cache /шлях/raions.geojson]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "geo" / "map.json"
META = ROOT / "models" / "service_model_meta.json"
SRC = ("https://unosatgis.cern.ch/rapidmapping/rest/services/FL20230606UKR/AdminLevel_UKR/MapServer/4/query"
       "?where=1%3D1&outFields=ADM2_UA,ADM1_UA,ADM2_PCODE&outSR=4326&f=geojson&returnGeometry=true"
       "&geometryPrecision=3")
WIDTH = 1000.0
TOL = 0.45          # допуск спрощення в одиницях viewBox (≈ піксель при ширині 1000)
RENAMED = {          # (область, район у межах 2023) -> район у моделі
    ("Дніпропетровська", "Новомосковський"): "Самарівський",
    ("Львівська", "Червоноградський"): "Шептицький",
    ("Харківська", "Красноградський"): "Берестинський",
    ("Дніпропетровська", "Кам’янський"): "Кам'янський",
}
OBLAST_AS_UNIT = {"Запорізька", "Тернопільська"}   # область -> одна одиниця «Область|*»
REASONS = {
    "Донецька": "райони майже весь час під тривогою й не відрізняються між собою",
    "Луганська": "окупована територія",
    "Автономна Республіка Крим": "окупована територія",
    "Севастополь": "окупована територія",
}


def unit_of(oblast: str, raion: str, units: set[str]) -> str | None:
    if oblast == "Київ":
        return "м. Київ|*"
    if oblast in OBLAST_AS_UNIT:
        u = f"{oblast} область|*"
    else:
        raion = RENAMED.get((oblast, raion), raion)
        u = f"{oblast} область|{raion} район"
    return u if u in units else None


def dp(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Дуглас–Пекер (ітеративно): перша й остання точки лишаються."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = pts[i], pts[j]
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy) or 1e-12
        dmax, k = 0.0, -1
        for m in range(i + 1, j):
            x, y = pts[m]
            d = abs(dy * (x - x1) - dx * (y - y1)) / n
            if d > dmax:
                dmax, k = d, m
        if dmax > tol:
            keep[k] = True
            stack += [(i, k), (k, j)]
    return [p for p, f in zip(pts, keep) if f]


def dp_ring(ring: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Замкнене кільце (перша = остання точка): ділимо по найвіддаленішій від початку точці й спрощуємо половини."""
    if len(ring) < 5:
        return ring
    x0, y0 = ring[0]
    k = max(range(1, len(ring) - 1), key=lambda i: (ring[i][0] - x0) ** 2 + (ring[i][1] - y0) ** 2)
    return dp(ring[:k + 1], tol)[:-1] + dp(ring[k:], tol)


def path_of(rings: list[list[tuple[float, float]]], closed: bool) -> str:
    out = []
    for r in rings:
        if len(r) < 2:
            continue
        out.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in r) + ("Z" if closed else ""))
    return "".join(out)


def chains(edges: set[frozenset]) -> list[list[tuple[float, float]]]:
    """Склеює ребра в ламані (жадібно, через вершини зі ступенем ≠ 2 спершу)."""
    adj: dict = defaultdict(set)
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    used: set[frozenset] = set()
    out = []

    def walk(start, nxt):
        line = [start, nxt]
        used.add(frozenset((start, nxt)))
        while True:
            cur = line[-1]
            if len(adj[cur]) != 2:
                break
            cand = [n for n in adj[cur] if frozenset((cur, n)) not in used]
            if not cand:
                break
            line.append(cand[0])
            used.add(frozenset((cur, cand[0])))
        return line

    for v in [v for v in adj if len(adj[v]) != 2] + list(adj):
        for n in list(adj[v]):
            if frozenset((v, n)) not in used:
                out.append(walk(v, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="локальна копія GeoJSON (якщо є — не качати)")
    args = ap.parse_args()
    units = set(json.loads(META.read_text())["units"])
    if args.cache and Path(args.cache).exists():
        gj = json.loads(Path(args.cache).read_text())
    else:
        r = requests.get(SRC, timeout=180)
        r.raise_for_status()
        gj = r.json()
        if args.cache:
            Path(args.cache).write_text(r.text)

    lat_mid = 48.3
    pts = [p for f in gj["features"] for poly in ([f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon"
           else f["geometry"]["coordinates"]) for ring in poly for p in ring]
    lon0, lon1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lat0, lat1 = min(p[1] for p in pts), max(p[1] for p in pts)
    k = WIDTH / ((lon1 - lon0) * math.cos(math.radians(lat_mid)))
    height = (lat1 - lat0) * k

    def prj(p):
        return round((p[0] - lon0) * math.cos(math.radians(lat_mid)) * k, 2), round((lat1 - p[1]) * k, 2)

    rings_by_unit: dict[str, list] = defaultdict(list)
    other: dict[tuple, list] = defaultdict(list)       # (область, причина) -> кільця
    edge_owner: dict[frozenset, list] = defaultdict(list)   # ребро -> [(ключ одиниці, область)]
    for f in gj["features"]:
        p = f["properties"]
        ob, ra = p["ADM1_UA"], p["ADM2_UA"]
        u = unit_of(ob, ra, units)
        key = u or f"{ob}|{ra}"
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            for ring in poly:
                pr = [prj(q) for q in ring]
                for a, b in zip(pr, pr[1:]):
                    if a != b:
                        edge_owner[frozenset((a, b))].append((key, ob))
                (rings_by_unit[u] if u else other[ob]).append(pr)

    # шари ліній: межа одиниць / межа областей / зовнішній контур
    unit_edges, oblast_edges, outline = set(), set(), set()
    for e, owners in edge_owner.items():
        keys = {k_ for k_, _ in owners}
        obl = {o for _, o in owners}
        if len(owners) == 1:
            outline.add(e)
        elif len(keys) > 1:
            (oblast_edges if len(obl) > 1 else unit_edges).add(e)

    def lines(edges):
        return path_of([dp(c, TOL) for c in chains(edges)], closed=False)

    def area_c(r):
        a = cx = cy = 0.0
        for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1]):
            c = x1 * y2 - x2 * y1
            a += c
            cx += (x1 + x2) * c
            cy += (y1 + y2) * c
        return a / 2, (cx / (6 * a), cy / (6 * a)) if a else (0, (r[0][0], r[0][1]))

    out_units = []
    for u in sorted(rings_by_unit):
        rs = rings_by_unit[u]
        w = [(abs(area_c(r)[0]), area_c(r)[1]) for r in rs]
        tot = sum(a for a, _ in w) or 1
        cx = sum(a * c[0] for a, c in w) / tot
        cy = sum(a * c[1] for a, c in w) / tot
        out_units.append({"id": u, "d": path_of([dp_ring(r, TOL) for r in rs], closed=True),
                          "cx": round(cx, 1), "cy": round(cy, 1)})
    missing = sorted(units - set(rings_by_unit))
    if missing:
        raise SystemExit(f"немає меж для одиниць: {missing}")

    out_other = [{"name": ob if ob != "Київська" else "Київська (зона відчуження)",
                  "reason": REASONS.get(ob, "немає окремої одиниці в моделі"),
                  "d": path_of([dp_ring(r, TOL) for r in rs], closed=True)} for ob, rs in other.items()]
    # Київська: зона відчуження — окремий поліґон поза одиницями; решта Київської — це одиниці
    doc = {"w": WIDTH, "h": round(height, 1), "units": out_units, "other": out_other,
           "unitBorders": lines(unit_edges), "oblastBorders": lines(oblast_edges), "outline": lines(outline),
           "source": "UNOSAT, ArcGIS MapServer FL20230606UKR/AdminLevel_UKR, шар «Raion» (2023)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.0f} КБ, одиниць {len(out_units)}, інших {len(out_other)}")


if __name__ == "__main__":
    main()
