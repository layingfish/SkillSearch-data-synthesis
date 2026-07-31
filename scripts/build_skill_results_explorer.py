from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def unique_preserving_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return None


def first_plain_paragraph(text: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith("#"):
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(("-", "*", "|")):
            continue
        current.append(stripped)
        if len(" ".join(current)) > 260:
            break
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs[0] if paragraphs else ""


def compact_body(text: str, limit: int = 2400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def unquote_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_skill_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in {"name", "description"}:
            out[key] = unquote_frontmatter_value(value)
    return out


def skill_id(row: dict[str, Any]) -> str:
    for key in ("skill_id", "id", "sid"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"skill row has no skill id: {sorted(row)}")


def normalize_skill(row: dict[str, Any]) -> dict[str, Any]:
    sid = skill_id(row)
    frontmatter = parse_skill_frontmatter(str(row.get("body") or ""))
    name = str(frontmatter.get("name") or row.get("name") or row.get("skill_name") or sid)
    description = str(frontmatter.get("description") or row.get("description") or "")
    return {
        "skill_id": sid,
        "name": name,
        "description": description,
    }


def fallback_skill(sid: str) -> dict[str, Any]:
    return {
        "skill_id": sid,
        "name": sid,
        "description": "",
    }


def normalize_semantic_id(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sid = skill_id(row)
    values = [int(value) for value in (row.get("semantic_id") or [])]
    tokens = [str(value) for value in (row.get("tokens") or [])]
    if not values and not tokens:
        raise ValueError(f"semantic-id row has neither semantic_id nor tokens: {sid}")
    return sid, {"values": values, "tokens": tokens}


def row_key(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("index", fallback_index))


def prediction_key(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("index", fallback_index))


def ranked_ids(prediction: dict[str, Any] | None) -> list[str]:
    if not prediction:
        return []
    values = prediction.get("ranked_skill_ids") or prediction.get("predicted_skill_ids") or []
    return unique_preserving_order(list(values))


def rank_lookup(ids: list[str]) -> dict[str, int]:
    return {sid: idx + 1 for idx, sid in enumerate(ids)}


def slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def build_explorer_payload(
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    cutoffs: list[int],
    title: str = "SkillSearch results explorer",
    semantic_ids: list[dict[str, Any]] | None = None,
    strict_skill_metadata: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows is empty")
    if not runs:
        raise ValueError("at least one run is required")
    if not cutoffs:
        raise ValueError("at least one cutoff is required")

    catalog = {skill["skill_id"]: skill for skill in (normalize_skill(row) for row in skills)}
    semantic_id_catalog = dict(normalize_semantic_id(row) for row in (semantic_ids or []))
    run_payloads: list[dict[str, Any]] = []
    prediction_maps: list[dict[str, dict[str, Any]]] = []
    used_run_ids: set[str] = set()
    for idx, run in enumerate(runs):
        run_name = str(run.get("name") or f"run_{idx}")
        base_run_id = slugify(run_name, f"run-{idx}")
        run_id = base_run_id
        suffix = 2
        while run_id in used_run_ids:
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1
        used_run_ids.add(run_id)
        predictions = list(run.get("predictions") or [])
        prediction_map = {
            prediction_key(prediction, fallback_index): prediction
            for fallback_index, prediction in enumerate(predictions)
        }
        prediction_maps.append(prediction_map)
        run_payloads.append(
            {
                "id": run_id,
                "name": run_name,
                "prediction_count": len(predictions),
            }
        )

    query_payloads: list[dict[str, Any]] = []
    displayed_skill_ids: set[str] = set()
    total_gold = 0
    for fallback_index, row in enumerate(rows):
        key = row_key(row, fallback_index)
        gold_ids = unique_preserving_order(list(row.get("selected_skill_ids") or []))
        displayed_skill_ids.update(gold_ids)
        gold_set = set(gold_ids)
        total_gold += len(gold_ids)
        per_run_ids = [ranked_ids(prediction_map.get(key)) for prediction_map in prediction_maps]
        per_run_ranked = [rank_lookup(ids) for ids in per_run_ids]

        skill_items: list[dict[str, Any]] = []
        for sid in gold_ids:
            catalog_meta = catalog.get(sid)
            if catalog_meta is None and strict_skill_metadata:
                raise ValueError(
                    f"missing metadata for displayed skill {sid!r}; "
                    "pass the complete skill catalog or use --allow-missing-skill-metadata"
                )
            meta = dict(catalog_meta or fallback_skill(sid))
            run_hits: dict[str, dict[str, Any]] = {}
            for run_meta, ranks in zip(run_payloads, per_run_ranked):
                rank = ranks.get(sid)
                run_hits[run_meta["id"]] = {
                    "rank": rank,
                    "hit_at": {str(k): rank if rank is not None and rank <= k else None for k in cutoffs},
                }
            meta["runs"] = run_hits
            skill_items.append(meta)

        coverage_by_cutoff: dict[str, dict[str, int]] = {}
        for k in cutoffs:
            covered_any = 0
            for item in skill_items:
                if any(hit["rank"] is not None and hit["rank"] <= k for hit in item["runs"].values()):
                    covered_any += 1
            coverage_by_cutoff[str(k)] = {
                "covered_any": covered_any,
                "missed_any": len(skill_items) - covered_any,
            }

        metrics_by_run: dict[str, dict[str, dict[str, float | int]]] = {}
        for run_meta in run_payloads:
            run_metrics: dict[str, dict[str, float | int]] = {}
            for k in cutoffs:
                hit_count = sum(
                    1
                    for item in skill_items
                    if item["runs"][run_meta["id"]]["rank"] is not None
                    and item["runs"][run_meta["id"]]["rank"] <= k
                )
                run_metrics[str(k)] = {
                    "hit_count": hit_count,
                    "gold_count": len(skill_items),
                    "hit": int(hit_count > 0),
                    "recall": float(hit_count / len(skill_items)) if skill_items else 0.0,
                    "full_coverage": int(bool(skill_items) and hit_count == len(skill_items)),
                }
            metrics_by_run[run_meta["id"]] = run_metrics

        ranked_results: dict[str, list[dict[str, Any]]] = {}
        max_cutoff = max(cutoffs)
        for run_meta, ids in zip(run_payloads, per_run_ids):
            ranked_items: list[dict[str, Any]] = []
            for rank, sid in enumerate(ids[:max_cutoff], start=1):
                displayed_skill_ids.add(sid)
                catalog_meta = catalog.get(sid)
                if catalog_meta is None and strict_skill_metadata:
                    raise ValueError(
                        f"missing metadata for displayed skill {sid!r}; "
                        "pass the complete skill catalog or use --allow-missing-skill-metadata"
                    )
                meta = dict(catalog_meta or fallback_skill(sid))
                ranked_items.append(
                    {
                        "rank": rank,
                        "skill_id": sid,
                        "name": meta["name"],
                        "description": meta["description"],
                        "is_gt": sid in gold_set,
                    }
                )
            ranked_results[run_meta["id"]] = ranked_items

        query_payloads.append(
            {
                "index": key,
                "display_index": fallback_index + 1,
                "query": str(row.get("query") or row.get("query_text") or ""),
                "domain": str(row.get("domain") or row.get("category") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "source_index": str(row.get("source_index") or ""),
                "target_skill_count": int(row.get("target_skill_count") or len(gold_ids)),
                "relevant_count": len(gold_ids),
                "skills": skill_items,
                "coverage_by_cutoff": coverage_by_cutoff,
                "metrics_by_run": metrics_by_run,
                "ranked_results": ranked_results,
            }
        )

    for run_meta in run_payloads:
        metrics_by_cutoff: dict[str, dict[str, float | int]] = {}
        for k in cutoffs:
            query_metrics = [query["metrics_by_run"][run_meta["id"]][str(k)] for query in query_payloads]
            hit_queries = sum(int(metric["hit"]) for metric in query_metrics)
            full_coverage_queries = sum(int(metric["full_coverage"]) for metric in query_metrics)
            total_hits = sum(int(metric["hit_count"]) for metric in query_metrics)
            recall_sum = sum(float(metric["recall"]) for metric in query_metrics)
            macro_recall = float(recall_sum / len(query_payloads)) if query_payloads else 0.0
            micro_recall = float(total_hits / total_gold) if total_gold else 0.0
            metrics_by_cutoff[str(k)] = {
                "hit_queries": hit_queries,
                "full_coverage_queries": full_coverage_queries,
                "query_count": len(query_payloads),
                "total_hits": total_hits,
                "total_gold": total_gold,
                "recall_sum": recall_sum,
                "hit": float(hit_queries / len(query_payloads)) if query_payloads else 0.0,
                "full_coverage": (
                    float(full_coverage_queries / len(query_payloads)) if query_payloads else 0.0
                ),
                "recall": macro_recall,
                "macro_recall": macro_recall,
                "micro_recall": micro_recall,
            }
        run_meta["metrics_by_cutoff"] = metrics_by_cutoff

    displayed_semantic_ids = {
        sid: semantic_id_catalog[sid]
        for sid in sorted(displayed_skill_ids)
        if sid in semantic_id_catalog
    }

    return {
        "title": title,
        "summary": {
            "query_count": len(query_payloads),
            "total_relevant_skills": total_gold,
            "cutoffs": cutoffs,
            "recall_aggregation": "macro",
        },
        "runs": run_payloads,
        "queries": query_payloads,
        "semantic_ids": displayed_semantic_ids,
    }


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty run name in --run {value!r}")
    return name, Path(path)


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")
    if any(k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must be positive integers")
    return cutoffs


def make_run(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path), "predictions": load_jsonl(path)}


def render_html(payload: dict[str, Any], *, default_cutoff: int) -> str:
    if not payload.get("runs"):
        raise ValueError("at least one run is required")
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #f3f5f7;
      --panel: #ffffff;
      --panel-subtle: #f8fafb;
      --line: #d7dee6;
      --line-strong: #c5cfda;
      --text: #202733;
      --muted: #657182;
      --muted-2: #8591a1;
      --blue: #276fb7;
      --blue-strong: #195c9f;
      --blue-weak: #eaf3fc;
      --green: #147743;
      --green-weak: #eef9f2;
      --green-line: #b7dfc5;
      --red: #b33c50;
      --red-weak: #fff1f3;
      --red-line: #edc1c9;
      --shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-size: 14px;
    }
    button, input { font: inherit; }
    button:focus-visible, input:focus-visible {
      outline: 3px solid rgba(39, 111, 183, 0.22);
      outline-offset: 2px;
    }
    .topbar {
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 12px 18px;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .title-wrap { min-width: 0; }
    .title {
      font-size: 18px;
      font-weight: 780;
      overflow-wrap: anywhere;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
      margin-top: 3px;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .control-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
    }
    .segmented {
      display: inline-flex;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .segmented button {
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      min-height: 38px;
      padding: 8px 14px;
      color: var(--muted);
      cursor: pointer;
      white-space: nowrap;
    }
    .segmented button:last-child { border-right: 0; }
    .segmented button.active {
      color: var(--blue-strong);
      font-weight: 760;
      background: var(--blue-weak);
    }
    .segmented button:active { transform: translateY(1px); }
    .search {
      width: 250px;
      height: 38px;
      padding: 0 14px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      box-shadow: var(--shadow);
      outline: none;
    }
    .search::placeholder { color: var(--muted-2); }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(720px, 1fr);
      min-height: calc(100vh - 76px);
    }
    .sidebar {
      max-height: calc(100vh - 76px);
      padding: 12px 8px 20px;
      border-right: 1px solid var(--line);
      background: #f8fafc;
      overflow: auto;
    }
    .query-card {
      min-height: 104px;
      padding: 13px 13px 14px;
      margin-bottom: 5px;
      border: 1px solid transparent;
      border-radius: 8px;
      cursor: pointer;
    }
    .query-card:hover { background: #f1f5f8; }
    .query-card.active {
      background: var(--panel);
      border-color: var(--line-strong);
      box-shadow: var(--shadow);
    }
    .q-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
      font-size: 15px;
      font-weight: 800;
    }
    .q-meta-right {
      max-width: 150px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .q-preview {
      display: -webkit-box;
      margin-bottom: 7px;
      color: #455163;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-line;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .q-stats {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      font-weight: 720;
    }
    .dots {
      display: flex;
      gap: 5px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--red);
    }
    .dot.hit { background: var(--green); }
    .main {
      max-height: calc(100vh - 76px);
      padding: 22px 28px 56px;
      overflow: auto;
    }
    .content { width: min(100%, 1320px); }
    .run-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: center;
      padding: 14px 16px;
      margin-bottom: 20px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .run-label {
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .run-name {
      color: var(--text);
      font-size: 18px;
      line-height: 1.25;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .run-meta {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      text-align: right;
      white-space: nowrap;
    }
    .run-tabs {
      grid-column: 1 / -1;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      overflow: visible;
    }
    .run-tab-list {
      width: 100%;
      min-width: 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel-subtle);
      overflow: hidden;
    }
    .run-tab {
      min-height: 40px;
      flex: 1;
      padding: 8px 14px;
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      line-height: 1.25;
      overflow-wrap: anywhere;
      cursor: pointer;
    }
    .run-tab:last-child { border-right: 0; }
    .run-tab:hover { background: #f1f5f8; }
    .run-tab.active {
      background: var(--blue-weak);
      color: var(--blue-strong);
      font-weight: 820;
      box-shadow: inset 0 -3px 0 var(--blue);
    }
    .run-tab:active { transform: translateY(1px); }
    .query-block {
      position: relative;
      padding: 16px 18px 18px;
      margin-bottom: 18px;
      border: 1px solid var(--line-strong);
      border-left: 4px solid var(--blue);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .query-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 11px;
    }
    .query-number {
      color: var(--blue-strong);
      font-size: 14px;
      font-weight: 820;
    }
    .query-tags {
      display: flex;
      justify-content: flex-end;
      gap: 6px;
      flex-wrap: wrap;
    }
    .query-tag {
      padding: 4px 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-subtle);
      color: var(--muted);
      font-size: 11px;
      font-weight: 680;
    }
    .query-text {
      max-width: 1120px;
      max-height: 360px;
      padding-right: 8px;
      color: #273142;
      font-size: 16px;
      line-height: 1.68;
      font-weight: 520;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      overflow-y: auto;
      scrollbar-gutter: stable;
    }
    .metrics-panel {
      margin-bottom: 18px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .metrics-heading {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--line);
    }
    .section-title {
      color: var(--text);
      font-size: 15px;
      font-weight: 820;
    }
    .formula-definitions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      margin-top: 10px;
    }
    .formula-item {
      min-width: 0;
      padding: 9px 11px 10px;
      border-left: 3px solid #c9dff3;
      background: #f8fafc;
    }
    .formula-label {
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .formula-expression {
      max-width: 100%;
      margin-top: 4px;
      overflow-x: auto;
      overflow-y: hidden;
    }
    .formula-math, .metric-math {
      color: #273142;
      font-family: "STIX Two Math", "Cambria Math", "Times New Roman", serif;
    }
    .formula-math {
      font-size: 16px;
    }
    .formula-math + .formula-math { margin-top: 4px; }
    .formula-note {
      margin-top: 9px;
      color: #445166;
      font-size: 11px;
      font-weight: 650;
    }
    .metric-scope + .metric-scope { border-top: 1px solid var(--line); }
    .metric-scope-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 11px 16px 9px;
      background: var(--panel-subtle);
    }
    .metric-scope-title {
      color: #374151;
      font-size: 13px;
      font-weight: 800;
    }
    .metric-scope-note {
      color: var(--muted);
      font-size: 11px;
    }
    .metric-scroll { overflow-x: auto; }
    .metric-table {
      display: grid;
      grid-template-columns: 150px repeat(3, minmax(190px, 1fr));
      min-width: 760px;
    }
    .metric-th, .metric-row-label, .metric-cell {
      min-width: 0;
      border-top: 1px solid #edf1f4;
      border-right: 1px solid #edf1f4;
    }
    .metric-th:last-child, .metric-cell:last-child { border-right: 0; }
    .metric-th {
      padding: 8px 12px;
      background: #fbfcfd;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      font-weight: 800;
    }
    .metric-row-label {
      display: flex;
      align-items: center;
      padding: 12px;
      color: #3f4a5a;
      font-weight: 800;
    }
    .metric-cell {
      padding: 10px 12px 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .metric-equation {
      min-height: 24px;
      display: flex;
      align-items: center;
      color: #374151;
      font-size: 12px;
      font-weight: 760;
      white-space: nowrap;
    }
    .metric-math {
      font-size: 14px;
      white-space: nowrap;
    }
    .metric-value {
      margin-top: 4px;
      color: var(--blue-strong);
      font-size: 11px;
    }
    .results-panel {
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .results-toolbar, .results-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
    }
    .results-toolbar { background: var(--panel-subtle); }
    .result-mode-buttons button { min-width: 136px; }
    .results-summary {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-align: right;
    }
    .legend {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 11px;
      font-weight: 680;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border: 1px solid;
      border-radius: 3px;
    }
    .legend-swatch.green {
      border-color: var(--green-line);
      background: var(--green-weak);
    }
    .legend-swatch.red {
      border-color: var(--red-line);
      background: var(--red-weak);
    }
    .result-list {
      display: flex;
      flex-direction: column;
    }
    .result-row {
      border-left: 4px solid transparent;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .result-row:last-child { border-bottom: 0; }
    .result-row.gt {
      border-left-color: var(--green);
      background: var(--green-weak);
    }
    .result-row.non-gt, .result-row.miss {
      border-left-color: var(--red);
      background: var(--red-weak);
    }
    .result-summary-button {
      width: 100%;
      min-height: 64px;
      display: grid;
      grid-template-columns: 66px minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 10px 14px 10px 10px;
      border: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .result-summary-button:hover { background: rgba(255, 255, 255, 0.34); }
    .result-summary-button:active { transform: translateY(1px); }
    .rank-token {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      font-weight: 820;
      text-align: center;
    }
    .skill-title {
      color: #263141;
      font-size: 14px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .result-status {
      min-width: 112px;
      padding: 6px 8px;
      border: 1px solid;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.25;
      font-weight: 800;
      text-align: center;
    }
    .result-row.gt .result-status {
      border-color: var(--green-line);
      background: rgba(255, 255, 255, 0.56);
      color: var(--green);
    }
    .result-row.non-gt .result-status, .result-row.miss .result-status {
      border-color: var(--red-line);
      background: rgba(255, 255, 255, 0.56);
      color: var(--red);
    }
    .result-detail {
      display: none;
      padding: 13px 16px 16px 88px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.58);
      color: var(--muted);
      line-height: 1.55;
    }
    .result-row.open .result-detail { display: block; }
    .detail-grid {
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr);
      gap: 9px 16px;
      max-width: 1000px;
    }
    .detail-label {
      color: var(--muted-2);
      font-size: 11px;
      font-weight: 780;
      text-transform: uppercase;
    }
    .detail-value {
      min-width: 0;
      color: #374151;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .semantic-id-value {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: normal;
    }
    .semantic-id-vector {
      color: #273142;
      font-size: 13px;
      font-weight: 760;
    }
    .semantic-id-components {
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .empty {
      padding: 40px 20px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 920px) {
      .topbar { align-items: flex-start; flex-direction: column; position: static; }
      .controls { justify-content: flex-start; width: 100%; }
      .layout { grid-template-columns: 1fr; }
      .sidebar {
        max-height: 280px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .main { max-height: none; padding: 18px 14px 40px; }
      .search { width: min(100%, 320px); }
      .run-header { grid-template-columns: 1fr; }
      .run-meta { text-align: left; }
    }
    @media (max-width: 640px) {
      .controls { align-items: flex-start; flex-direction: column; }
      .segmented { width: 100%; }
      .segmented button { flex: 1; padding-inline: 8px; }
      .search { width: 100%; }
      .query-block { padding: 14px; }
      .query-meta { align-items: flex-start; flex-direction: column; }
      .query-tags { justify-content: flex-start; }
      .query-text { font-size: 15px; }
      .formula-definitions { grid-template-columns: 1fr; }
      .metric-scope-head { align-items: flex-start; flex-direction: column; }
      .results-toolbar, .results-head { align-items: flex-start; flex-direction: column; }
      .result-mode-buttons button { min-width: 0; }
      .results-summary { text-align: left; }
      .run-tabs { overflow: visible; }
      .run-tab-list {
        width: 100%;
        min-width: 0;
        grid-template-columns: 1fr;
      }
      .run-tab {
        width: 100%;
        border-right: 0;
        border-bottom: 1px solid var(--line);
        text-align: left;
      }
      .run-tab:last-child { border-bottom: 0; }
      .run-tab.active { box-shadow: inset 3px 0 0 var(--blue); }
      .result-summary-button { grid-template-columns: 48px minmax(0, 1fr); }
      .result-status { grid-column: 2; justify-self: start; }
      .result-detail { padding-left: 16px; }
      .detail-grid { grid-template-columns: 1fr; gap: 4px; }
      .detail-label:not(:first-child) { margin-top: 8px; }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title-wrap">
      <div class="title" id="page-title"></div>
      <div class="subtitle" id="page-subtitle"></div>
    </div>
    <div class="controls">
      <span class="control-label">Result cutoff</span>
      <div class="segmented" id="cutoff-buttons"></div>
      <input class="search" id="query-filter" aria-label="Search queries and skills" placeholder="Search query or skill name">
    </div>
  </div>
  <div class="layout">
    <aside class="sidebar" id="sidebar"></aside>
    <main class="main" id="main-panel">
      <div class="content">
        <section class="run-header" id="run-header"></section>
        <section id="query-panel"></section>
      </div>
    </main>
  </div>
  <script>
    const DATA = __PAYLOAD_JSON__;
    const state = {
      cutoff: __DEFAULT_CUTOFF__,
      queryIndex: 0,
      activeRunId: DATA.runs[0].id,
      resultMode: "ground-truth",
      filter: ""
    };

    const $ = (id) => document.getElementById(id);
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[ch]));

    function formatScore(value) {
      const number = Number(value || 0);
      return `${number.toFixed(3)} (${(number * 100).toFixed(1)}%)`;
    }

    function semanticIdData(skillId) {
      return (DATA.semantic_ids || {})[skillId] || null;
    }

    function learnedSemanticComponents(semanticId) {
      if (!semanticId) return [];
      const values = semanticId.values || [];
      const tokens = semanticId.tokens || [];
      if (tokens.length === values.length && tokens.length > 0) {
        return values.map((value, index) => ({ value, token: tokens[index] }))
          .filter((component) => !component.token.startsWith("<DUP_"));
      }
      return values.map((value, index) => ({ value, token: `RQ${index}` }));
    }

    function semanticIdSearchText(skillId) {
      const semanticId = semanticIdData(skillId);
      if (!semanticId) return "";
      const components = learnedSemanticComponents(semanticId);
      return components.flatMap((component) => [component.value, component.token]).join(" ").toLowerCase();
    }

    function activeRun() {
      return DATA.runs.find((run) => run.id === state.activeRunId) || DATA.runs[0];
    }

    function queryMetric(query, cutoff) {
      return query.metrics_by_run[activeRun().id][String(cutoff)];
    }

    function renderCutoffs() {
      $("cutoff-buttons").innerHTML = DATA.summary.cutoffs.map((cutoff) => `
        <button type="button" class="${cutoff === state.cutoff ? "active" : ""}" data-cutoff="${cutoff}">Top ${cutoff}</button>
      `).join("");
      $("cutoff-buttons").querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          state.cutoff = Number(button.dataset.cutoff);
          render();
        });
      });
    }

    function queryMatchesFilter(query) {
      const needle = state.filter.trim().toLowerCase();
      if (!needle) return true;
      if (query.query.toLowerCase().includes(needle)) return true;
      if (query.index.toLowerCase().includes(needle)) return true;
      if (query.source_index.toLowerCase().includes(needle)) return true;
      return query.skills.some((skill) =>
        skill.skill_id.toLowerCase().includes(needle) ||
        skill.name.toLowerCase().includes(needle) ||
        skill.description.toLowerCase().includes(needle) ||
        semanticIdSearchText(skill.skill_id).includes(needle)
      ) || Object.values(query.ranked_results || {}).some((items) =>
        items.some((item) =>
          item.skill_id.toLowerCase().includes(needle) ||
          item.name.toLowerCase().includes(needle) ||
          item.description.toLowerCase().includes(needle) ||
          semanticIdSearchText(item.skill_id).includes(needle)
        )
      );
    }

    function renderSidebar() {
      const queries = DATA.queries.filter(queryMatchesFilter);
      if (!queries.length) {
        $("sidebar").innerHTML = '<div class="empty">No matching queries</div>';
        return;
      }
      if (!queries.some((query) => query.display_index - 1 === state.queryIndex)) {
        state.queryIndex = queries[0].display_index - 1;
      }
      $("sidebar").innerHTML = queries.map((query) => {
        const metric = queryMetric(query, state.cutoff);
        const dots = query.skills.map((skill) => {
          const rank = skill.runs[activeRun().id].rank;
          return `<span class="dot ${rank !== null && rank <= state.cutoff ? "hit" : ""}"></span>`;
        }).join("");
        const active = query.display_index - 1 === state.queryIndex ? "active" : "";
        const label = [query.domain, query.difficulty].filter(Boolean).join(" / ");
        return `
          <div class="query-card ${active}" data-query-index="${query.display_index - 1}" role="button" tabindex="0">
            <div class="q-head"><span>Q${query.display_index}</span><span class="q-meta-right">${escapeHtml(label)}</span></div>
            <div class="q-preview">${escapeHtml(query.query)}</div>
            <div class="q-stats">H@${state.cutoff}=${metric.hit} | R@${state.cutoff}=${Number(metric.recall).toFixed(3)}</div>
            <div class="dots" aria-label="GT coverage at selected cutoff">${dots}</div>
          </div>`;
      }).join("");
      $("sidebar").querySelectorAll(".query-card").forEach((card) => {
        const selectQuery = () => {
          state.queryIndex = Number(card.dataset.queryIndex);
          render();
          $("main-panel").scrollTop = 0;
        };
        card.addEventListener("click", selectQuery);
        card.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectQuery();
          }
        });
      });
    }

    function renderRunHeader() {
      const run = activeRun();
      const tabs = DATA.runs.map((item) => `
        <button type="button" class="run-tab ${item.id === run.id ? "active" : ""}" data-run-id="${item.id}" aria-pressed="${item.id === run.id}">
          ${escapeHtml(item.name)}
        </button>
      `).join("");
      $("run-header").innerHTML = `
        <div>
          <div class="run-label">Active run</div>
          <div class="run-name">${escapeHtml(run.name)}</div>
        </div>
        <div class="run-meta">${DATA.summary.query_count} queries | ${DATA.summary.total_relevant_skills} GT skills | ${DATA.runs.length} runs</div>
        <div class="run-tabs" aria-label="Switch retrieval run">
          <div class="run-tab-list">${tabs}</div>
        </div>
      `;
      $("run-header").querySelectorAll(".run-tab").forEach((button) => {
        button.addEventListener("click", () => {
          state.activeRunId = button.dataset.runId;
          render();
          $("main-panel").scrollTop = 0;
        });
      });
    }

    function metricTable(scope, query) {
      const isQuery = scope === "query";
      const run = activeRun();
      const metrics = DATA.summary.cutoffs.map((cutoff) =>
        isQuery ? queryMetric(query, cutoff) : run.metrics_by_cutoff[String(cutoff)]
      );
      const headers = DATA.summary.cutoffs.map((cutoff) => `<div class="metric-th">@${cutoff}</div>`).join("");
      const hitCells = metrics.map((metric) => {
        const equation = isQuery
          ? `<math class="metric-math" aria-label="one if ${metric.hit_count} is greater than zero equals ${metric.hit}">
               <mn>1</mn><mo>[</mo><mn>${metric.hit_count}</mn><mo>&gt;</mo><mn>0</mn><mo>]</mo>
               <mo>=</mo><mn>${metric.hit}</mn>
             </math>`
          : `<math class="metric-math" aria-label="${metric.hit_queries} divided by ${metric.query_count} equals ${Number(metric.hit).toFixed(3)}">
               <mfrac><mn>${metric.hit_queries}</mn><mn>${metric.query_count}</mn></mfrac>
               <mo>=</mo><mn>${Number(metric.hit).toFixed(3)}</mn>
             </math>`;
        return `
          <div class="metric-cell">
            <div class="metric-equation">${equation}</div>
            <div class="metric-value">${formatScore(metric.hit)}</div>
          </div>`;
      }).join("");
      const recallCells = metrics.map((metric) => {
        const equation = isQuery
          ? `<math class="metric-math" aria-label="${metric.hit_count} divided by ${metric.gold_count} equals ${Number(metric.recall).toFixed(3)}">
               <mfrac><mn>${metric.hit_count}</mn><mn>${metric.gold_count}</mn></mfrac>
               <mo>=</mo><mn>${Number(metric.recall).toFixed(3)}</mn>
             </math>`
          : `<math class="metric-math" aria-label="sum of per-query recalls ${Number(metric.recall_sum).toFixed(3)} divided by ${metric.query_count} equals ${Number(metric.recall).toFixed(3)}">
               <mfrac><mn>${Number(metric.recall_sum).toFixed(3)}</mn><mn>${metric.query_count}</mn></mfrac>
               <mo>=</mo><mn>${Number(metric.recall).toFixed(3)}</mn>
             </math>`;
        return `
          <div class="metric-cell">
            <div class="metric-equation">${equation}</div>
            <div class="metric-value">${formatScore(metric.recall)}</div>
          </div>`;
      }).join("");
      const fullCoverageCells = metrics.map((metric) => {
        const equation = isQuery
          ? `<math class="metric-math" aria-label="one if ${metric.hit_count} equals ${metric.gold_count} equals ${metric.full_coverage}">
               <mn>1</mn><mo>[</mo><mn>${metric.hit_count}</mn><mo>=</mo><mn>${metric.gold_count}</mn><mo>]</mo>
               <mo>=</mo><mn>${metric.full_coverage}</mn>
             </math>`
          : `<math class="metric-math" aria-label="${metric.full_coverage_queries} divided by ${metric.query_count} equals ${Number(metric.full_coverage).toFixed(3)}">
               <mfrac><mn>${metric.full_coverage_queries}</mn><mn>${metric.query_count}</mn></mfrac>
               <mo>=</mo><mn>${Number(metric.full_coverage).toFixed(3)}</mn>
             </math>`;
        return `
          <div class="metric-cell">
            <div class="metric-equation">${equation}</div>
            <div class="metric-value">${formatScore(metric.full_coverage)}</div>
          </div>`;
      }).join("");
      return `
        <div class="metric-scroll">
          <div class="metric-table">
            <div class="metric-th">Metric</div>${headers}
            <div class="metric-row-label">Hit@k</div>${hitCells}
            <div class="metric-row-label">Recall@k</div>${recallCells}
            <div class="metric-row-label">FC@k</div>${fullCoverageCells}
          </div>
        </div>`;
    }

    function renderMetrics(query) {
      return `
        <section class="metrics-panel">
          <div class="metrics-heading">
            <div class="section-title">Hit, Recall, and Full Coverage formulas</div>
            <div class="formula-definitions">
              <div class="formula-item">
                <div class="formula-label">Retrieved GT count</div>
                <div class="formula-expression">
                  <math class="formula-math" display="block" aria-label="h sub q of k equals the size of Top k of q intersect ground truth of q">
                    <msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo>
                    <mo>|</mo><msub><mtext>Top</mtext><mi>k</mi></msub><mo>(</mo><mi>q</mi><mo>)</mo>
                    <mo>&cap;</mo><mtext>GT</mtext><mo>(</mo><mi>q</mi><mo>)</mo><mo>|</mo>
                  </math>
                  <math class="formula-math" display="block" aria-label="G sub q equals the size of ground truth of q">
                    <msub><mi>G</mi><mi>q</mi></msub><mo>=</mo><mo>|</mo><mtext>GT</mtext><mo>(</mo><mi>q</mi><mo>)</mo><mo>|</mo>
                  </math>
                </div>
              </div>
              <div class="formula-item">
                <div class="formula-label">Per-query metrics</div>
                <div class="formula-expression">
                  <math class="formula-math" display="block" aria-label="Hit at k for q is one when h sub q of k is greater than zero">
                    <mtext>Hit</mtext><mo>@</mo><mi>k</mi><mo>(</mo><mi>q</mi><mo>)</mo><mo>=</mo>
                    <mn>1</mn><mo>[</mo><msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>&gt;</mo><mn>0</mn><mo>]</mo>
                  </math>
                  <math class="formula-math" display="block" aria-label="R sub q of k equals h sub q of k divided by G sub q">
                    <msub><mi>R</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo>
                    <mfrac><mrow><msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo></mrow><msub><mi>G</mi><mi>q</mi></msub></mfrac>
                  </math>
                  <math class="formula-math" display="block" aria-label="Full coverage at k for q is one when h sub q of k equals G sub q">
                    <mtext>FC</mtext><mo>@</mo><mi>k</mi><mo>(</mo><mi>q</mi><mo>)</mo><mo>=</mo>
                    <mn>1</mn><mo>[</mo><msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo><msub><mi>G</mi><mi>q</mi></msub><mo>]</mo>
                  </math>
                </div>
              </div>
              <div class="formula-item">
                <div class="formula-label">Run Hit</div>
                <div class="formula-expression">
                  <math class="formula-math" display="block" aria-label="Hit at k is one over Q times the sum over queries of the per-query hit indicator">
                    <mtext>Hit</mtext><mo>@</mo><mi>k</mi><mo>=</mo>
                    <mfrac><mn>1</mn><mi>Q</mi></mfrac>
                    <munderover><mo>&sum;</mo><mrow><mi>q</mi><mo>=</mo><mn>1</mn></mrow><mi>Q</mi></munderover>
                    <mn>1</mn><mo>[</mo><msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>&gt;</mo><mn>0</mn><mo>]</mo>
                  </math>
                </div>
              </div>
              <div class="formula-item">
                <div class="formula-label">Run Recall (macro averaged)</div>
                <div class="formula-expression">
                  <math class="formula-math" display="block" aria-label="Recall at k is one over Q times the sum of per-query recall from q equals one to Q">
                    <mtext>Recall</mtext><mo>@</mo><mi>k</mi><mo>=</mo>
                    <mfrac><mn>1</mn><mi>Q</mi></mfrac>
                    <munderover><mo>&sum;</mo><mrow><mi>q</mi><mo>=</mo><mn>1</mn></mrow><mi>Q</mi></munderover>
                    <msub><mi>R</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo>
                  </math>
                </div>
              </div>
              <div class="formula-item">
                <div class="formula-label">Run Full Coverage</div>
                <div class="formula-expression">
                  <math class="formula-math" display="block" aria-label="Full coverage at k is one over Q times the sum of per-query full coverage indicators">
                    <mtext>FC</mtext><mo>@</mo><mi>k</mi><mo>=</mo>
                    <mfrac><mn>1</mn><mi>Q</mi></mfrac>
                    <munderover><mo>&sum;</mo><mrow><mi>q</mi><mo>=</mo><mn>1</mn></mrow><mi>Q</mi></munderover>
                    <mn>1</mn><mo>[</mo><msub><mi>h</mi><mi>q</mi></msub><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo><msub><mi>G</mi><mi>q</mi></msub><mo>]</mo>
                  </math>
                </div>
              </div>
            </div>
            <div class="formula-note">Macro Recall gives every query equal weight, regardless of how many GT skills it has.</div>
          </div>
          <div class="metric-scope">
            <div class="metric-scope-head">
              <div class="metric-scope-title">This query</div>
              <div class="metric-scope-note">Per-query values before aggregation</div>
            </div>
            ${metricTable("query", query)}
          </div>
          <div class="metric-scope">
            <div class="metric-scope-head">
              <div class="metric-scope-title">Entire run</div>
              <div class="metric-scope-note">Macro averaged across ${DATA.summary.query_count} equally weighted queries</div>
            </div>
            ${metricTable("run", query)}
          </div>
        </section>`;
    }

    function resultDetail(name, description, skillId) {
      const semanticId = semanticIdData(skillId);
      const rqValues = learnedSemanticComponents(semanticId).map((component) => component.value);
      const vector = rqValues.length ? `[${rqValues.join(", ")}]` : "N/A";
      const components = rqValues.length
        ? rqValues.map((value, index) => `RQ${index}=${value}`).join(" &middot; ")
        : "";
      return `
        <div class="result-detail">
          <div class="detail-grid">
            <div class="detail-label">Name</div>
            <div class="detail-value">${escapeHtml(name)}</div>
            <div class="detail-label">Description</div>
            <div class="detail-value">${escapeHtml(description || "")}</div>
            <div class="detail-label">RQ semantic codes</div>
            <div class="detail-value semantic-id-value">
              <div class="semantic-id-vector">${escapeHtml(vector)}</div>
              ${components ? `<div class="semantic-id-components">${components}</div>` : ""}
            </div>
          </div>
        </div>`;
    }

    function renderGroundTruthResults(query) {
      const run = activeRun();
      const maxCutoff = Math.max(...DATA.summary.cutoffs);
      const metric = queryMetric(query, state.cutoff);
      const rows = query.skills.map((skill, index) => {
        const rank = skill.runs[run.id].rank;
        const isHit = rank !== null && rank <= state.cutoff;
        let status = `Not retrieved in Top ${maxCutoff}`;
        if (isHit) status = `Hit at #${rank}`;
        else if (rank !== null) status = `#${rank}, outside Top ${state.cutoff}`;
        return `
          <div class="result-row ${isHit ? "gt" : "miss"}" data-result-row="${index}">
            <button type="button" class="result-summary-button" aria-expanded="false">
              <div class="rank-token">${rank === null ? "N/A" : "#" + rank}</div>
              <div class="skill-title">${escapeHtml(skill.name)}</div>
              <div class="result-status">${escapeHtml(status)}</div>
            </button>
            ${resultDetail(skill.name, skill.description, skill.skill_id)}
          </div>`;
      }).join("");
      return `
        <div class="results-head">
          <div>
            <div class="section-title">Ground-truth retrieval</div>
            <div class="results-summary">${metric.hit_count} of ${metric.gold_count} GT skills retrieved within Top ${state.cutoff}</div>
          </div>
          <div class="legend">
            <span class="legend-item"><span class="legend-swatch green"></span>Retrieved within cutoff</span>
            <span class="legend-item"><span class="legend-swatch red"></span>Missed at cutoff</span>
          </div>
        </div>
        <div class="result-list">${rows}</div>`;
    }

    function renderTopKResults(query) {
      const run = activeRun();
      const items = (query.ranked_results[run.id] || []).filter((item) => item.rank <= state.cutoff);
      const gtCount = items.filter((item) => item.is_gt).length;
      const rows = items.map((item, index) => `
        <div class="result-row ${item.is_gt ? "gt" : "non-gt"}" data-result-row="${index}">
          <button type="button" class="result-summary-button" aria-expanded="false">
            <div class="rank-token">#${item.rank}</div>
            <div class="skill-title">${escapeHtml(item.name)}</div>
            <div class="result-status">${item.is_gt ? "Ground truth" : "Not ground truth"}</div>
          </button>
          ${resultDetail(item.name, item.description, item.skill_id)}
        </div>
      `).join("");
      return `
        <div class="results-head">
          <div>
            <div class="section-title">Top ${state.cutoff} retrieved skills</div>
            <div class="results-summary">${gtCount} GT | ${items.length - gtCount} non-GT</div>
          </div>
          <div class="legend">
            <span class="legend-item"><span class="legend-swatch green"></span>Ground truth</span>
            <span class="legend-item"><span class="legend-swatch red"></span>Not ground truth</span>
          </div>
        </div>
        <div class="result-list">${rows || '<div class="empty">No retrieved skills at this cutoff.</div>'}</div>`;
    }

    function bindResultRows() {
      $("result-view").querySelectorAll(".result-summary-button").forEach((button) => {
        button.addEventListener("click", () => {
          const row = button.closest(".result-row");
          const isOpen = row.classList.toggle("open");
          button.setAttribute("aria-expanded", String(isOpen));
        });
      });
    }

    function renderResultPanel(query) {
      const metric = queryMetric(query, state.cutoff);
      return `
        <section class="results-panel">
          <div class="results-toolbar">
            <div class="segmented result-mode-buttons" id="result-mode-buttons">
              <button type="button" data-mode="ground-truth" class="${state.resultMode === "ground-truth" ? "active" : ""}">Ground truth</button>
              <button type="button" data-mode="top-k" class="${state.resultMode === "top-k" ? "active" : ""}">Top-k retrieved</button>
            </div>
            <div class="results-summary">Selected cutoff: Top ${state.cutoff} | h_q(k) = ${metric.hit_count}</div>
          </div>
          <div id="result-view">
            ${state.resultMode === "ground-truth" ? renderGroundTruthResults(query) : renderTopKResults(query)}
          </div>
        </section>`;
    }

    function renderQueryPanel() {
      const query = DATA.queries[state.queryIndex] || DATA.queries[0];
      const tags = [query.domain, query.difficulty, query.source_index].filter(Boolean).map((value) =>
        `<span class="query-tag">${escapeHtml(value)}</span>`
      ).join("");
      $("query-panel").innerHTML = `
        <section class="query-block">
          <div class="query-meta">
            <div class="query-number">Q${query.display_index}</div>
            <div class="query-tags">${tags}</div>
          </div>
          <div class="query-text">${escapeHtml(query.query)}</div>
        </section>
        ${renderMetrics(query)}
        ${renderResultPanel(query)}
      `;
      $("result-mode-buttons").querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          state.resultMode = button.dataset.mode;
          renderQueryPanel();
        });
      });
      bindResultRows();
    }

    function render() {
      $("page-title").textContent = DATA.title;
      $("page-subtitle").textContent = `${DATA.summary.query_count} queries | ${DATA.summary.total_relevant_skills} GT skills | one active run at a time`;
      renderCutoffs();
      renderSidebar();
      renderRunHeader();
      renderQueryPanel();
    }

    $("query-filter").addEventListener("input", (event) => {
      state.filter = event.target.value;
      render();
    });
    if (!DATA.summary.cutoffs.includes(state.cutoff)) {
      state.cutoff = DATA.summary.cutoffs[DATA.summary.cutoffs.length - 1];
    }
    render();
  </script>
</body>
</html>
"""
    return (
        template.replace("__PAGE_TITLE__", html.escape(str(payload["title"])))
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__DEFAULT_CUTOFF__", str(int(default_cutoff)))
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a static SkillSearch retrieval results explorer.")
    parser.add_argument("--rows", type=Path, required=True, help="Evaluation rows JSONL.")
    parser.add_argument(
        "--skills",
        type=Path,
        required=True,
        help="Complete skill catalog JSONL used to display original names and descriptions.",
    )
    parser.add_argument(
        "--allow-missing-skill-metadata",
        action="store_true",
        help="Allow unknown skill IDs to fall back to the raw ID and an empty description.",
    )
    parser.add_argument(
        "--semantic-ids",
        type=Path,
        help="Optional shared semantic_ids.jsonl mapping used to display each skill's SemID.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form 'Display name=path/to/predictions.jsonl'. Repeat to add switchable runs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--title", default="SkillSearch results explorer")
    parser.add_argument("--cutoffs", default="10,50,100")
    parser.add_argument("--default-cutoff", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cutoffs = parse_cutoffs(args.cutoffs)
    if args.default_cutoff not in cutoffs:
        raise ValueError(f"--default-cutoff {args.default_cutoff} is not in --cutoffs {cutoffs}")

    rows = load_jsonl(args.rows)
    skills = load_jsonl(args.skills)
    semantic_ids = load_jsonl(args.semantic_ids) if args.semantic_ids else []
    runs = []
    for spec in args.run:
        name, path = parse_run_spec(spec)
        runs.append(make_run(name, path))

    payload = build_explorer_payload(
        rows,
        skills,
        runs,
        cutoffs=cutoffs,
        title=args.title,
        semantic_ids=semantic_ids,
        strict_skill_metadata=not args.allow_missing_skill_metadata,
    )
    html = render_html(payload, default_cutoff=args.default_cutoff)
    write_text(args.output, html)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queries": payload["summary"]["query_count"],
                "relevant_skills": payload["summary"]["total_relevant_skills"],
                "runs": [run["name"] for run in payload["runs"]],
                "cutoffs": cutoffs,
                "default_cutoff": args.default_cutoff,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
