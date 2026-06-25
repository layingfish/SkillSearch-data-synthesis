#!/usr/bin/env python3
"""Workflow role and subrole inference for skill-composition experiments."""

from __future__ import annotations

from typing import Any

from evaluate_cluster_composability import ROLE_ORDER, infer_role, text_tokens, value


SUBROLE_SPECS: list[tuple[str, str, list[str]]] = [
    ("config_load", "acquire_load", ["yaml config", "json config", "config file", "configuration file", "parameter file", "input parameters", "parameters"]),
    ("tabular_load", "acquire_load", ["csv", "xlsx", "excel", "spreadsheet", "dataframe", "rows", "columns"]),
    ("api_data_access", "acquire_load", ["api data", "database", "sql", "sqlite", "obspy", "usgs", "nws", "pubchem", "download", "fetch"]),
    ("document_file_load", "acquire_load", ["pdf", "pptx", "docx", "file", "filesystem", "presentation", "document", "open"]),
    ("media_signal_load", "acquire_load", ["video", "audio", "image", "mseed", "waveform", "pcap", "packets", "sensor", "lightcurve", "gwf"]),
    ("web_search_retrieve", "acquire_load", ["search", "crawl", "web", "browser", "retrieve", "collect", "find"]),
    ("structured_parse", "extract_parse", ["parse", "parser", "json", "yaml", "xml", "cif", "matpower", "scenario", "schema"]),
    ("table_extract", "extract_parse", ["extract", "table", "embedded", "spreadsheet", "form", "cell", "formula"]),
    ("text_media_extract", "extract_parse", ["ocr", "transcribe", "subtitle", "speech", "audio transcript", "text extraction"]),
    ("network_trace_extract", "extract_parse", ["pcap", "packet", "flow", "protocol", "port", "entropy"]),
    ("environment_setup", "prepare_transform", ["setup", "configure", "configuration", "settings", "dependency", "dependencies", "install", "profile", "environment", "tool binding"]),
    ("sampling_indexing", "prepare_transform", ["sample", "sampling", "index", "fps", "frame", "interval", "window"]),
    ("clean_filter_condition", "prepare_transform", ["clean", "filter", "outlier", "quality", "detrend", "remove", "condition", "preprocess"]),
    ("normalize_format_convert", "prepare_transform", ["normalize", "format", "convert", "resize", "csr", "sparse", "rational", "coordinate", "structure"]),
    ("mask_segmentation", "prepare_transform", ["mask", "segmentation", "dynamic-object", "binary", "csr"]),
    ("merge_feature_engineering", "prepare_transform", ["merge", "feature", "combine", "split", "vector", "embedding"]),
    ("control_logic", "model_compute", ["pid", "controller", "adaptive cruise", "acc", "control", "feedback", "throttle", "brake"]),
    ("physical_simulation", "model_compute", ["simulate", "simulation", "dynamics", "vehicle", "lake model", "glm", "power-flow", "egomotion", "motion"]),
    ("optimization_solver", "model_compute", ["optimize", "optimizer", "solver", "ipopt", "casadi", "dispatch", "economic", "placement", "adjacency", "least-cost", "nlp"]),
    ("software_implementation", "model_compute", ["implement", "implementation", "build", "develop", "code", "component", "service", "sdk", "framework", "agent", "tool"]),
    ("ml_signal_inference", "model_compute", ["model", "seisbench", "deep learning", "picker", "fingerprint", "tanimoto", "matched", "periodogram", "transit", "lomb", "classification"]),
    ("association_planning", "model_compute", ["association", "associate", "route", "planning", "path", "graph", "hex", "spatial"]),
    ("metric_evaluation", "analyze_validate", ["metric", "metrics", "score", "f1", "rmse", "performance", "overshoot", "rise time", "steady-state", "coefficient"]),
    ("security_analysis", "analyze_validate", ["vulnerability", "cvss", "threat", "scan", "cve", "audit", "risk", "trivy"]),
    ("validation_check", "analyze_validate", ["validate", "validation", "verify", "check", "test", "debug", "inspect", "self-check", "output-validation"]),
    ("detection_classification", "analyze_validate", ["detect", "detection", "classify", "identify", "anomaly", "period", "beaconing"]),
    ("comparison_impact", "analyze_validate", ["compare", "impact", "counterfactual", "what if", "lmp", "drop"]),
    ("structured_output", "generate_report", ["json", "csv", "yaml", "npz", "txt", "save", "output", "results"]),
    ("report_document", "generate_report", ["report", "markdown", "summary", "document", "write", "draft"]),
    ("presentation_spreadsheet", "generate_report", ["pptx", "presentation", "xlsx", "excel", "spreadsheet"]),
    ("visualization", "generate_report", ["plot", "chart", "dashboard", "visualize", "render"]),
    ("deploy_execute", "act_integrate", ["run", "execute", "deploy", "publish", "launch", "schedule", "automate", "orchestrate", "pipeline"]),
    ("external_update", "act_integrate", ["send", "notify", "update", "browser", "integrate", "integration", "mcp"]),
]

SUBROLE_TO_ROLE = {subrole: role for subrole, role, _ in SUBROLE_SPECS}
SUBROLE_ORDER = {
    subrole: ROLE_ORDER.get(role, 3) * 100 + index
    for index, (subrole, role, _) in enumerate(SUBROLE_SPECS)
}


def _field_texts(row: dict[str, Any]) -> list[tuple[float, str]]:
    return [
        (4.0, value(row.get("name")).lower()),
        (2.5, (value(row.get("description")) + " " + value(row.get("task_description"))).lower()),
        (0.45, value(row.get("body"))[:1500].lower()),
        (0.5, value(row.get("domain")) + " " + value(row.get("namespace_prefix"))),
    ]


def infer_subrole(row: dict[str, Any]) -> str:
    explicit = value(row.get("workflow_subrole")).strip()
    if explicit:
        return explicit

    field_texts = _field_texts(row)
    tokenized_fields = [(weight, text, text_tokens(text)) for weight, text in field_texts]
    scores: dict[str, float] = {}
    for subrole, _, keywords in SUBROLE_SPECS:
        score = 0.0
        for weight, text, toks in tokenized_fields:
            for kw in keywords:
                key = kw.lower()
                if " " in key or "-" in key or "_" in key:
                    if key in text:
                        score += 2.0 * weight
                elif key in toks:
                    score += 1.0 * weight
        scores[subrole] = score

    best_subrole, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_subrole
    return f"generic_{infer_role(row)}"


def subrole_parent_role(subrole: str) -> str:
    if subrole in SUBROLE_TO_ROLE:
        return SUBROLE_TO_ROLE[subrole]
    if subrole.startswith("generic_"):
        return subrole.removeprefix("generic_")
    return "general_capability"


def subrole_order(subrole: str) -> int:
    return SUBROLE_ORDER.get(subrole, ROLE_ORDER.get(subrole_parent_role(subrole), 3) * 100 + 99)
