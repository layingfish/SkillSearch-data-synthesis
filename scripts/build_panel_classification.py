#!/usr/bin/env python3
"""Build panel-oriented classification labels for skill-search data synthesis.

The output is not a workflow proposal. It is an index for later ToolLLM-style
candidate-panel construction: task-space labels keep panels coherent, operation
labels provide functional coverage, and artifact/modality labels help a program
avoid incoherent panels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from workflow_role_taxonomy import infer_subrole


GENERIC_OPERATION = "general_capability"


SUBROLE_TO_OPERATION: dict[str, tuple[str, str]] = {
    "config_load": ("configure", "configuration_file_setup"),
    "tabular_load": ("load", "tabular_data_load"),
    "api_data_access": ("load", "api_or_database_access"),
    "document_file_load": ("load", "document_file_load"),
    "media_signal_load": ("load", "media_signal_load"),
    "web_search_retrieve": ("discover", "web_search_retrieval"),
    "structured_parse": ("extract", "structured_data_parse"),
    "table_extract": ("extract", "table_extract"),
    "text_media_extract": ("extract", "ocr_or_transcription"),
    "network_trace_extract": ("extract", "network_trace_extract"),
    "environment_setup": ("configure", "environment_setup"),
    "sampling_indexing": ("transform", "sampling_or_indexing"),
    "clean_filter_condition": ("transform", "clean_filter_condition"),
    "normalize_format_convert": ("transform", "format_normalization_conversion"),
    "mask_segmentation": ("transform", "mask_or_segmentation"),
    "merge_feature_engineering": ("transform", "merge_or_feature_engineering"),
    "control_logic": ("implement", "control_logic_implementation"),
    "physical_simulation": ("analyze", "physical_simulation"),
    "optimization_solver": ("analyze", "optimization_solver"),
    "software_implementation": ("implement", "software_implementation"),
    "ml_signal_inference": ("analyze", "ml_signal_inference"),
    "association_planning": ("analyze", "association_or_planning"),
    "metric_evaluation": ("validate", "metric_evaluation"),
    "security_analysis": ("validate", "security_audit"),
    "validation_check": ("validate", "validation_check"),
    "detection_classification": ("analyze", "detection_or_classification"),
    "comparison_impact": ("analyze", "comparison_or_impact_analysis"),
    "structured_output": ("generate", "structured_output_generation"),
    "report_document": ("generate", "report_generation"),
    "presentation_spreadsheet": ("generate", "presentation_or_spreadsheet"),
    "visualization": ("generate", "visualization"),
    "deploy_execute": ("operate", "deployment_or_execution"),
    "external_update": ("integrate", "external_integration_or_update"),
}


REFINEMENT_RULES: dict[str, list[tuple[str, list[str]]]] = {
    "software_implementation": [
        ("security_mitigation_implementation", ["mitigation", "input filter", "output guard", "defensive prompting", "red team", "prompt injection"]),
        ("language_or_dsl_design", ["language design", "syntax", "semantics", "grammar", "dsl", "domain language"]),
        ("technical_design_specification", ["technical design", "design document", "architecture guidance", "feature design", "proposal"]),
        ("desktop_or_mobile_ui_implementation", ["pyside", "qt", "swiftui", "ios", "macos", "android", "react native", "mobile app"]),
        ("agent_tool_implementation", ["agent", "agents", "tool", "mcp", "langchain", "openai agents", "livekit", "pipecat", "claude"]),
        ("frontend_component_implementation", ["frontend", "react", "vue", "svelte", "component", "ui component", "next.js", "tailwind", "css"]),
        ("backend_api_implementation", ["backend", "api", "server", "express", "fastapi", "django", "rails", "endpoint", "service"]),
        ("data_pipeline_implementation", ["etl", "pipeline", "data pipeline", "pandas", "spark", "dag", "workflow", "airflow", "dbt"]),
        ("test_case_implementation", ["test", "testing", "unit test", "integration test", "pytest", "vitest", "jest", "spec"]),
        ("infrastructure_implementation", ["docker", "kubernetes", "k8s", "terraform", "ansible", "ci", "github actions", "deployment"]),
        ("code_translation_or_migration", ["migrate", "migration", "translate", "port", "upgrade", "refactor", "rewrite"]),
        ("game_or_simulation_implementation", ["unity", "unreal", "game", "simulation", "scene", "shader"]),
        ("library_or_framework_usage", ["sdk", "library", "framework", "package", "plugin"]),
    ],
    "environment_setup": [
        ("game_engine_setup", ["unity", "unreal", "game engine", "scene setup"]),
        ("business_process_setup", ["crm", "sales funnel", "customer success", "onboarding", "playbook"]),
        ("analytics_or_trading_setup", ["chart analysis", "trade setup", "trading", "market", "smart money", "ict"]),
        ("sdk_or_framework_setup", ["sdk", "framework", "library", "toolchain", "provider"]),
        ("project_bootstrap_setup", ["project setup", "scaffold", "starter", "template", "create project", "new project"]),
        ("dependency_installation_setup", ["dependency", "dependencies", "install", "package manager", "npm", "pip", "uv", "cargo"]),
        ("infrastructure_environment_setup", ["docker", "kubernetes", "k8s", "terraform", "ansible", "vm", "server", "cloud", "ec2", "proxmox"]),
        ("configuration_profile_setup", ["profile", "settings", "config", "configuration", "env", "environment variable"]),
        ("prompt_or_model_configuration", ["prompt", "model", "llm", "temperature", "system prompt"]),
    ],
    "validation_check": [
        ("infrastructure_health_validation", ["cluster health", "cluster configuration", "network policy", "production incident", "triage"]),
        ("groundedness_or_citation_verification", ["citation", "grounded", "claims", "sources", "faithful", "factual"]),
        ("release_readiness_validation", ["pre-merge", "pull request", "release", "deployment", "production"]),
        ("unit_or_integration_test", ["unit test", "integration test", "pytest", "vitest", "jest", "test suite", "testing"]),
        ("debug_or_troubleshooting", ["debug", "troubleshoot", "fix", "error", "failure", "bug", "diagnose"]),
        ("security_review", ["security", "vulnerability", "cve", "threat", "audit", "risk"]),
        ("ui_or_design_review", ["ui", "ux", "visual", "design", "layout", "accessibility", "responsive"]),
        ("output_validation", ["validate output", "schema", "json", "result", "correctness", "verify output"]),
        ("performance_review", ["performance", "latency", "speed", "benchmark", "memory", "profile"]),
        ("documentation_review", ["documentation", "docs", "readme", "guide", "reference"]),
    ],
    "api_data_access": [
        ("version_control_access", ["git", "github", "pull request", "branch", "commit", "repository"]),
        ("knowledge_source_access", ["arxiv", "paper", "external resource", "article", "blog post", "knowledge base"]),
        ("automation_platform_access", ["github actions", "slack", "notion", "jira", "linear", "vercel"]),
        ("database_access", ["database", "sql", "sqlite", "postgres", "mysql", "query", "warehouse"]),
        ("external_api_access", ["api", "rest", "graphql", "endpoint", "client", "sdk"]),
        ("cloud_service_access", ["aws", "gcp", "azure", "cloud", "s3", "ec2", "lambda"]),
        ("scientific_data_access", ["obspy", "usgs", "nws", "pubchem", "mseed", "netcdf", "seismic"]),
    ],
    "tabular_load": [
        ("spreadsheet_load", ["spreadsheet", "excel", "xlsx", "google sheets", "sheet"]),
        ("csv_or_dataframe_load", ["csv", "dataframe", "pandas", "rows", "columns", "table"]),
    ],
    "config_load": [
        ("configuration_file_setup", ["yaml", "json config", "config file", "settings", "parameters", "env"]),
        ("prompt_parameter_configuration", ["prompt", "midjourney", "flux", "parameter", "style", "model"]),
    ],
}


OPERATION_OVERRIDE_RULES: list[tuple[str, str, list[str]]] = [
    ("generate", "diagram_generation", ["svg diagram", "plantuml", "mermaid", "diagram generation", "satori"]),
    ("generate", "release_note_generation", ["release notes", "changelog", "app store release"]),
    ("generate", "technical_design_specification", ["technical design document", "feature technical design", "architecture document"]),
    ("generate", "learning_material_generation", ["learning graph", "course description", "lesson plan", "curriculum"]),
    ("analyze", "market_or_financial_analysis", ["chart analysis", "trade setup", "trading", "smart money", "ict-style", "market analysis"]),
    ("analyze", "language_or_dsl_design", ["language design", "syntax decisions", "semantics", "grammar authoring", "domainlang", "dsl"]),
    ("validate", "groundedness_or_citation_verification", ["citation verifier", "provided sources", "strictly grounded", "faithful", "factual"]),
    ("validate", "infrastructure_health_validation", ["cluster health", "cluster configuration", "network policy", "production incident"]),
    ("operate", "deployment_automation", ["deploying to production", "vercel deployment", "ci/cd pipeline", "production deployment"]),
    ("implement", "security_mitigation_implementation", ["implement mitigations", "input filters", "output guards", "defensive prompting"]),
]


ARTIFACT_KEYWORDS: dict[str, list[str]] = {
    "requirements": ["requirement", "spec", "brief", "user story", "acceptance criteria"],
    "documentation": ["documentation", "docs", "readme", "guide", "manual", "reference"],
    "source_code": ["code", "source", "repository", "repo", "component", "function", "class", "module"],
    "configuration": ["config", "configuration", "settings", "yaml", "toml", "env", "profile"],
    "api_schema": ["api", "endpoint", "openapi", "swagger", "graphql", "schema"],
    "database": ["database", "sql", "sqlite", "postgres", "mysql", "query"],
    "tabular_data": ["csv", "xlsx", "spreadsheet", "dataframe", "table", "rows", "columns"],
    "document": ["pdf", "docx", "document", "presentation", "pptx", "markdown"],
    "image": ["image", "png", "jpeg", "jpg", "screenshot", "photo", "diagram"],
    "audio": ["audio", "speech", "voice", "transcribe", "stt", "tts"],
    "video": ["video", "frame", "subtitle", "clip"],
    "web_page": ["web", "browser", "html", "crawl", "scrape", "website"],
    "test_results": ["test result", "coverage", "failure", "traceback", "log", "ci"],
    "metrics": ["metric", "score", "benchmark", "performance", "latency", "rmse", "f1"],
    "report": ["report", "summary", "analysis", "findings"],
    "visualization": ["chart", "plot", "dashboard", "visualization", "graph"],
    "deployment": ["deploy", "deployment", "docker", "kubernetes", "server", "cloud"],
    "agent_tool": ["agent", "tool calling", "mcp", "langchain", "crewai", "pipecat"],
    "skill_package": ["skill", "plugin", "extension", "marketplace"],
    "prompt_template": ["prompt", "system prompt", "instruction"],
    "ui_component": ["component", "ui", "frontend", "react", "vue", "swiftui"],
    "api_endpoint": ["endpoint", "route", "handler", "client"],
    "test_suite": ["test suite", "unit test", "integration test", "pytest", "jest", "vitest"],
    "ci_pipeline": ["ci", "ci/cd", "github actions", "workflow"],
    "cloud_resource": ["aws", "gcp", "azure", "s3", "ec2", "lambda"],
    "security_control": ["security control", "policy", "guard", "mitigation", "vulnerability"],
    "research_evidence": ["paper", "citation", "source", "literature", "arxiv"],
    "business_record": ["crm", "sales", "customer", "funnel"],
}


MODALITY_KEYWORDS: dict[str, list[str]] = {
    "code": ["code", "source", "function", "class", "component", "sdk", "library", "framework"],
    "text": ["text", "markdown", "prompt", "copy", "story", "summary", "document"],
    "table": ["table", "csv", "xlsx", "spreadsheet", "dataframe", "rows", "columns"],
    "document": ["pdf", "docx", "pptx", "document", "presentation"],
    "image": ["image", "screenshot", "png", "jpeg", "photo", "diagram"],
    "audio": ["audio", "voice", "speech", "transcribe", "tts", "stt"],
    "video": ["video", "frame", "subtitle", "clip"],
    "api": ["api", "endpoint", "graphql", "rest", "webhook", "mcp"],
    "database": ["database", "sql", "sqlite", "postgres", "mysql"],
    "web": ["web", "browser", "html", "crawl", "scrape", "website"],
}


TASK_OBJECT_KEYWORDS: dict[str, list[str]] = {
    "llm_agent": ["agent", "agents", "agentic", "tool calling", "langchain", "crewai", "openai agents", "pipecat", "livekit"],
    "skill_or_plugin": ["skill creation", "skill development", "skill package", "codex skill", "claude skill", "plugin", "extension", "marketplace"],
    "prompt": ["prompt", "system prompt", "instruction", "prompt injection", "defensive prompting"],
    "frontend_ui": ["frontend", "ui", "ux", "react", "vue", "svelte", "component", "tailwind", "css", "html"],
    "backend_api": ["backend", "api", "endpoint", "rest", "graphql", "fastapi", "django", "rails", "express", "server"],
    "database": ["database", "sql", "sqlite", "postgres", "mysql", "warehouse", "query", "schema migration"],
    "test_suite": ["test", "testing", "test suite", "unit test", "integration test", "pytest", "jest", "vitest"],
    "ci_cd_pipeline": ["ci/cd", "github actions", "build pipeline", "deployment pipeline"],
    "container_orchestration": ["docker", "kubernetes", "k8s", "container", "service mesh", "helm"],
    "cloud_infrastructure": ["aws", "gcp", "azure", "cloud", "terraform", "serverless", "lambda", "ec2", "s3"],
    "observability_stack": ["observability", "monitoring", "logs", "metrics", "tracing", "alert"],
    "security_policy": ["security", "vulnerability", "cve", "threat", "audit", "risk", "oscal", "nist", "fedramp"],
    "version_control": ["git", "github", "pull request", "branch", "commit", "merge"],
    "issue_or_project": ["issue", "ticket", "jira", "linear", "kanban", "project management", "task management"],
    "documentation_site": ["documentation", "docs", "readme", "manual"],
    "software_architecture": ["architecture", "architectural", "design pattern", "system design", "technology assessment"],
    "performance_profile": ["performance", "optimization", "latency", "benchmark", "profiling"],
    "development_environment": ["developer tooling", "project setup", "dev environment", "tooling environment"],
    "design_system": ["design system", "brand", "style guide", "visual design", "layout", "accessibility"],
    "mobile_app": ["ios", "android", "mobile", "swiftui", "react native", "macos"],
    "game_scene": ["unity", "unreal", "game", "scene", "shader", "physics"],
    "workflow_automation": ["workflow orchestration", "automation workflow", "orchestration", "automate task"],
    "creative_story": ["creative writing", "storytelling", "scene", "narrative", "story"],
    "web_page": ["web page", "website", "browser", "crawl", "scrape", "html"],
    "spreadsheet": ["spreadsheet", "excel", "xlsx", "google sheets", "sheet"],
    "tabular_dataset": ["csv", "dataframe", "table", "rows", "columns", "tabular"],
    "document_corpus": ["pdf", "docx", "document", "corpus", "presentation", "pptx"],
    "research_paper": ["paper", "arxiv", "literature", "citation", "sources", "research"],
    "image_asset": ["image", "screenshot", "photo", "png", "jpeg", "diagram", "svg"],
    "audio_stream": ["audio", "voice", "speech", "transcribe", "tts", "stt"],
    "video_media": ["video", "frame", "subtitle", "clip"],
    "time_series": ["time series", "sensor", "seismic", "waveform", "mseed", "lightcurve", "climate"],
    "network_traffic": ["pcap", "packet", "network trace", "beaconing"],
    "geospatial_data": ["map", "geospatial", "coordinate", "spatial", "path planning"],
    "ml_model": ["model", "deep learning", "inference", "classification", "training", "embedding model"],
    "embedding_index": ["embedding", "vector", "semantic search", "retrieval index"],
    "report_document": ["report", "summary", "findings", "analysis report"],
    "presentation_deck": ["presentation", "slides", "pptx", "deck"],
    "crm_sales_process": ["crm", "sales", "funnel", "customer success", "churn"],
    "financial_market_data": ["financial", "finance", "financial market", "market data", "trading", "stock", "smart money", "lmp"],
    "healthcare_record": ["healthcare", "clinical", "patient", "medical", "rct", "consort"],
    "legal_contract": ["legal", "contract", "clause"],
    "educational_material": ["course", "lesson", "curriculum", "learning graph", "teaching"],
    "marketing_content": ["marketing", "campaign", "seo", "copy", "social media"],
    "product_requirement": ["product requirement", "prd", "user story", "acceptance criteria"],
    "codebase": ["code", "source", "repository", "repo", "function", "class", "module"],
}


TASK_OBJECT_FAMILY: dict[str, str] = {
    "llm_agent": "software_ai",
    "skill_or_plugin": "software_ai",
    "prompt": "software_ai",
    "frontend_ui": "software_system",
    "backend_api": "software_system",
    "database": "software_data",
    "test_suite": "software_quality",
    "ci_cd_pipeline": "software_delivery",
    "container_orchestration": "software_delivery",
    "cloud_infrastructure": "software_delivery",
    "observability_stack": "software_operations",
    "security_policy": "software_security",
    "version_control": "software_workflow",
    "issue_or_project": "project_workflow",
    "documentation_site": "knowledge_content",
    "software_architecture": "software_architecture",
    "performance_profile": "software_quality",
    "development_environment": "software_workflow",
    "design_system": "design_content",
    "mobile_app": "software_system",
    "game_scene": "interactive_media",
    "workflow_automation": "automation_workflow",
    "creative_story": "creative_content",
    "web_page": "web_content",
    "spreadsheet": "structured_data",
    "tabular_dataset": "structured_data",
    "document_corpus": "document_content",
    "research_paper": "research_content",
    "image_asset": "visual_media",
    "audio_stream": "audio_media",
    "video_media": "video_media",
    "time_series": "scientific_data",
    "network_traffic": "security_data",
    "geospatial_data": "spatial_data",
    "ml_model": "ml_system",
    "embedding_index": "ml_system",
    "report_document": "knowledge_content",
    "presentation_deck": "knowledge_content",
    "crm_sales_process": "business_process",
    "financial_market_data": "business_finance",
    "healthcare_record": "domain_healthcare",
    "legal_contract": "domain_legal",
    "educational_material": "domain_education",
    "marketing_content": "business_marketing",
    "product_requirement": "product_workflow",
    "codebase": "software_system",
}


TASK_OBJECT_ARTIFACT: dict[str, str] = {
    "llm_agent": "agent_tool",
    "skill_or_plugin": "skill_package",
    "prompt": "prompt_template",
    "frontend_ui": "ui_component",
    "backend_api": "api_endpoint",
    "database": "database",
    "test_suite": "test_suite",
    "ci_cd_pipeline": "ci_pipeline",
    "container_orchestration": "deployment",
    "cloud_infrastructure": "cloud_resource",
    "observability_stack": "metrics",
    "security_policy": "security_control",
    "version_control": "source_code",
    "issue_or_project": "requirements",
    "documentation_site": "documentation",
    "software_architecture": "requirements",
    "performance_profile": "metrics",
    "development_environment": "configuration",
    "design_system": "ui_component",
    "mobile_app": "ui_component",
    "game_scene": "source_code",
    "workflow_automation": "configuration",
    "creative_story": "document",
    "web_page": "web_page",
    "spreadsheet": "tabular_data",
    "tabular_dataset": "tabular_data",
    "document_corpus": "document",
    "research_paper": "research_evidence",
    "image_asset": "image",
    "audio_stream": "audio",
    "video_media": "video",
    "time_series": "metrics",
    "network_traffic": "test_results",
    "geospatial_data": "metrics",
    "ml_model": "metrics",
    "embedding_index": "database",
    "report_document": "report",
    "presentation_deck": "document",
    "crm_sales_process": "business_record",
    "financial_market_data": "metrics",
    "healthcare_record": "document",
    "legal_contract": "document",
    "educational_material": "documentation",
    "marketing_content": "report",
    "product_requirement": "requirements",
    "codebase": "source_code",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def value(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False)


def row_text(row: dict[str, Any], body_chars: int) -> str:
    fields = [
        row.get("name"),
        row.get("description"),
        row.get("task_description"),
        row.get("domain"),
        row.get("namespace_prefix"),
        row.get("tags"),
        row.get("body", "")[:body_chars],
    ]
    return " ".join(value(field) for field in fields if value(field)).lower()


def row_local_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("name"),
        row.get("description"),
        row.get("task_description"),
        row.get("domain"),
        row.get("namespace_prefix"),
        row.get("tags"),
    ]
    return " ".join(value(field) for field in fields if value(field)).lower()


def row_context_text(row: dict[str, Any]) -> str:
    raw_cluster = str(row.get("coarse_cluster_id") or row.get("source_cluster_id") or row.get("cluster_id") or "")
    raw_cluster = raw_cluster.split("::role=")[0]
    return f"{raw_cluster} {slugify(raw_cluster).replace('_', ' ')}".lower()


def row_core_text(row: dict[str, Any]) -> str:
    fields = [
        row_local_text(row),
        row_context_text(row),
    ]
    return " ".join(value(field) for field in fields if value(field)).lower()


def slugify(text: str) -> str:
    text = text.strip().lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def raw_task_space_from_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    raw = str(row.get("coarse_cluster_id") or row.get("source_cluster_id") or row.get("cluster_id") or "unknown")
    raw = raw.split("::role=")[0]
    parts = [slugify(part) for part in raw.split("/") if part.strip()]
    if not parts:
        parts = ["unknown"]
    return "/".join(parts), parts


def keyword_score(text: str, keywords: list[str]) -> int:
    score = 0
    tokens = set(re.findall(r"[a-z][a-z0-9_+-]+", text))
    for keyword in keywords:
        key = keyword.lower()
        if " " in key or "-" in key:
            if key in text:
                score += 2
        elif key in tokens:
            score += 1
    return score


def refine_operation_type(subrole: str, base_type: str, text: str) -> tuple[str, int]:
    rules = REFINEMENT_RULES.get(subrole, [])
    best_type = base_type
    best_score = 0
    for candidate, keywords in rules:
        score = keyword_score(text, keywords)
        if score > best_score:
            best_type = candidate
            best_score = score
    return best_type, best_score


def operation_override(text: str) -> tuple[str, str, int] | None:
    best: tuple[str, str, int] | None = None
    for role, operation_type, keywords in OPERATION_OVERRIDE_RULES:
        score = keyword_score(text, keywords)
        if score >= 2 and (best is None or score > best[2]):
            best = (role, operation_type, score)
    return best


def operation_from_row(row: dict[str, Any], text: str) -> tuple[str, str, float, list[str]]:
    subrole = str(row.get("workflow_subrole") or row.get("operation_source_subrole") or "")
    if not subrole:
        subrole = infer_subrole(row)
    if subrole.startswith("generic_"):
        override = operation_override(text)
        if override is None:
            return GENERIC_OPERATION, subrole, 0.45, ["generic_operation_label"]
        role, operation_type, _ = override
        return role, operation_type, 0.68, ["generic_operation_repaired"]

    role, base_type = SUBROLE_TO_OPERATION.get(subrole, (GENERIC_OPERATION, subrole))
    operation_type, refine_score = refine_operation_type(subrole, base_type, text)
    confidence = 0.82
    flags: list[str] = []
    if operation_type != base_type:
        confidence = 0.9
    elif subrole in REFINEMENT_RULES:
        confidence = 0.72
        flags.append("coarse_operation_type")
    if role == GENERIC_OPERATION:
        confidence = min(confidence, 0.5)
        flags.append("unknown_operation_mapping")
    if refine_score == 0 and subrole in {"software_implementation", "environment_setup", "validation_check"}:
        flags.append("large_bucket_without_refinement")
    override = operation_override(text)
    if override is not None and override[2] > refine_score:
        role, operation_type, _ = override
        confidence = max(confidence, 0.88)
        flags = [flag for flag in flags if flag not in {"coarse_operation_type", "large_bucket_without_refinement"}]
        flags.append("operation_reassigned_by_text")
    return role, operation_type, confidence, flags


def matched_labels(text: str, specs: dict[str, list[str]]) -> list[str]:
    labels = [label for label, keywords in specs.items() if keyword_score(text, keywords) > 0]
    return sorted(labels)


def ranked_labels(text: str, specs: dict[str, list[str]], limit: int = 5) -> list[tuple[str, int]]:
    scored = [
        (label, keyword_score(text, keywords))
        for label, keywords in specs.items()
    ]
    scored = [(label, score) for label, score in scored if score > 0]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def task_objects_from_text(text: str) -> tuple[list[str], float, list[str]]:
    ranked = ranked_labels(text, TASK_OBJECT_KEYWORDS, limit=6)
    labels = [label for label, _ in ranked]
    flags: list[str] = []

    specific = [label for label in labels if label != "codebase"]
    if specific:
        labels = specific[:5]
    else:
        labels = labels[:3]

    if not labels:
        flags.append("weak_task_object_signal")
        return [], 0.35, flags

    top_score = ranked[0][1]
    confidence = min(0.95, 0.45 + 0.1 * len(labels) + 0.04 * top_score)
    return labels, round(confidence, 3), flags


def merged_task_objects(row: dict[str, Any]) -> tuple[list[str], list[str], list[str], float, str, list[str]]:
    local_objects, local_confidence, _ = task_objects_from_text(row_local_text(row))
    context_objects, context_confidence, _ = task_objects_from_text(row_context_text(row))
    flags: list[str] = []

    if local_objects:
        merged = list(dict.fromkeys(local_objects + context_objects))[:5]
        source = "local_plus_context" if context_objects else "local"
        confidence = min(0.95, local_confidence + (0.03 if context_objects else 0.0))
    elif context_objects:
        merged = context_objects[:4]
        source = "context_only"
        confidence = min(0.68, context_confidence)
        flags.append("context_only_task_object")
    else:
        merged = []
        source = "none"
        confidence = 0.35
        flags.append("weak_task_object_signal")

    return merged, local_objects, context_objects, round(confidence, 3), source, flags


def repair_task_space(
    raw_task_space_id: str,
    raw_task_space_path: list[str],
    task_objects: list[str],
    operation_type: str,
) -> tuple[str, list[str], float, list[str], str]:
    if len(raw_task_space_path) >= 3:
        return raw_task_space_id, raw_task_space_path, 0.9, [], "agentskillos_depth3"

    flags: list[str] = []
    source = "fallback"
    if task_objects:
        suffix = task_objects[0]
        confidence = 0.84
        source = "task_object"
        flags.append("repaired_coarse_task_space")
    elif operation_type and operation_type != GENERIC_OPERATION:
        suffix = operation_type
        confidence = 0.74
        source = "operation_type"
        flags.extend(["repaired_coarse_task_space", "operation_based_task_space"])
    else:
        suffix = "general"
        confidence = 0.62
        flags.append("coarse_task_space")

    suffix = slugify(suffix)
    repaired = list(raw_task_space_path)
    if not repaired or repaired[-1] != suffix:
        repaired.append(suffix)
    return "/".join(repaired), repaired, confidence, flags, source


def artifact_io(operation_role: str, operation_type: str, artifacts: list[str]) -> tuple[list[str], list[str]]:
    artifact_set = set(artifacts)
    if operation_role == "discover":
        return ["search_query"], sorted(artifact_set | {"retrieved_context"})
    if operation_role == "load":
        return sorted(artifact_set or {"external_resource"}), sorted(artifact_set | {"loaded_data"})
    if operation_role == "extract":
        return sorted(artifact_set or {"raw_artifact"}), sorted(artifact_set | {"extracted_content"})
    if operation_role == "transform":
        return sorted(artifact_set or {"input_artifact"}), sorted(artifact_set | {"transformed_artifact"})
    if operation_role == "configure":
        return sorted(artifact_set | {"requirements"}), sorted(artifact_set | {"configuration"})
    if operation_role == "implement":
        return sorted(artifact_set | {"requirements"}), sorted(artifact_set | {"source_code"})
    if operation_role == "analyze":
        return sorted(artifact_set or {"input_artifact"}), sorted(artifact_set | {"analysis_result"})
    if operation_role == "validate":
        return sorted(artifact_set or {"candidate_output"}), sorted(artifact_set | {"validation_result"})
    if operation_role == "generate":
        return sorted(artifact_set | {"requirements"}), sorted(artifact_set | {"generated_artifact"})
    if operation_role == "integrate":
        return sorted(artifact_set | {"source_artifact"}), sorted(artifact_set | {"integrated_system"})
    if operation_role == "operate":
        return sorted(artifact_set | {"runtime_target"}), sorted(artifact_set | {"operation_result"})
    return sorted(artifact_set), sorted(artifact_set)


def classify_kind_and_scope(text: str, operation_role: str) -> tuple[str, str, list[str]]:
    flags: list[str] = []
    if any(term in text for term in ["validator", "validate", "reviewer", "auditor"]):
        kind = "validator"
    elif any(term in text for term in ["framework", "sdk", "library", "patterns", "best practices"]):
        kind = "framework_or_guide"
    elif any(term in text for term in ["generate", "generator", "create", "writer"]):
        kind = "generator"
    elif operation_role in {"operate", "integrate"}:
        kind = "automation_or_integration"
    elif any(term in text for term in ["reference", "cheatsheet", "overview", "principles"]):
        kind = "reference_guide"
    else:
        kind = "actionable_guide"

    if any(term in text for term in ["end-to-end", "workflow", "pipeline", "orchestrate", "multi-step"]):
        scope = "workflow_like"
        flags.append("workflow_like_skill")
    elif any(term in text for term in ["comprehensive", "complete guide", "framework", "all-in-one"]):
        scope = "composite"
        flags.append("broad_skill")
    elif kind == "reference_guide":
        scope = "reference"
    else:
        scope = "atomic"
    return kind, scope, flags


def classify_row(row: dict[str, Any], body_chars: int) -> dict[str, Any]:
    text = row_text(row, body_chars)
    raw_task_space_id, raw_task_space_path = raw_task_space_from_row(row)
    operation_role, operation_type, operation_confidence, operation_flags = operation_from_row(row, text)
    task_objects, local_task_objects, context_task_objects, task_object_confidence, task_object_source, task_object_flags = merged_task_objects(row)
    task_space_id, task_space_path, task_confidence, task_space_flags, task_space_source = repair_task_space(
        raw_task_space_id,
        raw_task_space_path,
        task_objects,
        operation_type,
    )
    object_artifacts = [TASK_OBJECT_ARTIFACT[obj] for obj in task_objects if obj in TASK_OBJECT_ARTIFACT]
    artifacts = sorted(set(matched_labels(text, ARTIFACT_KEYWORDS)) | set(object_artifacts))
    modalities = matched_labels(text, MODALITY_KEYWORDS)
    input_artifacts, output_artifacts = artifact_io(operation_role, operation_type, artifacts)
    skill_kind, scope_level, scope_flags = classify_kind_and_scope(text, operation_role)
    task_object_families = sorted({TASK_OBJECT_FAMILY[obj] for obj in task_objects if obj in TASK_OBJECT_FAMILY})

    flags = list(dict.fromkeys(operation_flags + task_object_flags + task_space_flags + scope_flags))
    if not artifacts:
        flags.append("weak_artifact_signal")
    if not modalities:
        flags.append("weak_modality_signal")
    artifact_confidence = min(0.94, 0.42 + 0.1 * len(artifacts) + 0.05 * len(modalities) + 0.04 * len(task_objects))

    out = dict(row)
    out.update({
        "skill_id": row.get("id") or row.get("skill_id"),
        "skill_name": row.get("name"),
        "source_task_space_id": raw_task_space_id,
        "source_task_space_path": raw_task_space_path,
        "task_space_id": task_space_id,
        "task_space_path": task_space_path,
        "task_space_l1": task_space_path[0] if len(task_space_path) > 0 else "unknown",
        "task_space_l2": task_space_path[1] if len(task_space_path) > 1 else "unknown",
        "task_space_l3": task_space_path[2] if len(task_space_path) > 2 else "unknown",
        "task_space_refinement_source": task_space_source,
        "operation_role": operation_role,
        "operation_type": operation_type,
        "operation_source_subrole": row.get("workflow_subrole") or infer_subrole(row),
        "task_object_types": task_objects,
        "local_task_object_types": local_task_objects,
        "context_task_object_types": context_task_objects,
        "primary_task_object_type": task_objects[0] if task_objects else "unknown",
        "task_object_signal_source": task_object_source,
        "task_object_families": task_object_families,
        "input_artifact_types": input_artifacts,
        "output_artifact_types": output_artifacts,
        "modality_types": modalities,
        "skill_kind": skill_kind,
        "scope_level": scope_level,
        "classification_confidence": {
            "task_space": round(task_confidence, 3),
            "operation": round(operation_confidence, 3),
            "artifact": round(artifact_confidence, 3),
            "task_object": round(task_object_confidence, 3),
        },
        "quality_flags": flags,
        "panel_classification_version": "v2_rule_based_task_object",
    })
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "task_spaces": len({row["task_space_id"] for row in rows}),
        "operation_roles": Counter(row["operation_role"] for row in rows).most_common(),
        "operation_types_top": Counter(row["operation_type"] for row in rows).most_common(25),
        "task_object_types_top": Counter(obj for row in rows for obj in row["task_object_types"]).most_common(30),
        "local_task_object_types_top": Counter(obj for row in rows for obj in row["local_task_object_types"]).most_common(30),
        "context_task_object_types_top": Counter(obj for row in rows for obj in row["context_task_object_types"]).most_common(30),
        "task_object_families_top": Counter(fam for row in rows for fam in row["task_object_families"]).most_common(25),
        "task_object_signal_sources": Counter(row["task_object_signal_source"] for row in rows).most_common(),
        "task_space_refinement_sources": Counter(row["task_space_refinement_source"] for row in rows).most_common(),
        "scope_levels": Counter(row["scope_level"] for row in rows).most_common(),
        "skill_kinds": Counter(row["skill_kind"] for row in rows).most_common(),
        "quality_flags_top": Counter(flag for row in rows for flag in row["quality_flags"]).most_common(25),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--body-chars", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    classified = [classify_row(row, args.body_chars) for row in rows]
    count = write_jsonl(args.output, classified)
    summary = summarize(classified)
    summary["input"] = str(args.input)
    summary["output"] = str(args.output)
    summary["written"] = count
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
