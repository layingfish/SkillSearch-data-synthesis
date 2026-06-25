#!/usr/bin/env python3
"""Replay the 2026-06-01 400-row pilot with the current query prompts.

The historical pilot already contains the sampled single skills and redundant
multi-skill candidate panels. This script keeps those rows fixed and only
regenerates the query payload with the current SkillRouter-aligned prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import generate_multiskill_probe as multi  # noqa: E402
import generate_single_skill_probe as single  # noqa: E402


DEFAULT_INPUTS = {
    1: Path("outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_1skill_combined_100.jsonl"),
    2: Path("outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_2skills_combined_100_manual_clean.jsonl"),
    3: Path("outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_3skills_combined_100_manual_clean.jsonl"),
    4: Path("outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_4skills_combined_100_clean.jsonl"),
}


STAT_ONLY_WARNINGS = {"query_too_long"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    for key in ["base_url", "api_key", "model"]:
        if not profile.get(key):
            raise SystemExit(f"profile missing required field: {key}")
    return profile


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    if "api_key" in out:
        out["api_key"] = "<redacted>"
    return out


def configure_profile_transport(profile: dict[str, Any]) -> None:
    proxy_url = str(profile.get("proxy_url") or "").strip()
    if proxy_url:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
        urllib.request.install_opener(opener)

    extra_headers = profile.get("extra_headers") or {}
    if isinstance(extra_headers, dict) and extra_headers:
        single.DEFAULT_EXTRA_HEADERS = {
            str(key): str(value)
            for key, value in extra_headers.items()
        }
        multi.DEFAULT_EXTRA_HEADERS = dict(single.DEFAULT_EXTRA_HEADERS)


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def load_skill_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = skill_id(row)
        if sid:
            lookup[sid] = row
    return lookup


def panel_skill_from_lookup(sid: str, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = lookup.get(sid, {})
    return {
        "skill_id": sid,
        "skill_name": full.get("skill_name") or full.get("name") or sid,
        "operation_role": full.get("operation_role"),
        "operation_type": full.get("operation_type"),
        "task_object_types": full.get("task_object_types") or [],
        "task_object_signal_source": full.get("task_object_signal_source"),
        "input_artifact_types": full.get("input_artifact_types") or [],
        "output_artifact_types": full.get("output_artifact_types") or [],
        "modality_types": full.get("modality_types") or [],
    }


def build_replay_panel(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = [str(sid) for sid in (row.get("candidate_skill_ids") or []) if sid]
    if not candidate_ids:
        old_selected = (row.get("generated") or {}).get("selected_skill_ids") or []
        candidate_ids = [str(sid) for sid in old_selected if sid]
    return {
        "task_space_id": row.get("task_space_id"),
        "anchor_task_object": row.get("anchor_task_object"),
        "skills": [panel_skill_from_lookup(sid, lookup) for sid in candidate_ids],
    }


def normalize_id_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def basename(skill_id_value: str) -> str:
    return str(skill_id_value or "").rsplit("/", 1)[-1]


def repair_selected_ids(
    generated: dict[str, Any],
    candidate_ids: list[str],
    panel_skills: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [str(sid).strip() for sid in (generated.get("selected_skill_ids") or [])]
    if not selected:
        return generated
    repaired: list[str] = []
    repairs: list[dict[str, str]] = []

    candidate_by_lower = {sid.lower(): sid for sid in candidate_ids}
    by_basename: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        by_basename.setdefault(basename(candidate_id).lower(), []).append(candidate_id)
    by_skill_name: dict[str, list[str]] = {}
    for skill in panel_skills:
        candidate_id = str(skill.get("skill_id") or "")
        name_key = normalize_id_text(str(skill.get("skill_name") or ""))
        if candidate_id and name_key:
            by_skill_name.setdefault(name_key, []).append(candidate_id)

    for raw_sid in selected:
        sid = re.sub(r"^\s*\d+\s*[\).\:-]\s*", "", raw_sid).strip()
        replacement = ""
        repair_type = ""
        if sid.isdigit():
            index = int(sid)
            if 1 <= index <= len(candidate_ids):
                replacement = candidate_ids[index - 1]
                repair_type = "candidate_number_to_skill_id"
        if not replacement and sid.lower() in candidate_by_lower:
            replacement = candidate_by_lower[sid.lower()]
            repair_type = "case_normalization"
        if not replacement and sid.lower() in by_basename and len(by_basename[sid.lower()]) == 1:
            replacement = by_basename[sid.lower()][0]
            repair_type = "basename_to_full_skill_id"
        if not replacement:
            name_key = normalize_id_text(sid)
            if name_key in by_skill_name and len(by_skill_name[name_key]) == 1:
                replacement = by_skill_name[name_key][0]
                repair_type = "skill_name_to_skill_id"

        if replacement:
            repaired.append(replacement)
            if replacement != raw_sid:
                repairs.append({"type": repair_type, "from": raw_sid, "to": replacement})
        else:
            repaired.append(sid)
    if not repairs:
        return generated
    out = dict(generated)
    out["selected_skill_ids"] = repaired
    out["selected_skills"] = [{"skill_id": sid} for sid in repaired]
    out["selected_id_repairs"] = repairs
    return out


def token_set(text: str) -> set[str]:
    return {
        token
        for token in normalize_id_text(text).split()
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "from", "into", "this", "that"}
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def workflow_quality_filter(
    generated: dict[str, Any],
    evaluation: dict[str, Any],
    panel: dict[str, Any],
    target: int,
) -> dict[str, Any]:
    hard_issues = list(evaluation.get("hard_issues") or [])
    warnings = list(evaluation.get("quality_warnings") or [])
    selected_ids = [str(sid) for sid in (generated.get("selected_skill_ids") or [])]
    by_id = {
        str(skill.get("skill_id")): skill
        for skill in (panel.get("skills") or [])
        if skill.get("skill_id")
    }
    selected_skills = [by_id.get(sid, {}) for sid in selected_ids]

    reject_reasons: list[str] = []
    weak_reasons: list[str] = []
    ignored_warnings = sorted({warning for warning in warnings if warning in STAT_ONLY_WARNINGS})
    workflow_type = "single_skill" if target < 2 else "unknown"
    if hard_issues:
        reject_reasons.append("deterministic_hard_issue")
    if target >= 2 and any(not skill for skill in selected_skills):
        reject_reasons.append("selected_skill_metadata_missing")

    if target >= 2 and not reject_reasons:
        roles = [str(skill.get("operation_role") or "") for skill in selected_skills if skill]
        types = [str(skill.get("operation_type") or "") for skill in selected_skills if skill]
        object_sets = [
            {str(item) for item in (skill.get("task_object_types") or []) if item}
            for skill in selected_skills
            if skill
        ]
        shared_objects = set.intersection(*object_sets) if object_sets and all(object_sets) else set()
        union_objects = set.union(*object_sets) if object_sets else set()

        name_tokens = [
            token_set(f"{skill.get('skill_id')} {skill.get('skill_name')}")
            for skill in selected_skills
            if skill
        ]
        near_duplicate_pairs = [
            (left, right)
            for left, right in combinations(range(len(name_tokens)), 2)
            if jaccard(name_tokens[left], name_tokens[right]) >= 0.62
        ]
        if near_duplicate_pairs and len(set(types)) <= max(1, len(types) - 1):
            reject_reasons.append("near_duplicate_selected_skills")

        if roles and len(set(roles)) == 1:
            weak_reasons.append("low_role_diversity")
        if types and len(set(types)) < len(types):
            weak_reasons.append("duplicate_operation_type")
        if not shared_objects and len(union_objects) >= target * 2:
            weak_reasons.append("weak_shared_task_context")

        complementary_role_pairs = {
            ("load", "extract"), ("load", "analyze"), ("load", "generate"),
            ("extract", "analyze"), ("extract", "transform"), ("extract", "generate"),
            ("discover", "analyze"), ("discover", "validate"),
            ("configure", "implement"), ("configure", "operate"), ("configure", "validate"),
            ("implement", "validate"), ("implement", "operate"), ("implement", "generate"),
            ("generate", "validate"), ("generate", "transform"), ("generate", "publish"),
            ("transform", "validate"), ("transform", "generate"),
            ("analyze", "generate"), ("analyze", "report"), ("validate", "report"),
        }
        role_set = {role for role in roles if role}
        has_complementary_roles = any(
            (left, right) in complementary_role_pairs or (right, left) in complementary_role_pairs
            for left, right in combinations(role_set, 2)
        )
        if has_complementary_roles:
            workflow_type = "sequential_workflow"
        elif shared_objects:
            workflow_type = "capability_bundle"
        elif "weak_shared_task_context" in weak_reasons:
            workflow_type = "weak_overlap"
        else:
            workflow_type = "capability_bundle"

        severe_weak_combo = (
            "weak_shared_task_context" in weak_reasons
            and ("low_role_diversity" in weak_reasons or "duplicate_operation_type" in weak_reasons)
            and not has_complementary_roles
        )
        if severe_weak_combo:
            reject_reasons.append("weak_unrelated_or_overlapping_bundle")

    accepted = not reject_reasons
    if not accepted:
        workflow_type = "invalid"
    if accepted and warnings:
        weak_reasons.extend(f"warning:{warning}" for warning in warnings if warning not in STAT_ONLY_WARNINGS)
    return {
        "accepted": accepted,
        "status": "accepted" if accepted and not weak_reasons else ("weak" if accepted else "rejected"),
        "reject_reasons": sorted(set(reject_reasons)),
        "weak_reasons": sorted(set(weak_reasons)),
        "ignored_warnings": ignored_warnings,
        "workflow_type": workflow_type,
    }


def raw_response_path(output_dir: Path, target: int, index: int) -> Path:
    return output_dir / "raw_responses" / f"k{target}" / f"{index:04d}.json"


def call_single(
    row: dict[str, Any],
    row_number: int,
    lookup: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sid = str(row.get("skill_id") or "")
    skill = lookup.get(sid)
    if not skill:
        raise ValueError(f"missing skill in lookup: {sid}")
    messages = single.build_messages(skill, args.desc_chars_single, 1, 1)
    response = single.call_chat_completion(
        profile["base_url"],
        profile["api_key"],
        profile["model"],
        messages,
        args.temperature,
        args.top_p,
        args.timeout,
        use_schema=False,
        max_tokens=args.max_tokens,
    )
    generated = single.payload_from_task_description(skill, single.parse_content(response))
    evaluation = single.score_generated(generated, skill)
    return {
        "index": row_number,
        "source_index": row.get("index"),
        "source_file": args.input_1.as_posix(),
        "skill_id": sid,
        "skill_name": skill.get("skill_name") or skill.get("name") or row.get("skill_name"),
        "task_space_id": skill.get("task_space_id") or row.get("task_space_id"),
        "operation_role": skill.get("operation_role") or row.get("operation_role"),
        "operation_type": skill.get("operation_type") or row.get("operation_type"),
        "target_skill_count": 1,
        "old_generated": row.get("generated"),
        "generated": generated,
        "usage": response.get("usage"),
        "model": response.get("model", profile["model"]),
        "chosen_candidate_index": 1,
        "chosen_evaluation": evaluation,
        "parse_status": "ok",
        "raw_response_path": raw_response_path(args.output_dir, 1, row_number).as_posix(),
    }, response


def call_multi(
    row: dict[str, Any],
    row_number: int,
    lookup: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    args: argparse.Namespace,
    target: int,
    input_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel = build_replay_panel(row, lookup)
    candidate_ids = [str(skill.get("skill_id")) for skill in panel.get("skills") or []]
    messages = multi.build_messages(
        panel,
        lookup,
        max_skills=len(candidate_ids),
        desc_chars=args.desc_chars_multi,
        target_skill_count=target,
        query_style="scenario",
        candidate_index=1,
        best_of_n=1,
        strict_candidate_filter=False,
    )
    response = multi.call_chat_completion(
        profile["base_url"],
        profile["api_key"],
        profile["model"],
        messages,
        args.temperature,
        args.top_p,
        args.timeout,
        use_schema=False,
        target_skill_count=target,
        max_tokens=args.max_tokens,
    )
    generated = multi.parse_content(response, target)
    generated = repair_selected_ids(generated, candidate_ids, panel.get("skills") or [])
    evaluation = multi.score_generated(
        generated,
        panel,
        lookup,
        max_skills=len(candidate_ids),
        target_skill_count=target,
        strict_candidate_filter=False,
    )
    workflow_quality = workflow_quality_filter(generated, evaluation, panel, target)
    return {
        "index": row_number,
        "source_index": row.get("index"),
        "source_file": input_path.as_posix(),
        "task_space_id": row.get("task_space_id"),
        "anchor_task_object": row.get("anchor_task_object"),
        "candidate_skill_ids": candidate_ids,
        "panel_skills": panel.get("skills") or [],
        "target_skill_count": target,
        "old_generated": row.get("generated"),
        "old_selected_skill_ids": (row.get("generated") or {}).get("selected_skill_ids"),
        "generated": generated,
        "workflow_quality_filter": workflow_quality,
        "accepted_for_training": workflow_quality["accepted"],
        "usage": response.get("usage"),
        "model": response.get("model", profile["model"]),
        "query_style": "scenario",
        "chosen_candidate_index": 1,
        "chosen_evaluation": evaluation,
        "parse_status": "ok" if generated.get("query") and generated.get("selected_skill_ids") else "partial",
        "raw_response_path": raw_response_path(args.output_dir, target, row_number).as_posix(),
    }, response


def completed_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int]] = set()
    for row in load_jsonl(path):
        target = int(row.get("target_skill_count") or 0)
        index = int(row.get("index") or 0)
        keys.add((target, index))
    return keys


def run_one(
    target: int,
    row_number: int,
    row: dict[str, Any],
    input_path: Path,
    lookup: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        if target == 1:
            result, response = call_single(row, row_number, lookup, profile, args)
        else:
            result, response = call_multi(row, row_number, lookup, profile, args, target, input_path)
        if target == 1:
            evaluation = result.get("chosen_evaluation") or {}
            workflow_quality = workflow_quality_filter(
                result.get("generated") or {},
                evaluation,
                {"skills": []},
                target,
            )
            result["workflow_quality_filter"] = workflow_quality
            result["accepted_for_training"] = workflow_quality["accepted"]
        return result, response
    except Exception as exc:
        result = {
            "index": row_number,
            "source_index": row.get("index"),
            "source_file": input_path.as_posix(),
            "target_skill_count": target,
            "error": f"{type(exc).__name__}: {exc}",
            "generated": {},
            "chosen_evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
            "parse_status": "error",
            "workflow_quality_filter": {
                "accepted": False,
                "status": "rejected",
                "reject_reasons": ["generation_error"],
                "weak_reasons": [],
            },
            "accepted_for_training": False,
        }
        return result, None


def parse_k_values(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    bad = [value for value in values if value not in {1, 2, 3, 4}]
    if bad:
        raise SystemExit(f"unsupported k values: {bad}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", type=Path, default=Path("outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl"))
    parser.add_argument("--profile", type=Path, default=Path("profiles/deepseek_v32_windhub.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-1", type=Path, default=DEFAULT_INPUTS[1])
    parser.add_argument("--input-2", type=Path, default=DEFAULT_INPUTS[2])
    parser.add_argument("--input-3", type=Path, default=DEFAULT_INPUTS[3])
    parser.add_argument("--input-4", type=Path, default=DEFAULT_INPUTS[4])
    parser.add_argument("--k-values", default="1,2,3,4")
    parser.add_argument("--limit-per-k", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--desc-chars-single", type=int, default=420)
    parser.add_argument("--desc-chars-multi", type=int, default=420)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    configure_profile_transport(profile)
    if args.temperature is None:
        args.temperature = float(profile.get("temperature", 0.4))
    if args.top_p is None:
        args.top_p = float(profile.get("top_p", 0.0) or 0.0)
    if args.max_tokens is None:
        args.max_tokens = int(profile.get("max_tokens", 1200))
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    inputs = {1: args.input_1, 2: args.input_2, 3: args.input_3, 4: args.input_4}
    k_values = parse_k_values(args.k_values)
    output_path = args.output_dir / "pilot_400_skillrouter_prompt_replay.jsonl"
    accepted_path = args.output_dir / "accepted.jsonl"
    rejected_path = args.output_dir / "rejected.jsonl"
    write_json(args.output_dir / "run_config.json", {
        "profile": public_profile(profile),
        "skills": args.skills.as_posix(),
        "inputs": {str(k): inputs[k].as_posix() for k in k_values},
        "k_values": k_values,
        "limit_per_k": args.limit_per_k,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "desc_chars_single": args.desc_chars_single,
        "desc_chars_multi": args.desc_chars_multi,
        "prompt_family": "skillrouter_aligned_text_output",
    })

    lookup = load_skill_lookup(args.skills)
    tasks: list[tuple[int, int, dict[str, Any], Path]] = []
    done = completed_keys(output_path) if args.resume else set()
    for target in k_values:
        rows = load_jsonl(inputs[target])
        if args.limit_per_k > 0:
            rows = rows[:args.limit_per_k]
        for row_number, row in enumerate(rows, 1):
            if (target, row_number) in done:
                continue
            tasks.append((target, row_number, row, inputs[target]))

    lock = threading.Lock()
    if not args.resume and output_path.exists():
        output_path.unlink()
    if not args.resume:
        for split_path in [accepted_path, rejected_path]:
            if split_path.exists():
                split_path.unlink()

    def finish(task: tuple[int, int, dict[str, Any], Path]) -> dict[str, Any]:
        target, row_number, row, input_path = task
        result, response = run_one(target, row_number, row, input_path, lookup, profile, args)
        raw_path = raw_response_path(args.output_dir, target, row_number)
        if response is not None:
            write_json(raw_path, response)
            result["raw_response_path"] = raw_path.as_posix()
        append_jsonl(output_path, result, lock)
        append_jsonl(accepted_path if result.get("accepted_for_training") else rejected_path, result, lock)
        print(json.dumps({
            "target": target,
            "index": row_number,
            "source_index": row.get("index"),
            "selected": (result.get("generated") or {}).get("selected_skill_ids"),
            "old_selected": result.get("old_selected_skill_ids"),
            "score": (result.get("chosen_evaluation") or {}).get("score"),
            "hard_issues": (result.get("chosen_evaluation") or {}).get("hard_issues"),
            "quality_warnings": (result.get("chosen_evaluation") or {}).get("quality_warnings"),
            "workflow_status": (result.get("workflow_quality_filter") or {}).get("status"),
            "workflow_reject_reasons": (result.get("workflow_quality_filter") or {}).get("reject_reasons"),
            "parse_status": result.get("parse_status"),
            "error": result.get("error"),
        }, ensure_ascii=False), flush=True)
        if args.sleep:
            time.sleep(args.sleep)
        return result

    if args.workers == 1:
        for task in tasks:
            finish(task)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(finish, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    print(json.dumps({
        "output": output_path.as_posix(),
        "accepted_output": accepted_path.as_posix(),
        "rejected_output": rejected_path.as_posix(),
        "scheduled": len(tasks),
        "existing_skipped": len(done),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
