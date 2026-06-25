#!/usr/bin/env python3
"""Generate single-skill query training examples with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from generate_multiskill_probe import (
    NAMED_CONTEXT_TERMS,
    QUERY_REQUIRED_CONTEXT_TERMS,
    content_tokens,
    normalize,
    selected_context_warnings,
    token_coverage,
)
from evaluate_multiskill_probe import is_suspicious_skill_name


DEFAULT_EXTRA_HEADERS: dict[str, str] = {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def skill_text(row: dict[str, Any]) -> str:
    return normalize(" ".join([
        skill_id(row),
        str(row.get("skill_name") or row.get("name") or ""),
        str(row.get("description") or row.get("task_description") or ""),
        str(row.get("operation_role") or ""),
        str(row.get("operation_type") or ""),
        " ".join(row.get("task_object_types") or []),
        " ".join(row.get("input_artifact_types") or []),
        " ".join(row.get("output_artifact_types") or []),
    ]))


def compact_skill(row: dict[str, Any], desc_chars: int) -> dict[str, Any]:
    return {
        "skill_id": skill_id(row),
        "name": row.get("skill_name") or row.get("name"),
        "description": str(row.get("description") or row.get("task_description") or "")[:desc_chars],
        "task_space_id": row.get("task_space_id"),
        "operation": f"{row.get('operation_role')}/{row.get('operation_type')}",
        "task_objects": row.get("task_object_types") or [],
        "input_artifacts": row.get("input_artifact_types") or [],
        "output_artifacts": row.get("output_artifact_types") or [],
    }


def eligible_skill(row: dict[str, Any]) -> bool:
    desc = str(row.get("description") or row.get("task_description") or "").strip()
    if len(desc) < 40:
        return False
    if row.get("operation_role") == "general_capability":
        return False
    if row.get("operation_type") == "generic_general_capability":
        return False
    if not skill_id(row):
        return False
    return True


def choose_skills(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if eligible_skill(row):
            groups[str(row.get("task_space_id") or "unknown")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    group_keys = list(groups)
    rng.shuffle(group_keys)

    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit and group_keys:
        next_keys: list[str] = []
        for key in group_keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                if groups[key]:
                    next_keys.append(key)
            if len(chosen) >= limit:
                break
        group_keys = next_keys
    return chosen


def build_messages(skill: dict[str, Any], desc_chars: int, candidate_index: int, best_of_n: int) -> list[dict[str, str]]:
    spec = compact_skill(skill, desc_chars)
    skill_name = str(spec.get("name") or spec.get("skill_id") or "")
    category = str(spec.get("task_space_id") or spec.get("operation") or "uncategorized")
    body_preview = "; ".join([
        f"Operation: {spec.get('operation')}",
        f"Task objects: {', '.join(spec.get('task_objects') or [])}",
        f"Inputs: {', '.join(spec.get('input_artifacts') or [])}",
        f"Outputs: {', '.join(spec.get('output_artifacts') or [])}",
    ])
    system = (
        "You are an experienced user of AI assistants. You write clear, "
        "realistic task requests that describe what you need to accomplish."
    )
    user = (
        "Given the following skill specification, write a realistic task description "
        "that someone would ask an AI assistant to help with. The task should naturally "
        "require the capabilities described in this skill.\n\n"
        f"Skill name: {skill_name}\n"
        f"Category: {category}\n"
        f"Description: {spec.get('description')}\n"
        f"Skill body: {body_preview}\n\n"
        "Requirements:\n"
        "(1) Describe a concrete scenario with specific inputs/outputs.\n"
        "(2) Include enough detail that the skill would be clearly useful.\n"
        f"(3) Do NOT mention the skill name \"{skill_name}\" anywhere in the task.\n\n"
        "Output ONLY the task description."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    timeout: int,
    use_schema: bool,
    max_tokens: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    if top_p > 0:
        body["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "SkillSearch-workflow-first/0.1",
    }
    headers.update(DEFAULT_EXTRA_HEADERS)

    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    retryable_http_codes = {403, 408, 409, 425, 429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in retryable_http_codes or attempt == 3:
                raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt == 3:
                raise
        time.sleep(min(12.0, 1.5 * (2 ** attempt)))
    assert last_exc is not None
    raise last_exc


def parse_content(response: dict[str, Any]) -> str:
    text = str(response["choices"][0]["message"]["content"] or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def payload_from_task_description(skill: dict[str, Any], query: str) -> dict[str, Any]:
    sid = skill_id(skill)
    return {
        "query": query.strip(),
        "target_skill_count": 1,
        "selected_skill_ids": [sid],
        "selected_skill": {"skill_id": sid},
    }


def score_generated(generated: dict[str, Any], skill: dict[str, Any]) -> dict[str, Any]:
    sid = skill_id(skill)
    selected = [str(item) for item in (generated.get("selected_skill_ids") or [])]
    selected_skill = generated.get("selected_skill") or {}
    query = str(generated.get("query") or "")
    qnorm = normalize(query)
    text = skill_text(skill)

    hard_issues: list[str] = []
    quality_warnings: list[str] = []

    if int(generated.get("target_skill_count") or 0) != 1:
        hard_issues.append("wrong_target_skill_count")
    if selected != [sid]:
        hard_issues.append("selected_id_mismatch")
    if selected_skill and str(selected_skill.get("skill_id") or "") != sid:
        hard_issues.append("selected_skill_id_mismatch")
    if word_count(query) < 10:
        quality_warnings.append("query_too_short")
    if word_count(query) > 120:
        quality_warnings.append("query_too_long")
    if any(term in qnorm for term in ["target skill", "skill id", "operation label", "this specification"]):
        hard_issues.append("query_mentions_generation_artifact")
    if sid.lower() in query.lower():
        hard_issues.append("query_mentions_skill_id")

    name = str(skill.get("skill_name") or skill.get("name") or "")
    name_norm = normalize(name)
    if name_norm and len(name_norm) >= 4 and name_norm in qnorm and is_suspicious_skill_name(name, sid):
        quality_warnings.append("query_mentions_suspicious_skill_name")

    query_terms = set(qnorm.split())
    if any(term not in text for term in NAMED_CONTEXT_TERMS & query_terms):
        quality_warnings.append("unsupported_named_context_in_query")
    if any(term in text and term not in qnorm for term in QUERY_REQUIRED_CONTEXT_TERMS):
        quality_warnings.append("selected_context_requires_query_context")
    quality_warnings.extend(selected_context_warnings(text, qnorm))

    claim_tokens = content_tokens(" ".join([
        str(selected_skill.get("role_for_query") or ""),
        str(selected_skill.get("unique_capability") or ""),
        str(selected_skill.get("evidence_from_skill_description") or ""),
    ]))
    evidence_tokens = content_tokens(text)
    if len(claim_tokens) >= 5 and token_coverage(claim_tokens, evidence_tokens) < 0.12:
        quality_warnings.append("unsupported_skill_claim_by_evidence")

    if hard_issues:
        score = -1000 - 50 * len(hard_issues) - 5 * len(quality_warnings)
    else:
        score = 100
        score += 8 if 15 <= word_count(query) <= 80 else 0
        score -= 12 * len(set(quality_warnings))

    return {
        "score": score,
        "hard_issues": sorted(set(hard_issues)),
        "quality_warnings": sorted(set(quality_warnings)),
        "selected_skill_ids": selected,
        "query_word_count": word_count(query),
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[bool, bool, int, int]:
    evaluation = candidate.get("evaluation", {})
    return (
        not bool(evaluation.get("hard_issues")),
        not bool(evaluation.get("quality_warnings")),
        int(evaluation.get("score", -9999)),
        -int(candidate.get("candidate_index") or 0),
    )


def generate_candidate(
    *,
    skill_index: int,
    candidate_index: int,
    skill: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    messages = build_messages(skill, args.desc_chars, candidate_index, args.best_of_n)
    try:
        try:
            response = call_chat_completion(
                args.api_base,
                api_key,
                args.model,
                messages,
                args.temperature,
                args.top_p,
                args.timeout,
                use_schema=not args.no_schema,
                max_tokens=args.max_tokens,
            )
        except urllib.error.HTTPError as exc:
            if not args.no_schema:
                response = call_chat_completion(
                    args.api_base,
                    api_key,
                    args.model,
                    messages,
                    args.temperature,
                    args.top_p,
                    args.timeout,
                    use_schema=False,
                    max_tokens=args.max_tokens,
                )
            else:
                raise exc
        generated = payload_from_task_description(skill, parse_content(response))
        return {
            "skill_index": skill_index,
            "candidate_index": candidate_index,
            "generated": generated,
            "evaluation": score_generated(generated, skill),
            "usage": response.get("usage"),
            "model": response.get("model", args.model),
        }
    except Exception as exc:
        return {
            "skill_index": skill_index,
            "candidate_index": candidate_index,
            "error": f"{type(exc).__name__}: {exc}",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        }


def _candidate_process_worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        queue.put(generate_candidate(**kwargs))
    except BaseException as exc:
        queue.put({
            "skill_index": kwargs.get("skill_index"),
            "candidate_index": kwargs.get("candidate_index"),
            "error": f"{type(exc).__name__}: {exc}",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        })


def generate_candidate_with_process_timeout(kwargs: dict[str, Any], timeout: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_candidate_process_worker, args=(queue, kwargs))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {
            "skill_index": kwargs.get("skill_index"),
            "candidate_index": kwargs.get("candidate_index"),
            "error": f"TimeoutError: hard API-call timeout after {timeout}s",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        }
    if not queue.empty():
        return queue.get()
    return {
        "skill_index": kwargs.get("skill_index"),
        "candidate_index": kwargs.get("candidate_index"),
        "error": f"ProcessError: worker exited with code {process.exitcode} without a result",
        "evaluation": {
            "score": -2000,
            "hard_issues": ["generation_error"],
            "quality_warnings": [],
        },
    }


def candidate_kwargs_for(
    skill_index: int,
    candidate_index: int,
    skill: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    return {
        "skill_index": skill_index,
        "candidate_index": candidate_index,
        "skill": skill,
        "args": args,
        "api_key": api_key,
    }


def run_candidate(candidate_kwargs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.hard_timeout > 0:
        return generate_candidate_with_process_timeout(candidate_kwargs, args.hard_timeout)
    return generate_candidate(**candidate_kwargs)


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    generated = candidate.get("generated") or {}
    return {
        "candidate_index": candidate.get("candidate_index"),
        "score": candidate.get("evaluation", {}).get("score"),
        "hard_issues": candidate.get("evaluation", {}).get("hard_issues"),
        "quality_warnings": candidate.get("evaluation", {}).get("quality_warnings"),
        "selected_skill_ids": generated.get("selected_skill_ids"),
        "query": generated.get("query"),
        "usage": candidate.get("usage"),
        "error": candidate.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("CHATANYWHERE_MODEL", "gpt-4o"))
    parser.add_argument("--api-base", default=os.environ.get("CHATANYWHERE_API_BASE", "https://api.chatanywhere.tech/v1/"))
    parser.add_argument("--api-key-env", default="CHATANYWHERE_API_KEY")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--desc-chars", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.0, help="set >0 to pass top_p to the chat completion API")
    parser.add_argument("--best-of-n", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--drop-non-ok", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--hard-timeout", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=900, help="maximum completion tokens per API call; set <=0 to omit")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")
    if args.best_of_n < 1:
        raise SystemExit("--best-of-n must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    skills = choose_skills(load_jsonl(args.skills), args.limit, args.seed)
    outputs: list[dict[str, Any]] = []
    for index, skill in enumerate(skills, 1):
        candidates: list[dict[str, Any]] = []
        if args.workers == 1:
            for candidate_index in range(1, args.best_of_n + 1):
                kwargs = candidate_kwargs_for(index, candidate_index, skill, args, api_key)
                candidates.append(run_candidate(kwargs, args))
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_candidate,
                        candidate_kwargs_for(index, candidate_index, skill, args, api_key),
                        args,
                    )
                    for candidate_index in range(1, args.best_of_n + 1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    candidates.append(future.result())
                    if args.sleep:
                        time.sleep(args.sleep)

        candidates.sort(key=candidate_sort_key, reverse=True)
        best = candidates[0]
        if best.get("error") and not args.continue_on_error:
            raise RuntimeError(best["error"])
        best_eval = best.get("evaluation", {})
        if args.drop_non_ok and (best_eval.get("hard_issues") or best_eval.get("quality_warnings")):
            print(json.dumps({
                "index": index,
                "skill_id": skill_id(skill),
                "skipped": True,
                "reason": "best_candidate_not_ok",
                "score": best_eval.get("score"),
                "hard_issues": best_eval.get("hard_issues"),
                "quality_warnings": best_eval.get("quality_warnings"),
            }, ensure_ascii=False))
            continue

        row = {
            "index": index,
            "skill_id": skill_id(skill),
            "skill_name": skill.get("skill_name") or skill.get("name"),
            "task_space_id": skill.get("task_space_id"),
            "operation_role": skill.get("operation_role"),
            "operation_type": skill.get("operation_type"),
            "target_skill_count": 1,
            "generated": best.get("generated"),
            "usage": best.get("usage"),
            "model": best.get("model", args.model),
            "best_of_n": args.best_of_n,
            "chosen_candidate_index": best.get("candidate_index"),
            "chosen_evaluation": best_eval,
            "candidate_summaries": [summarize_candidate(candidate) for candidate in candidates],
        }
        if best.get("error"):
            row["error"] = best["error"]
        if args.keep_candidates:
            row["candidates"] = candidates
        outputs.append(row)
        print(json.dumps({
            "index": index,
            "skill_id": skill_id(skill),
            "selected_skill_ids": (best.get("generated") or {}).get("selected_skill_ids"),
            "score": best_eval.get("score"),
            "hard_issues": best_eval.get("hard_issues"),
            "quality_warnings": best_eval.get("quality_warnings"),
            "chosen_candidate_index": best.get("candidate_index"),
        }, ensure_ascii=False))

    write_jsonl(args.output, outputs)


if __name__ == "__main__":
    main()
