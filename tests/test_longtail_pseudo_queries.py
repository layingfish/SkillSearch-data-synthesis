from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_single_skill_probe as single  # noqa: E402

from build_longtail_pseudo_queries import (  # noqa: E402
    CandidateQuery,
    apply_task_spec_override,
    build_catalog_generation_record,
    build_catalog_pseudo_row,
    build_generation_messages,
    build_isolated_generation_record,
    build_isolated_mode_messages,
    build_isolated_pseudo_row,
    build_pseudo_row,
    canonicalize_query,
    collect_train_frequencies,
    exact_leak_reasons,
    extract_official_single_targets,
    normalize_task_spec_layout,
    parse_generated_queries,
    parse_single_generated_query,
    parse_single_generated_query_with_mode,
    reselect_catalog_generation_record,
    response_choice_contents,
    sanitize_skill_references,
    select_diverse_queries,
    select_zero_multi_cooccurrence_skills,
    validate_clean_catalog_pseudo_rows,
    validate_clean_generation_profile,
    validate_isolated_mode_pseudo_rows,
    write_augmented_rows,
    write_clean_augmented_rows,
    write_isolated_augmented_rows,
)


class LongtailPseudoQueriesTest(unittest.TestCase):
    def test_chat_completion_retries_direct_timeout_errors(self) -> None:
        with (
            patch.object(single.urllib.request, "urlopen", side_effect=TimeoutError("read timed out")) as urlopen,
            patch.object(single.time, "sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "read timed out"):
                single.call_chat_completion(
                    "https://example.invalid/v1",
                    "test-key",
                    "test-model",
                    [{"role": "user", "content": "test"}],
                    temperature=0.7,
                    top_p=0.9,
                    timeout=1,
                    use_schema=False,
                    max_tokens=16,
                    n=8,
                )

        self.assertEqual(urlopen.call_count, 6)

    def test_chat_completion_retries_502_and_uses_retry_after(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self) -> bytes:
                return b'{"choices": []}'

        error = single.urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            502,
            "Bad Gateway",
            {"Retry-After": "3"},
            None,
        )
        with (
            patch.object(single.urllib.request, "urlopen", side_effect=[error, FakeResponse()]) as urlopen,
            patch.object(single.random, "uniform", return_value=0.0),
            patch.object(single.time, "sleep") as sleep,
        ):
            response = single.call_chat_completion(
                "https://example.invalid/v1",
                "test-key",
                "test-model",
                [{"role": "user", "content": "test"}],
                temperature=0.7,
                top_p=0.9,
                timeout=1,
                use_schema=False,
                max_tokens=16,
                n=8,
            )

        self.assertEqual(response, {"choices": []})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(3.0)

    def test_canonicalize_query_truncates_at_complete_sentence(self) -> None:
        text = "A" * 1250 + ". " + "B" * 400

        query = canonicalize_query(text, max_chars=1500)

        self.assertLessEqual(len(query), 1500)
        self.assertTrue(query.endswith("."))

    def test_meta_skill_outputs_receive_grounded_task_spec_overrides(self) -> None:
        sql, sql_override = apply_task_spec_override(
            "SELECT id FROM users WHERE DATE(created_at) = '2024-01-01';",
            {"skill_id": "other/optimizing-query-text", "name": "optimizing-query-text"},
        )
        galaxy, galaxy_override = apply_task_spec_override(
            "I have paired-end FASTQ reads and need a tool recommendation for quality control.",
            {"skill_id": "other/galaxy-query-generation", "name": "Galaxy Query Generation (Repo Copy)"},
        )

        self.assertEqual(sql_override, "snowflake_sql_input")
        self.assertIn("You need to:", sql)
        self.assertIn("```sql\nSELECT id FROM users", sql)
        self.assertIn("semantically identical", sql)
        self.assertEqual(galaxy_override, "galaxy_benchmark_intent")
        self.assertIn("You need to:", galaxy)
        self.assertIn("data/benchmark/v1_items.jsonl", galaxy)
        self.assertIn("exactly one stable tool identifier", galaxy)

    def test_task_spec_prompt_is_single_message_and_uses_canonical_skill_view(self) -> None:
        messages = build_generation_messages(
            {
                "name": "Internal Skill Name",
                "description": "D" * 300 + "DESCRIPTION_TAIL",
                "body": "B" * 2500 + "BODY_TAIL",
            },
            mode="task_spec",
            count=5,
            body_chars=6000,
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        prompt = messages[0]["content"]
        self.assertNotIn("DESCRIPTION_TAIL", prompt)
        self.assertNotIn("BODY_TAIL", prompt)
        self.assertIn("Generate exactly 5 self-contained", prompt)
        self.assertIn("solvable using this skill alone", prompt)
        self.assertIn("`You need to:` followed by 2-5 numbered requirements", prompt)
        self.assertIn("Aim for 800-1,500 characters", prompt)
        self.assertIn("never fabricate it", prompt)

    def test_zero_multi_selector_uses_only_training_cooccurrence(self) -> None:
        catalog = [
            {"skill_id": "demo/alpha"},
            {"skill_id": "demo/beta"},
            {"skill_id": "demo/gamma"},
            {"skill_id": "demo/delta"},
        ]
        train_rows = [
            {"selected_skill_ids": ["demo/alpha"]},
            {"selected_skill_ids": ["demo/beta", "demo/gamma", "demo/beta"]},
        ]

        selected = select_zero_multi_cooccurrence_skills(catalog, train_rows)

        self.assertEqual([row["skill_id"] for row in selected], ["demo/alpha", "demo/delta"])

    def test_isolated_mode_prompts_are_distinct_single_messages(self) -> None:
        skill = {
            "skill_id": "demo/tabular-cleaning",
            "skill_name": "Tabular Cleaning",
            "description": "Normalize inconsistent values in a CSV.",
            "body": "Read a CSV, normalize supported values, validate rows, and save the result.",
        }
        prompts = {
            mode: build_isolated_mode_messages(skill, mode, body_chars=2500)
            for mode in ("outcome_first", "workflow_embedded", "constraint_failure")
        }

        self.assertTrue(all(len(messages) == 1 for messages in prompts.values()))
        self.assertTrue(all(messages[0]["role"] == "user" for messages in prompts.values()))
        self.assertEqual(len({messages[0]["content"] for messages in prompts.values()}), 3)
        for mode, messages in prompts.items():
            prompt = messages[0]["content"]
            self.assertIn(f"Diversity mode: {mode}", prompt)
            self.assertIn("exactly one specialized retrieval target", prompt)
            self.assertIn("`You need to:` followed by 2-5 numbered requirements", prompt)
            self.assertNotIn("official", prompt.lower())

    def test_isolated_task_layout_normalization_is_semantics_preserving(self) -> None:
        query = "The input is ready. You need: 1. Read it. 2. Validate it. Deliverable: report.json."

        normalized = normalize_task_spec_layout(query)

        self.assertIn("You need to:\n1. Read it.\n2. Validate it.", normalized)
        self.assertIn("Deliverable: report.json.", normalized)

        missing_heading = "The input is ready and must be checked.\n\n1. Read it.\n2. Validate it."
        self.assertIn("checked.\n\nYou need to:\n1. Read it.", normalize_task_spec_layout(missing_heading))

        localized_heading = "Os dados estão prontos. Você precisa: 1. Ler os dados. 2. Validar os dados."
        self.assertIn("You need to:\n1. Ler os dados.\n2. Validar os dados.", normalize_task_spec_layout(localized_heading))

    def test_extract_official_single_targets_deduplicates_and_keeps_row_indices(self) -> None:
        official_rows = [
            {"index": "a", "selected_skill_ids": ["gt/alpha"], "query": "first"},
            {"index": "b", "selected_skill_ids": ["gt/beta", "gt/gamma"], "query": "multi"},
            {"index": "c", "selected_skill_ids": ["gt/alpha"], "query": "second"},
            {"index": "d", "selected_skill_ids": ["gt/delta"], "query": "third"},
        ]
        train_rows = [
            {"selected_skill_ids": ["gt/alpha", "gt/other"]},
            {"selected_skill_ids": ["gt/delta"]},
            {"selected_skill_ids": ["gt/delta"]},
        ]

        targets = extract_official_single_targets(
            official_rows,
            collect_train_frequencies(train_rows),
            max_train_frequency=1,
        )

        self.assertEqual([target.skill_id for target in targets], ["gt/alpha"])
        self.assertEqual(targets[0].official_indices, ["a", "c"])
        self.assertEqual(targets[0].train_frequency, 1)
        self.assertEqual(targets[0].official_queries, ["first", "second"])

    def test_parse_generated_queries_accepts_json_arrays_and_numbered_lists(self) -> None:
        json_text = json.dumps(
            [
                {"query": "How can I normalize vendor names in this spreadsheet?"},
                "Please reconcile the customer records across two CSV exports.",
            ]
        )
        numbered_text = "1. Find missing values in the budget workbook.\n2) Produce a clean JSON report."

        self.assertEqual(
            parse_generated_queries(json_text),
            [
                "How can I normalize vendor names in this spreadsheet?",
                "Please reconcile the customer records across two CSV exports.",
            ],
        )
        self.assertEqual(
            parse_generated_queries(numbered_text),
            [
                "Find missing values in the budget workbook.",
                "Produce a clean JSON report.",
            ],
        )

    def test_parse_generated_queries_preserves_task_spec_layout(self) -> None:
        query = "You need to process `input.csv`.\n\n1. Validate the rows.\n2. Save `report.json`."

        self.assertEqual(
            parse_generated_queries(json.dumps([{"query": query}])),
            [query],
        )

    def test_multi_choice_response_yields_one_task_per_completion(self) -> None:
        response = {
            "choices": [
                {"index": 3, "message": {"content": json.dumps({"query": "First task\n\nYou need to:\n1. Do A."})}},
                {"index": 7, "message": {"content": "Second task\n\nYou need to:\n1. Do B."}},
            ]
        }

        self.assertEqual([index for index, _ in response_choice_contents(response)], [3, 7])
        self.assertEqual(
            parse_single_generated_query(response["choices"][0]["message"]["content"]),
            "First task\n\nYou need to:\n1. Do A.",
        )
        self.assertEqual(
            parse_single_generated_query(response["choices"][1]["message"]["content"]),
            "Second task\n\nYou need to:\n1. Do B.",
        )

    def test_single_query_parser_recovers_first_query_from_truncated_json(self) -> None:
        complete_first = (
            '```json\n[{"query":"First task\\n\\nYou need to:\\n1. Do A."},'
            '{"query":"Second task that is cut off'
        )
        truncated_first = (
            '[{"query":"A dataset is ready.\\n\\nYou need to:\\n1. Validate every row.\\n'
            'Deliverable: Save a report that is cut off'
        )

        query, mode = parse_single_generated_query_with_mode(complete_first)
        self.assertEqual(query, "First task\n\nYou need to:\n1. Do A.")
        self.assertEqual(mode, "json_prefix_complete")

        query, mode = parse_single_generated_query_with_mode(truncated_first)
        self.assertIn("You need to:\n1. Validate every row.", query)
        self.assertNotIn('"query"', query)
        self.assertEqual(mode, "json_prefix_truncated")

    def test_exact_leak_reasons_flags_skill_id_name_and_official_overlap(self) -> None:
        skill = {"skill_id": "gt/academic-pdf-redaction", "skill_name": "Academic PDF Redaction"}

        self.assertIn(
            "skill_id",
            exact_leak_reasons(
                "Use gt/academic-pdf-redaction for these files.",
                skill,
                official_queries=[],
            ),
        )
        self.assertIn(
            "skill_name",
            exact_leak_reasons(
                "Use the Academic PDF Redaction skill on three papers.",
                skill,
                official_queries=[],
            ),
        )
        self.assertIn(
            "official_ngram_overlap",
            exact_leak_reasons(
                "Please redact all information that could reveal authorship before submission.",
                skill,
                official_queries=["Please redact all information that could reveal authorship from the papers."],
                official_ngram_size=6,
            ),
        )

        self.assertEqual(
            exact_leak_reasons(
                "Process the expense claims and reconcile each receipt.",
                {"skill_id": "data/claims", "skill_name": "claims"},
                official_queries=[],
            ),
            [],
        )
        self.assertEqual(
            exact_leak_reasons(
                "Analyze the character arc across all manuscript chapters.",
                {"skill_id": "data/character-arc", "skill_name": "character-arc"},
                official_queries=[],
            ),
            [],
        )

    def test_select_diverse_queries_uses_relevance_and_pairwise_dissimilarity(self) -> None:
        skill_text = (
            "Normalize clinical lab measurements from multiple hospitals, harmonize units, "
            "convert values, and output a clean CSV with validation warnings."
        )
        candidates = [
            CandidateQuery("a", "generic", "Normalize lab values from two hospitals and convert units into one CSV.", 0, {}),
            CandidateQuery("b", "generic", "Normalize lab values from hospitals and convert units into a CSV file.", 1, {}),
            CandidateQuery("c", "generic", "Check unit conversions and produce validation warnings for abnormal ranges.", 2, {}),
            CandidateQuery("d", "generic", "Summarize a movie plot in three bullet points.", 3, {}),
        ]

        selected = select_diverse_queries(candidates, skill_text, keep=2)

        self.assertEqual([item.query_id for item in selected], ["b", "c"])

    def test_task_selection_prefers_complete_format_and_untruncated_candidates(self) -> None:
        skill_text = "Validate a CSV and save a report."
        candidates = [
            CandidateQuery(
                f"good-{index}",
                "task_spec",
                f"Input file_{index}.csv is ready.\n\nYou need to:\n1. Validate rows.\n2. Save report_{index}.json.",
                index,
                {"was_truncated": False},
            )
            for index in range(5)
        ]
        candidates.extend(
            [
                CandidateQuery(
                    "missing-marker",
                    "task_spec",
                    "Validate every CSV row and save a detailed JSON report.",
                    5,
                    {"was_truncated": False},
                ),
                CandidateQuery(
                    "truncated",
                    "task_spec",
                    "Input.csv is ready.\n\nYou need to:\n1. Validate rows and save a JSON report.",
                    6,
                    {"was_truncated": True},
                ),
            ]
        )

        selected = select_diverse_queries(candidates, skill_text, keep=5)

        self.assertEqual({item.query_id for item in selected}, {f"good-{index}" for index in range(5)})

    def test_build_pseudo_row_and_augment_rows_are_training_compatible(self) -> None:
        row = build_pseudo_row(
            query="Recover missing values in a workbook and save a completed file.",
            skill_id="gt/data-reconciliation",
            mode="generic",
            rank=0,
            source_candidate_count=12,
            official_indices=["row-1"],
            train_frequency=1,
        )

        self.assertEqual(row["selected_skill_ids"], ["gt/data-reconciliation"])
        self.assertEqual(row["target_skill_count"], 1)
        self.assertEqual(row["positive_skill_id"], "gt/data-reconciliation")
        self.assertIn("Query:Recover missing values", row["query_text"])
        self.assertTrue(row["pseudo_query"]["transductive_diagnostic"])

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.jsonl"
            pseudo = Path(tmpdir) / "pseudo.jsonl"
            output = Path(tmpdir) / "train_plus_pq.jsonl"
            manifest = Path(tmpdir) / "manifest.json"
            base.write_text(json.dumps({"index": 0, "query": "base", "selected_skill_ids": ["gt/base"]}) + "\n")
            pseudo.write_text(json.dumps(row) + "\n")

            write_augmented_rows([base], [pseudo], output, manifest)

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            info = json.loads(manifest.read_text())

        self.assertEqual(len(rows), 2)
        self.assertEqual(info["base_rows"], 1)
        self.assertEqual(info["pseudo_rows"], 1)

    def test_catalog_generation_is_fixed_budget_clean_and_selects_five(self) -> None:
        skill = {
            "skill_id": "demo/tabular-cleaning",
            "skill_name": "Tabular Cleaning",
            "description": "Clean inconsistent values in tabular files and validate the result.",
            "body": "Read a CSV, normalize values, validate rows, and save a cleaned CSV plus a report.",
        }
        choices = []
        for index in range(8):
            query = (
                f"A data team has received input_{index}.csv with inconsistent values in column_{index}.\n\n"
                "You need to:\n"
                "1. Read the supplied CSV and identify inconsistent values.\n"
                "2. Normalize those values while preserving the remaining fields.\n"
                f"3. Save cleaned_{index}.csv and a validation report confirming every row was processed."
            )
            choices.append({"index": index, "message": {"content": json.dumps([{"query": query}])}})
        response = {
            "model": "gpt-4o-2024-11-20",
            "usage": {"prompt_tokens": 100, "completion_tokens": 800, "total_tokens": 900},
            "choices": choices,
        }

        record = build_catalog_generation_record(skill, response, requested_candidates=8, keep=5, max_query_chars=1500)

        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["returned_choices"], 8)
        self.assertEqual(record["selected_count"], 5)
        self.assertEqual(len(record["choices"]), 8)
        self.assertEqual(len(record["selected_queries"]), 5)
        self.assertNotIn("official", json.dumps(record).lower())
        self.assertNotIn("transductive", json.dumps(record).lower())

        stale = json.loads(json.dumps(record))
        stale["status"] = "insufficient_candidates"
        stale["selected_queries"] = []
        stale["selected_count"] = 0
        for choice in stale["choices"]:
            choice["reject_reasons"] = ["stale_rule"]
            choice["selected"] = False
        reselected = reselect_catalog_generation_record(skill, stale, keep=5)
        self.assertEqual(reselected["status"], "complete")
        self.assertEqual(reselected["selected_count"], 5)

    def test_clean_catalog_rows_preserve_layout_and_require_full_coverage(self) -> None:
        skills = [
            {"skill_id": "demo/alpha", "skill_name": "Internal Alpha"},
            {"skill_id": "demo/beta", "skill_name": "Internal Beta"},
        ]
        rows = []
        for skill in skills:
            sid = skill["skill_id"]
            for rank in range(2):
                rows.append(
                    build_catalog_pseudo_row(
                        query=f"Process input_{rank}.csv.\n\nYou need to:\n1. Validate row group {rank}.",
                        sid=sid,
                        rank=rank,
                        source_candidate_count=8,
                        score=1.0,
                        candidate_index=rank,
                        generator_model="gpt-4o-2024-11-20",
                        source_was_truncated=False,
                    )
                )

        validate_clean_catalog_pseudo_rows(rows, skills, queries_per_skill=2, max_query_chars=1500)
        with self.assertRaisesRegex(ValueError, "skill coverage mismatch"):
            validate_clean_catalog_pseudo_rows(rows[:2], skills, queries_per_skill=2, max_query_chars=1500)
        self.assertIn("\n\nYou need to:\n", rows[0]["query"])

        bare_sql = build_catalog_pseudo_row(
            query="SELECT id FROM users;",
            sid="other/optimizing-query-text",
            rank=0,
            source_candidate_count=8,
            score=1.0,
            candidate_index=0,
            generator_model="gpt-4o-2024-11-20",
            source_was_truncated=False,
        )
        with self.assertRaisesRegex(ValueError, "bare SQL"):
            validate_clean_catalog_pseudo_rows(
                [bare_sql],
                [{"skill_id": "other/optimizing-query-text", "name": "optimizing-query-text"}],
                queries_per_skill=1,
                max_query_chars=1500,
            )

    def test_openrouter_is_rejected_for_clean_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "OpenRouter is disabled"):
            validate_clean_generation_profile(
                {"name": "legacy", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"}
            )
        validate_clean_generation_profile(
            {"name": "gpt4o_huayanapi", "base_url": "https://www.huayanapi.com/v1", "model": "gpt-4o"}
        )

    def test_internal_command_paths_are_sanitized_without_losing_arguments(self) -> None:
        skill = {"skill_id": "data/ansible-fix", "skill_name": "ansible-fix"}
        query = (
            "Run `.claude/skills/ansible-fix/ansible-lint-fix.py roles/webserver` and then validate the role."
        )

        sanitized, edit_types = sanitize_skill_references(query, skill)

        self.assertNotIn("ansible-fix", sanitized)
        self.assertIn("`<PROJECT_COMMAND> roles/webserver`", sanitized)
        self.assertIn("skill_name_token", edit_types)
        self.assertEqual(exact_leak_reasons(sanitized, skill, official_queries=[]), [])

    def test_catalog_generation_sanitizes_all_identifier_bearing_choices(self) -> None:
        skill = {
            "skill_id": "data/add-home-widget-selector",
            "skill_name": "add-home-widget-selector",
            "description": "Add a configurable home widget selector.",
            "body": "Run the project command with a plugin path and widget arguments, then verify registration.",
        }
        choices = []
        for index in range(8):
            query = (
                f"The plugin at lib/plugins/demo_{index} is ready.\n\nYou need to:\n"
                f"1. Run `/add-home-widget-selector lib/plugins/demo_{index} --widget-id widget_{index}`.\n"
                "2. Verify that the selector is registered and the project validation succeeds."
            )
            choices.append({"index": index, "message": {"content": json.dumps([{"query": query}])}})
        response = {"model": "gpt-4o-2024-11-20", "choices": choices}

        record = build_catalog_generation_record(skill, response, requested_candidates=8, keep=5, max_query_chars=1500)

        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["accepted_candidates"], 8)
        self.assertTrue(all("add-home-widget-selector" not in row["query"] for row in record["choices"]))
        self.assertTrue(all(row["metadata"]["sanitized_reference_types"] for row in record["choices"]))

    def test_clean_augment_writes_training_rows_after_validation(self) -> None:
        skill = {"skill_id": "demo/alpha", "skill_name": "Internal Alpha"}
        pseudo = build_catalog_pseudo_row(
            query="Process input.csv.\n\nYou need to:\n1. Validate every row.",
            sid="demo/alpha",
            rank=0,
            source_candidate_count=8,
            score=1.0,
            candidate_index=2,
            generator_model="gpt-4o-2024-11-20",
            source_was_truncated=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.jsonl"
            pseudo_path = root / "pseudo.jsonl"
            skills_path = root / "skills.jsonl"
            output_path = root / "augmented.jsonl"
            manifest_path = root / "manifest.json"
            base_path.write_text(json.dumps({"index": "base:0", "query": "base"}) + "\n")
            pseudo_path.write_text(json.dumps(pseudo) + "\n")
            skills_path.write_text(json.dumps(skill) + "\n")

            write_clean_augmented_rows(
                [base_path],
                [pseudo_path],
                skills_path,
                output_path,
                manifest_path,
                queries_per_skill=1,
                max_query_chars=1500,
            )

            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(output_rows), 2)
        self.assertTrue(manifest["clean_catalog"])
        self.assertEqual(manifest["pseudo_rows"], 1)

    def test_isolated_mode_rows_validate_and_append_without_benchmark_metadata(self) -> None:
        skill = {
            "skill_id": "demo/alpha",
            "skill_name": "Internal Alpha",
            "description": "Validate records and save a structured report.",
            "body": "Read the supplied records, validate each item, and save a report.",
        }
        rows = []
        for index, mode in enumerate(("outcome_first", "workflow_embedded", "constraint_failure")):
            query = (
                f"A team has supplied records_{index}.csv and needs a verified result for its next step.\n\n"
                "You need to:\n"
                "1. Read every supplied record.\n"
                "2. Validate each item using the documented checks.\n"
                f"3. Save report_{index}.json and confirm that every input item is represented.\n\n"
                "Deliverable: A structured validation report.\n"
                "Acceptance criteria: Every supplied record has a validation result."
            )
            response = {
                "model": "gpt-4o-2024-11-20",
                "choices": [{"index": 0, "message": {"content": json.dumps({"query": query})}}],
            }
            record = build_isolated_generation_record(skill, mode, response, max_query_chars=1500)
            self.assertEqual(record["status"], "complete")
            rows.append(build_isolated_pseudo_row(record, "gpt-4o-2024-11-20"))

        modes = ["outcome_first", "workflow_embedded", "constraint_failure"]
        validate_isolated_mode_pseudo_rows(rows, [skill], modes, max_query_chars=1500)
        serialized = json.dumps(rows).lower()
        self.assertNotIn("official", serialized)
        self.assertNotIn("transductive", serialized)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.jsonl"
            pseudo_path = root / "pseudo.jsonl"
            skills_path = root / "skills.jsonl"
            output_path = root / "augmented.jsonl"
            manifest_path = root / "manifest.json"
            base_path.write_text(
                json.dumps(
                    {
                        "index": "base:0",
                        "query": "Check the source records.",
                        "selected_skill_ids": ["demo/alpha"],
                    }
                )
                + "\n"
            )
            pseudo_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            skills_path.write_text(json.dumps(skill) + "\n")

            write_isolated_augmented_rows(
                [base_path],
                [pseudo_path],
                skills_path,
                output_path,
                manifest_path,
                modes,
                max_query_chars=1500,
            )

            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(output_rows), 4)
        self.assertEqual(manifest["target_skills"], 1)
        self.assertEqual(manifest["pseudo_rows"], 3)


if __name__ == "__main__":
    unittest.main()
