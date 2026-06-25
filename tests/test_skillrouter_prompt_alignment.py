from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_multiskill_probe import build_messages as build_multiskill_messages
from generate_multiskill_probe import parse_text_selection_payload
from generate_single_skill_probe import build_messages as build_single_messages
from generate_single_skill_probe import payload_from_task_description
from replay_pilot_400_skillrouter_prompt import repair_selected_ids
from replay_pilot_400_skillrouter_prompt import workflow_quality_filter


class SkillRouterPromptAlignmentTest(unittest.TestCase):
    def test_single_skill_prompt_matches_skillrouter_text_only_shape(self) -> None:
        skill = {
            "skill_id": "dev/example",
            "skill_name": "example-skill",
            "task_space_id": "software_dev",
            "description": "Implement a concrete software change from a user request.",
            "operation_role": "implement",
            "operation_type": "code_change",
            "task_object_types": ["codebase"],
            "input_artifact_types": ["source_code"],
            "output_artifact_types": ["code_patch"],
        }

        messages = build_single_messages(skill, desc_chars=200, candidate_index=1, best_of_n=3)
        user = messages[1]["content"]

        self.assertEqual(
            messages[0]["content"],
            "You are an experienced user of AI assistants. "
            "You write clear, realistic task requests that describe what you need to accomplish.",
        )
        self.assertIn("Given the following skill specification, write a realistic task description", user)
        self.assertIn("Skill name:", user)
        self.assertIn("Category:", user)
        self.assertIn("Description:", user)
        self.assertIn("Skill body:", user)
        self.assertIn("(1) Describe a concrete scenario with specific inputs/outputs.", user)
        self.assertIn("(2) Include enough detail that the skill would be clearly useful.", user)
        self.assertIn('(3) Do NOT mention the skill name "example-skill" anywhere in the task.', user)
        self.assertIn("Output ONLY the task description.", user)
        self.assertNotIn("Skill id:", user)
        self.assertNotIn("Output ONLY valid JSON", user)
        self.assertNotIn("candidate attempt", user)
        self.assertNotIn("operation labels", user)

    def test_single_skill_text_output_is_wrapped_with_known_skill_id(self) -> None:
        skill = {"skill_id": "dev/example", "skill_name": "example-skill"}

        payload = payload_from_task_description(
            skill,
            "I have a small service repository and need the requested endpoint implemented with the finished code patch returned.",
        )

        self.assertTrue(payload["query"].startswith("I have a small service repository"))
        self.assertEqual(payload["target_skill_count"], 1)
        self.assertEqual(payload["selected_skill_ids"], ["dev/example"])

    def test_single_skill_prompt_uses_metadata_as_body_preview(self) -> None:
        skill = {
            "skill_id": "dev/example",
            "skill_name": "example-skill",
            "task_space_id": "software_dev",
            "description": "Implement a concrete software change from a user request.",
            "body": "BODY_PREFIX_SENTINEL " + ("x" * 300),
            "operation_role": "implement",
            "operation_type": "code_change",
            "task_object_types": ["codebase"],
            "input_artifact_types": ["source_code"],
            "output_artifact_types": ["code_patch"],
        }

        messages = build_single_messages(skill, desc_chars=64, candidate_index=1, best_of_n=1)
        user = messages[1]["content"]

        self.assertIn("Skill body: Operation: implement/code_change", user)
        self.assertIn("Task objects: codebase", user)
        self.assertIn("Inputs: source_code", user)
        self.assertIn("Outputs: code_patch", user)
        self.assertNotIn("BODY_PREFIX_SENTINEL", user)

    def test_multiskill_prompt_keeps_skillrouter_style_with_two_line_output_contract(self) -> None:
        panel = {
            "task_space_id": "software_dev",
            "anchor_task_object": "codebase",
            "metrics": {"operation_role_count": 3, "operation_type_count": 3, "dominant_task_object_share": 0.3},
            "skills": [
                {
                    "skill_id": "dev/spec",
                    "skill_name": "Spec Writer",
                    "operation_role": "analyze",
                    "operation_type": "spec",
                    "task_object_types": ["codebase"],
                    "input_artifact_types": ["requirements"],
                    "output_artifact_types": ["implementation_spec"],
                },
                {
                    "skill_id": "dev/impl",
                    "skill_name": "Implementer",
                    "operation_role": "implement",
                    "operation_type": "code_change",
                    "task_object_types": ["codebase"],
                    "input_artifact_types": ["implementation_spec"],
                    "output_artifact_types": ["code_patch"],
                },
            ],
        }
        lookup = {
            "dev/spec": {"description": "Turn product requirements into a concrete implementation spec."},
            "dev/impl": {"description": "Implement code changes from a concrete implementation spec."},
        }

        messages = build_multiskill_messages(
            panel,
            lookup,
            max_skills=4,
            desc_chars=200,
            target_skill_count=2,
            query_style="scenario",
            candidate_index=1,
            best_of_n=2,
            strict_candidate_filter=False,
        )
        user = messages[1]["content"]

        self.assertIn("Given the following candidate skill specifications", user)
        self.assertIn("write a realistic task description that someone would ask an AI assistant to help with", user)
        self.assertIn("(1) Describe a concrete scenario with specific inputs/outputs.", user)
        self.assertIn("(2) Include enough detail that every selected skill would be clearly useful.", user)
        self.assertIn("(3) Select exactly 2 skills, no fewer and no more.", user)
        self.assertIn("copy the exact full Skill id strings from the candidate list", user)
        self.assertIn("Do not use candidate numbers", user)
        self.assertIn("Output ONLY the following two lines:", user)
        self.assertIn("Selected skill ids:", user)
        self.assertIn("Task description:", user)
        self.assertNotIn("Output ONLY valid JSON", user)
        self.assertNotIn("latent workflow", user)
        self.assertNotIn("candidate attempt", user)

    def test_multiskill_prompt_uses_metadata_as_body_preview(self) -> None:
        panel = {
            "task_space_id": "software_dev",
            "anchor_task_object": "codebase",
            "skills": [
                {
                    "skill_id": "dev/spec",
                    "skill_name": "Spec Writer",
                    "operation_role": "analyze",
                    "operation_type": "spec",
                    "task_object_types": ["codebase"],
                    "input_artifact_types": ["requirements"],
                    "output_artifact_types": ["implementation_spec"],
                    "modality_types": ["text"],
                }
            ],
        }
        lookup = {
            "dev/spec": {
                "description": "Turn product requirements into a concrete implementation spec.",
                "body": "SPEC_BODY_SENTINEL " + ("y" * 300),
            }
        }

        messages = build_multiskill_messages(
            panel,
            lookup,
            max_skills=1,
            desc_chars=80,
            target_skill_count=1,
            query_style="scenario",
            candidate_index=1,
            best_of_n=1,
            strict_candidate_filter=False,
        )
        user = messages[1]["content"]

        self.assertIn("Skill body: Operation: analyze/spec", user)
        self.assertIn("Task objects: codebase", user)
        self.assertIn("Outputs: implementation_spec", user)
        self.assertIn("Modalities: text", user)
        self.assertNotIn("SPEC_BODY_SENTINEL", user)
        self.assertNotIn("Workflow metadata:", user)

    def test_multiskill_two_line_text_output_is_parsed_to_generated_payload(self) -> None:
        payload = parse_text_selection_payload(
            "Selected skill ids: dev/spec, dev/impl\n"
            "Task description: I have product requirements for a backend change and need a concrete implementation spec plus the finished code patch.",
            target_skill_count=2,
        )

        self.assertEqual(payload["selected_skill_ids"], ["dev/spec", "dev/impl"])
        self.assertEqual(payload["target_skill_count"], 2)
        self.assertTrue(payload["query"].startswith("I have product requirements"))
        self.assertEqual(payload["selected_skills"], [{"skill_id": "dev/spec"}, {"skill_id": "dev/impl"}])

    def test_selected_id_repair_handles_candidate_numbers_and_missing_namespace(self) -> None:
        payload = {
            "query": "I need separate investigations synchronized into one report.",
            "target_skill_count": 2,
            "selected_skill_ids": ["1", "synchronize"],
            "selected_skills": [{"skill_id": "1"}, {"skill_id": "synchronize"}],
        }
        repaired = repair_selected_ids(
            payload,
            ["other/parallel-investigation", "other/synchronize"],
            [
                {"skill_id": "other/parallel-investigation", "skill_name": "parallel-investigation"},
                {"skill_id": "other/synchronize", "skill_name": "synchronize"},
            ],
        )

        self.assertEqual(repaired["selected_skill_ids"], ["other/parallel-investigation", "other/synchronize"])
        self.assertEqual(len(repaired["selected_id_repairs"]), 2)

    def test_workflow_quality_filter_rejects_near_duplicate_overlap(self) -> None:
        generated = {
            "query": "I need two equivalent web automation helpers for the same browser testing task.",
            "target_skill_count": 2,
            "selected_skill_ids": ["testing/playwright", "testing/playwright-helper"],
            "selected_skills": [{"skill_id": "testing/playwright"}, {"skill_id": "testing/playwright-helper"}],
        }
        evaluation = {"hard_issues": [], "quality_warnings": []}
        panel = {
            "skills": [
                {
                    "skill_id": "testing/playwright",
                    "skill_name": "playwright",
                    "operation_role": "discover",
                    "operation_type": "web_search_retrieval",
                    "task_object_types": ["web_page", "test_suite"],
                },
                {
                    "skill_id": "testing/playwright-helper",
                    "skill_name": "playwright-helper",
                    "operation_role": "discover",
                    "operation_type": "web_search_retrieval",
                    "task_object_types": ["web_page", "test_suite"],
                },
            ]
        }

        verdict = workflow_quality_filter(generated, evaluation, panel, target=2)

        self.assertFalse(verdict["accepted"])
        self.assertIn("near_duplicate_selected_skills", verdict["reject_reasons"])

    def test_workflow_quality_filter_does_not_weaken_for_query_length_only(self) -> None:
        generated = {
            "query": "I need a detailed but valid task request for a single concrete skill.",
            "target_skill_count": 1,
            "selected_skill_ids": ["other/example"],
            "selected_skills": [{"skill_id": "other/example"}],
        }
        evaluation = {"hard_issues": [], "quality_warnings": ["query_too_long"]}

        verdict = workflow_quality_filter(generated, evaluation, {"skills": []}, target=1)

        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["status"], "accepted")
        self.assertNotIn("warning:query_too_long", verdict["weak_reasons"])

    def test_workflow_quality_filter_classifies_valid_bundle_without_unclear_role_path(self) -> None:
        generated = {
            "query": (
                "A documentation team needs to update project docs from the latest codebase, "
                "reorganize the docs into a clearer bilingual structure, and return refreshed "
                "documentation files."
            ),
            "target_skill_count": 2,
            "selected_skill_ids": ["other/update-docs", "other/reorganize-docs"],
            "selected_skills": [{"skill_id": "other/update-docs"}, {"skill_id": "other/reorganize-docs"}],
        }
        evaluation = {"hard_issues": [], "quality_warnings": []}
        panel = {
            "skills": [
                {
                    "skill_id": "other/update-docs",
                    "skill_name": "update-docs",
                    "operation_role": "update",
                    "operation_type": "documentation_update",
                    "task_object_types": ["documentation", "codebase"],
                    "input_artifact_types": ["source_code", "docs"],
                    "output_artifact_types": ["updated_docs"],
                },
                {
                    "skill_id": "other/reorganize-docs",
                    "skill_name": "reorganize-docs",
                    "operation_role": "organize",
                    "operation_type": "documentation_structure",
                    "task_object_types": ["documentation", "codebase"],
                    "input_artifact_types": ["docs"],
                    "output_artifact_types": ["organized_docs"],
                },
            ]
        }

        verdict = workflow_quality_filter(generated, evaluation, panel, target=2)

        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["status"], "accepted")
        self.assertEqual(verdict["workflow_type"], "capability_bundle")
        self.assertNotIn("unclear_role_path", verdict["weak_reasons"])


if __name__ == "__main__":
    unittest.main()
