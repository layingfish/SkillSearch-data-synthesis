from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from skill_gr_pipeline import (  # noqa: E402
    DEFAULT_GR_MAX_SKILLS,
    DEFAULT_GR_MAX_SOURCE_LENGTH,
    DEFAULT_GR_NUM_BEAMS,
    DEFAULT_PAG_QUERY_PREFIX_MAX_LENGTH,
    METHOD_SETTING_PAG,
    METHOD_SETTING_RIPOR,
    METHOD_SETTING_RIPOR_PAG,
    SKILLROUTER_QUERY_MAX_CHARS,
    SEMANTIC_ID_MODE_DEDUP,
    SEMANTIC_ID_MODE_LEARNED,
    advance_structured_beam,
    append_loss_history,
    audit_source_token_lengths,
    allowed_next_tokens_for_skill_trie,
    build_decoder_target,
    build_arg_parser,
    build_ripor_candidate_batch,
    build_teacher_rank_examples,
    build_single_sid_catalog_rows,
    expand_single_sid_query_rows,
    build_prefix_trie,
    collect_query_skill_interactions,
    canonicalize_exact_skill_documents,
    compute_at_k_metrics,
    compute_retrieval_metrics,
    command_rescore_ranked,
    command_write_pag_msp_protocol,
    deduplicate_code_tuples,
    decoder_source_text,
    encode_structured_sid_target,
    format_skillrouter_query,
    format_skillrouter_skill,
    global_sid_token_id,
    infer_structured_sid_config,
    load_semantic_ids_with_mode,
    load_structured_decoder_checkpoint,
    make_structured_t5_sid_model,
    merge_pag_set_sequence_states,
    load_ranked_skill_candidates_with_scores,
    load_pag_set_token_ids,
    native_semantic_id_text,
    pag_quadratic_regularization_weight,
    pag_msp_protocol_provenance,
    ripor_prefix_alphas,
    pag_prefix_planner_score,
    prepared_selected_skill_ids,
    ranked_skill_ids_from_structured_beams,
    ranked_skill_ids_from_marginal_structured_beams,
    remap_row_skill_aliases,
    resolve_eval_max_source_length,
    save_structured_decoder_checkpoint,
    semantic_ids_sha256,
    setting_contract_provenance,
    split_and_expand_single_sid_query_rows,
    stratified_split_rows,
    structured_allowed_values,
    summarize_source_token_lengths,
    truncate_pag_query_prefix,
    validate_checkpoint_semantic_ids,
    validate_letter_tokenizer_manifest,
    validate_method_setting_contract,
    validate_rank_capacity,
)
from distributed_utils import DistributedContext, distributed_epoch_batches_no_padding  # noqa: E402
from train_letter_rqvae_official import (  # noqa: E402
    compute_last_code_costs,
    minimum_cost_unique_last_codes,
    vectorized_letter_diversity_loss_main_entry,
)


class SkillGRPipelineTest(unittest.TestCase):
    def test_no_padding_distributed_epoch_visits_each_row_once(self) -> None:
        rows = list(range(19))
        batches_by_rank = []
        for rank in range(8):
            context = DistributedContext(rank=rank, local_rank=rank, world_size=8, device=None)
            batches_by_rank.append(
                list(
                    distributed_epoch_batches_no_padding(
                        rows,
                        2,
                        context,
                        seed=17,
                        epoch=3,
                    )
                )
            )
        self.assertEqual({len(batches) for batches in batches_by_rank}, {2})
        visited = [value for batches in batches_by_rank for batch in batches for value in batch]
        self.assertEqual(len(visited), len(rows))
        self.assertEqual(sorted(visited), rows)

    def test_append_loss_history_writes_jsonl_rows(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "loss_history.jsonl"

            append_loss_history(path, {"stage": "query", "step": 1, "train_loss": 3.0})
            append_loss_history(path, {"stage": "query", "step": 2, "dev_loss": 2.5})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(
            rows,
            [
                {"stage": "query", "step": 1, "train_loss": 3.0},
                {"stage": "query", "step": 2, "dev_loss": 2.5},
            ],
        )

    def test_decoder_source_text_accepts_prepared_query_rows_without_decoder_input(self) -> None:
        self.assertEqual(decoder_source_text({"decoder_input": "catalog doc", "query_text": "query text"}), "catalog doc")
        self.assertEqual(
            decoder_source_text({"query_text": "truncated query text", "query": "full raw query"}),
            format_skillrouter_query("full raw query"),
        )
        self.assertEqual(decoder_source_text({"query": "raw query"}), format_skillrouter_query("raw query"))
        self.assertEqual(decoder_source_text({"query_text": "legacy query text"}), "legacy query text")
        prefixed_long_query = format_skillrouter_query("x" * 2000, max_chars=None)
        self.assertEqual(
            decoder_source_text({"query": prefixed_long_query}),
            format_skillrouter_query("x" * 2000),
        )

    def test_collect_query_skill_interactions_supports_prepared_and_raw_rows(self) -> None:
        rows = [
            {"selected_skill_ids": ["skill/a", "skill/b", "skill/a"]},
            {"generated": {"selected_skill_ids": ["skill/c", "missing"]}},
        ]

        interactions, missing = collect_query_skill_interactions(
            rows,
            {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        )

        self.assertEqual(prepared_selected_skill_ids(rows[0]), ["skill/a", "skill/b", "skill/a"])
        self.assertEqual(interactions, [(0, 0), (0, 1), (1, 2)])
        self.assertEqual(missing["missing"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_structured_decoder_checkpoint_restores_optimizer_state_with_requested_lr(self) -> None:
        import tempfile

        import torch

        class DummyTokenizer:
            def save_pretrained(self, path: str) -> None:
                Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()

            save_structured_decoder_checkpoint(
                checkpoint_dir,
                model,
                DummyTokenizer(),
                "tiny",
                codebook_size=4,
                depth=2,
                dedup_size=2,
                seed=7,
                training_formulation="unit-test",
                extra={"best_step": 11},
                optimizer=optimizer,
            )

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
            checkpoint = load_structured_decoder_checkpoint(
                restored_model,
                checkpoint_dir,
                torch.device("cpu"),
                optimizer=restored_optimizer,
                lr=1e-4,
            )

        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertEqual(checkpoint["best_step"], 11)
        self.assertTrue(restored_optimizer.state_dict()["state"])
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 1e-4)
        for saved, restored in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(saved, restored))

    def test_semantic_id_fingerprint_is_order_invariant_and_checkpoint_checked(self) -> None:
        semantic_ids = {"skill/b": [4, 5, 6, 0], "skill/a": [1, 2, 3, 0]}
        reordered = {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 6, 0]}
        fingerprint = semantic_ids_sha256(semantic_ids)

        self.assertEqual(fingerprint, semantic_ids_sha256(reordered))
        self.assertTrue(
            validate_checkpoint_semantic_ids(
                {"semantic_ids_sha256": fingerprint, "semantic_id_count": 2},
                reordered,
            )
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_checkpoint_semantic_ids(
                {"semantic_ids_sha256": fingerprint, "semantic_id_count": 2},
                {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 7, 0]},
            )

    def test_eval_source_length_uses_checkpoint_and_rejects_silent_override(self) -> None:
        checkpoint = {"max_source_length": 2048}

        self.assertEqual(resolve_eval_max_source_length(None, checkpoint, False), (2048, "checkpoint"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_eval_max_source_length(512, checkpoint, False)
        self.assertEqual(resolve_eval_max_source_length(512, checkpoint, True), (512, "cli"))
        with self.assertRaisesRegex(ValueError, "legacy decoder checkpoint"):
            resolve_eval_max_source_length(None, {}, False)

    def test_skillrouter_skill_format_uses_name_description_body_only(self) -> None:
        skill = {
            "skill_id": "demo/skill",
            "skill_name": "demo-skill",
            "description": "short description",
            "body": "full body text",
            "operation_role": "validate",
            "task_object_types": ["image"],
        }

        text = format_skillrouter_skill(skill)

        self.assertEqual(text, "demo-skill | short description | full body text")
        self.assertNotIn("validate", text)
        self.assertNotIn("image", text)

    def test_skillrouter_query_format_uses_instruction_prefix(self) -> None:
        text = format_skillrouter_query("Fix a broken release workflow.")

        self.assertEqual(
            text,
            "Instruct: Given a task description, retrieve the most relevant "
            "skill document that would help an agent complete the task\n"
            "Query:Fix a broken release workflow.",
        )

    def test_skillrouter_query_format_uses_shared_character_cap_by_default(self) -> None:
        query = "x" * 2000

        self.assertTrue(format_skillrouter_query(query).endswith("x" * SKILLROUTER_QUERY_MAX_CHARS))
        self.assertEqual(
            len(format_skillrouter_query(query)) - len(format_skillrouter_query("")),
            SKILLROUTER_QUERY_MAX_CHARS,
        )
        self.assertTrue(format_skillrouter_query(query, max_chars=None).endswith(query))

    def test_source_length_audit_fails_before_implicit_truncation(self) -> None:
        class CharacterTokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [list(range(len(text))) for text in texts]}

        rows = [{"decoder_input": "short"}, {"decoder_input": "x" * 12}]
        summary = summarize_source_token_lengths(rows, CharacterTokenizer(), max_source_length=10)

        self.assertEqual(summary["over_limit_rows"], 1)
        self.assertEqual(summary["max_tokens"], 12)
        with self.assertRaisesRegex(ValueError, "increase --max-source-length"):
            audit_source_token_lengths(
                "unit",
                rows,
                CharacterTokenizer(),
                max_source_length=10,
                allow_source_truncation=False,
            )

    def test_deduplicate_code_tuples_appends_hit_count_column(self) -> None:
        deduped = deduplicate_code_tuples(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
                "c": [1, 2, 4],
            }
        )

        self.assertEqual(deduped["a"], [1, 2, 3, 0])
        self.assertEqual(deduped["b"], [1, 2, 3, 1])
        self.assertEqual(deduped["c"], [1, 2, 4, 0])

    def test_learned_semantic_id_schema_is_loaded_and_inferred_without_dedup(self) -> None:
        rows = [
            {
                "skill_id": "skill/a",
                "semantic_id": [1, 2, 3, 4],
                "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
            },
            {
                "skill_id": "skill/b",
                "semantic_id": [4, 3, 2, 1],
                "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "semantic_ids.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            semantic_ids, mode = load_semantic_ids_with_mode(path)

        config = infer_structured_sid_config(
            semantic_ids,
            codebook_size=256,
            semantic_id_mode=mode,
        )
        self.assertEqual(mode, SEMANTIC_ID_MODE_LEARNED)
        self.assertEqual(config["depth"], 4)
        self.assertEqual(config["dedup_size"], 0)
        self.assertEqual(config["semantic_id_mode"], SEMANTIC_ID_MODE_LEARNED)

    def test_legacy_semantic_id_rows_without_schema_remain_backward_compatible(self) -> None:
        rows = [
            {"skill_id": "skill/a", "semantic_id": [1, 2, 3, 0]},
            {"skill_id": "skill/b", "semantic_id": [4, 5, 6, 0]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "semantic_ids.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            _, mode = load_semantic_ids_with_mode(path)

        self.assertEqual(mode, SEMANTIC_ID_MODE_DEDUP)

    def test_checkpoint_rejects_semantic_id_schema_mismatch(self) -> None:
        semantic_ids = {"skill/a": [1, 2, 3, 4], "skill/b": [4, 3, 2, 1]}
        checkpoint = {
            "semantic_id_mode": SEMANTIC_ID_MODE_DEDUP,
            "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, SEMANTIC_ID_MODE_LEARNED),
        }

        with self.assertRaisesRegex(ValueError, "mode does not match"):
            validate_checkpoint_semantic_ids(checkpoint, semantic_ids, SEMANTIC_ID_MODE_LEARNED)

    def test_global_sid_token_id_offsets_each_rq_level(self) -> None:
        self.assertEqual(global_sid_token_id(level=0, value=7, codebook_size=256, depth=3, dedup_size=4), 7)
        self.assertEqual(global_sid_token_id(level=1, value=7, codebook_size=256, depth=3, dedup_size=4), 263)
        self.assertEqual(global_sid_token_id(level=2, value=7, codebook_size=256, depth=3, dedup_size=4), 519)
        self.assertEqual(global_sid_token_id(level=3, value=2, codebook_size=256, depth=3, dedup_size=4), 770)
        self.assertEqual(global_sid_token_id(level=4, value=1, codebook_size=256, depth=3, dedup_size=4), 773)

    def test_structured_sid_target_uses_per_level_labels_and_control_steps(self) -> None:
        semantic_ids = {
            "skill/b": [2, 3, 4, 0],
            "skill/a": [1, 3, 5, 0],
        }

        encoded = encode_structured_sid_target(
            ["skill/b", "skill/a"],
            semantic_ids,
            codebook_size=256,
            depth=3,
            dedup_size=4,
        )

        self.assertEqual(encoded["types"], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(encoded["labels"], [1, 3, 5, 0, 0, 2, 3, 4, 0, 1])
        self.assertEqual(encoded["decoder_input_ids"][0], encoded["bos_id"])
        self.assertEqual(encoded["decoder_input_ids"][1:], encoded["target_token_ids"][:-1])
        self.assertEqual(encoded["skill_ids"], ["skill/a", "skill/b"])

    def test_learned_structured_sid_target_has_four_code_steps_and_no_dedup_step(self) -> None:
        semantic_ids = {"skill/a": [1, 2, 3, 4]}

        encoded = encode_structured_sid_target(
            ["skill/a"],
            semantic_ids,
            codebook_size=256,
            depth=4,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )

        self.assertEqual(encoded["types"], [0, 1, 2, 3, 4])
        self.assertEqual(encoded["labels"], [1, 2, 3, 4, 1])
        self.assertEqual(encoded["target_token_ids"], [1, 258, 515, 772, 1025])
        self.assertEqual(encoded["bos_id"], 1026)

    def test_learned_structured_beam_completes_after_four_codes(self) -> None:
        beam = {
            "decoder_input_ids": [1026, 1, 258, 515],
            "current_prefix": [1, 2, 3],
            "completed": [],
            "score": -1.0,
            "stopped": False,
        }

        code_beam = advance_structured_beam(
            beam,
            target_type=3,
            value=4,
            log_prob=-0.2,
            codebook_size=256,
            depth=4,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )
        stopped_beam = advance_structured_beam(
            code_beam,
            target_type=4,
            value=1,
            log_prob=-0.3,
            codebook_size=256,
            depth=4,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )

        self.assertEqual(code_beam["decoder_input_ids"][-1], 772)
        self.assertEqual(stopped_beam["decoder_input_ids"][-1], 1025)
        self.assertEqual(stopped_beam["completed"], [(1, 2, 3, 4)])
        self.assertTrue(stopped_beam["stopped"])

    def test_learned_structured_allowed_values_treats_fourth_code_as_sid_not_control(self) -> None:
        trie = build_prefix_trie([[1, 2, 3, 4], [1, 2, 3, 5]])

        self.assertEqual(
            structured_allowed_values(
                [1, 2, 3],
                [],
                target_type=3,
                trie=trie,
                depth=4,
                max_skills=2,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            ),
            [4, 5],
        )
        self.assertEqual(
            structured_allowed_values(
                [1, 2, 3, 4],
                [],
                target_type=4,
                trie=trie,
                depth=4,
                max_skills=2,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            ),
            [1, 0],
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_learned_structured_model_has_four_code_heads_and_no_dedup_parameters(self) -> None:
        import torch

        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(d_model=8)

        class DummyAutoModel:
            @staticmethod
            def from_pretrained(_base_model: str):
                return DummyBackbone()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = make_structured_t5_sid_model(
                "dummy",
                codebook_size=256,
                depth=4,
                dedup_size=0,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            )

        self.assertEqual(len(model.code_heads), 4)
        self.assertTrue(all(head.bias is not None for head in model.code_heads))
        self.assertIsNone(model.dedup_head)
        self.assertEqual(model.control_type, 4)
        self.assertEqual(model.sid_embedding.num_embeddings, 1027)
        self.assertFalse(any(name.startswith("dedup_head") for name in model.state_dict()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_ripor_progressive_prefix_slices_divergence_mask_and_count(self) -> None:
        import torch

        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(d_model=8)

        class DummyAutoModel:
            @staticmethod
            def from_pretrained(_base_model: str):
                return DummyBackbone()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = make_structured_t5_sid_model(
                "dummy",
                codebook_size=16,
                depth=4,
                dedup_size=0,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            )

        prefix_scores = torch.tensor([[[1.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0]]])
        model.ripor_prefix_log_scores = lambda *_args, **_kwargs: prefix_scores
        candidate_codes = torch.tensor([[[1, 2, 3, 4], [1, 9, 3, 4]]])
        loss_sum, pair_count, used_scores = model.ripor_prefix_rank_loss(
            torch.zeros(1, 1, 8),
            torch.ones(1, 1, dtype=torch.long),
            candidate_codes,
            torch.tensor([[True, True]]),
            torch.tensor([[True, False]]),
            margin=1.0,
            temperature=1.0,
            candidate_teacher_scores=torch.tensor([[3.0, 1.0]]),
            prefix_max_length=2,
        )

        alpha_at_two = ripor_prefix_alphas(4)[1]
        self.assertEqual(tuple(used_scores.shape), (1, 2, 2))
        self.assertEqual(float(pair_count), 1.0)
        self.assertAlmostEqual(float(loss_sum), (2.0 - 2.0 * alpha_at_two) ** 2, places=6)

    def test_advance_structured_beam_appends_codes_and_control_tokens(self) -> None:
        beam = {
            "decoder_input_ids": [772, 1, 259, 517],
            "current_prefix": [1, 3, 5],
            "completed": [],
            "score": -2.0,
            "stopped": False,
        }

        dedup_beam = advance_structured_beam(
            beam,
            target_type=3,
            value=0,
            log_prob=-0.25,
            codebook_size=256,
            depth=3,
            dedup_size=4,
        )
        stopped_beam = advance_structured_beam(
            dedup_beam,
            target_type=4,
            value=1,
            log_prob=-0.5,
            codebook_size=256,
            depth=3,
            dedup_size=4,
        )

        self.assertEqual(dedup_beam["decoder_input_ids"][-1], 768)
        self.assertEqual(dedup_beam["current_prefix"], [1, 3, 5, 0])
        self.assertEqual(dedup_beam["completed"], [])
        self.assertFalse(dedup_beam["stopped"])
        self.assertEqual(stopped_beam["decoder_input_ids"][-1], 773)
        self.assertEqual(stopped_beam["current_prefix"], [])
        self.assertEqual(stopped_beam["completed"], [(1, 3, 5, 0)])
        self.assertTrue(stopped_beam["stopped"])
        self.assertAlmostEqual(stopped_beam["score"], -2.75)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_ripor_candidate_batch_uses_current_positive_and_excludes_siblings(self) -> None:
        import torch

        semantic_ids = {"skill/a": [1, 2], "skill/b": [1, 3], "skill/c": [4, 5]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "index": "query-1",
                        "ranked_skill_ids": ["skill/b", "skill/c", "skill/a"],
                        "ranked_skill_scores": {"skill/b": 0.7, "skill/c": 0.2, "skill/a": 0.8},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_map, score_map = load_ranked_skill_candidates_with_scores(path)

        batch = build_ripor_candidate_batch(
            [
                {
                    "index": "query-1",
                    "selected_skill_ids": ["skill/a"],
                    "source_selected_skill_ids": ["skill/a", "skill/b"],
                }
            ],
            semantic_ids,
            depth=2,
            max_negatives=1,
            hard_negative_map=candidate_map,
            teacher_score_map=score_map,
            seed=7,
            device=torch.device("cpu"),
        )

        self.assertEqual(batch["ripor_candidate_positive"].tolist()[0], [True, False])
        self.assertAlmostEqual(float(batch["ripor_candidate_teacher_scores"][0, 0]), 0.8, places=5)
        self.assertEqual(tuple(batch["ripor_candidate_codes"].shape), (1, 2, 2))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_ripor_random_negative_sampling_changes_across_presentations(self) -> None:
        import torch

        semantic_ids = {
            "skill/a": [1, 1],
            "skill/b": [2, 2],
            "skill/c": [3, 3],
            "skill/d": [4, 4],
        }
        row = {"index": "query-1", "selected_skill_ids": ["skill/a"]}
        candidates = {"index:query-1": ["skill/b", "skill/c", "skill/d"]}
        scores = {"index:query-1": {"skill/a": 1.0, "skill/b": 0.3, "skill/c": 0.2, "skill/d": 0.1}}
        sampled_codes = set()
        for seed in range(8):
            batch = build_ripor_candidate_batch(
                [row],
                semantic_ids,
                depth=2,
                max_negatives=1,
                hard_negative_map=candidates,
                teacher_score_map=scores,
                seed=seed,
                device=torch.device("cpu"),
                require_teacher_scores=True,
                negative_sampling="random",
            )
            sampled_codes.add(tuple(batch["ripor_candidate_codes"][0, 1].tolist()))
        self.assertGreater(len(sampled_codes), 1)

    def test_letter_tokenizer_manifest_pins_external_lightgcn_semantic_ids(self) -> None:
        semantic_ids = {"a": [1, 2], "b": [3, 4]}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_path = root / "semantic_ids.jsonl"
            semantic_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "skill_id": skill_id,
                            "semantic_id": values,
                            "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
                        }
                    )
                    for skill_id, values in semantic_ids.items()
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "rqvae_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "method": "official_letter_rqvae_adapter",
                        "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
                        "skills": 2,
                        "num_emb_list": [256, 256],
                        "external_cf_embeddings": "/models/lightgcn.pt",
                        "cf_source": "/models/lightgcn.pt",
                    }
                ),
                encoding="utf-8",
            )

            provenance = validate_letter_tokenizer_manifest(
                manifest_path,
                semantic_path,
                semantic_ids,
                SEMANTIC_ID_MODE_LEARNED,
            )

        self.assertTrue(provenance["frozen"])
        self.assertEqual(provenance["skill_count"], 2)
        self.assertEqual(provenance["lightgcn_embeddings"], "/models/lightgcn.pt")

    def test_three_method_setting_contracts_keep_letter_ids_and_isolate_losses(self) -> None:
        common = {
            "letter_tokenizer_manifest": Path("rqvae_manifest.json"),
            "setting_contract": Path("setting_contract.json"),
            "ripor_relevance_checkpoint": None,
            "ripor_codebook_embeddings": None,
            "ripor_allow_weak_teacher": False,
            "pag_set_allow_weak_teacher": False,
            "reset_optimizer": True,
            "query_target_mode": "single_sid",
            "pag_set_stage_only": False,
            "pag_sparse_checkpoint": None,
            "catalog_epochs": 0,
            "eval_steps": 0,
            "load_best_model_at_end": False,
            "pag_set_loss_weight": 0.0,
            "pag_set_init_model_dir": None,
            "pag_seq_init_model_dir": None,
            "init_model_dir": Path("base"),
            "ripor_loss_weight": 1.0,
            "sequential_ce_weight": 0.0,
        }
        ripor_args = argparse.Namespace(method_setting=METHOD_SETTING_RIPOR, **common)
        self.assertEqual(
            validate_method_setting_contract(ripor_args, pag_set_enabled=False),
            METHOD_SETTING_RIPOR,
        )

        pag_values = {
            **common,
            "pag_set_loss_weight": 1.0,
            "pag_set_init_model_dir": Path("mset"),
            "pag_seq_init_model_dir": Path("base"),
            "init_model_dir": None,
            "ripor_loss_weight": 0.0,
            "sequential_ce_weight": 1.0,
        }
        pag_args = argparse.Namespace(method_setting=METHOD_SETTING_PAG, **pag_values)
        self.assertEqual(
            validate_method_setting_contract(pag_args, pag_set_enabled=True),
            METHOD_SETTING_PAG,
        )

        both_values = {**pag_values, "ripor_loss_weight": 1.0, "sequential_ce_weight": 0.0}
        both_args = argparse.Namespace(method_setting=METHOD_SETTING_RIPOR_PAG, **both_values)
        self.assertEqual(
            validate_method_setting_contract(both_args, pag_set_enabled=True),
            METHOD_SETTING_RIPOR_PAG,
        )

        ripor_args.ripor_relevance_checkpoint = Path("m0.pt")
        with self.assertRaisesRegex(ValueError, "cannot take RIPOR M0/RQ"):
            validate_method_setting_contract(ripor_args, pag_set_enabled=False)

        ripor_args.ripor_relevance_checkpoint = None
        ripor_args.query_target_mode = "structured"
        with self.assertRaisesRegex(ValueError, "require --query-target-mode single_sid"):
            validate_method_setting_contract(ripor_args, pag_set_enabled=False)

        ripor_args.query_target_mode = "single_sid"
        ripor_args.setting_contract = None
        with self.assertRaisesRegex(ValueError, "requires --setting-contract"):
            validate_method_setting_contract(ripor_args, pag_set_enabled=False)

        ripor_args.setting_contract = Path("setting_contract.json")
        ripor_args.ripor_allow_weak_teacher = True
        with self.assertRaisesRegex(ValueError, "forbids --ripor-allow-weak-teacher"):
            validate_method_setting_contract(ripor_args, pag_set_enabled=False)

        both_args.pag_set_allow_weak_teacher = True
        with self.assertRaisesRegex(ValueError, "forbids --pag-set-allow-weak-teacher"):
            validate_method_setting_contract(both_args, pag_set_enabled=True)

    def test_setting_contract_provenance_binds_checkpoint_to_contract_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "setting_contract.json"
            path.write_text(
                json.dumps(
                    {
                        "fixed_letter_semantic_ids": True,
                        "query_target_mode": "single_sid",
                        "settings": ["ripor", "pag", "ripor_pag"],
                        "input_files": {"rows": {"sha256": "abc"}},
                    }
                ),
                encoding="utf-8",
            )
            provenance = setting_contract_provenance(path)

        self.assertEqual(provenance["settings"], ["ripor", "pag", "ripor_pag"])
        self.assertEqual(provenance["input_sha256"], {"rows": "abc"})
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_pag_msp_protocol_amends_only_the_failed_sparse_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            setting_path = root / "setting_contract.json"
            protocol_path = root / "pag_msp_protocol.json"
            setting_path.write_text(
                json.dumps(
                    {
                        "fixed_letter_semantic_ids": True,
                        "query_target_mode": "single_sid",
                        "settings": ["ripor", "pag", "ripor_pag"],
                        "input_files": {},
                        "msp_steps": 4700,
                        "pag_sparse_max_length": 128,
                        "per_device_batch": 8,
                        "global_batch": 64,
                        "lr": 5e-4,
                        "warmup_ratio": 0.045,
                        "weight_decay": 0.0,
                        "world_size": 8,
                        "seed": 20260626,
                        "mset_steps": 5000,
                        "final_stage_steps": 5000,
                    }
                ),
                encoding="utf-8",
            )
            command_write_pag_msp_protocol(
                argparse.Namespace(
                    setting_contract=setting_path,
                    output=protocol_path,
                    failed_job_id="tide-test-collapse",
                )
            )
            setting = setting_contract_provenance(setting_path)
            protocol = pag_msp_protocol_provenance(protocol_path, setting)
            payload = json.loads(protocol_path.read_text(encoding="utf-8"))

            self.assertEqual(protocol["effective"]["max_steps"], 200000)
            self.assertEqual(protocol["effective"]["global_batch"], 256)
            self.assertEqual(payload["superseded_base_values"]["msp_steps"], 4700)
            self.assertEqual(payload["scope"].split(";")[0], "PAG Msp only")

            payload["effective"]["warmup_ratio"] = 0.045
            protocol_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deviates from the preregistered official profile"):
                pag_msp_protocol_provenance(protocol_path, setting)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_pag_merge_preserves_letter_sequential_heads_bitwise(self) -> None:
        import torch

        set_state = {
            "backbone.weight": torch.tensor([2.0, 4.0]),
            "sid_embedding.weight": torch.tensor([99.0, 99.0]),
            "code_heads.0.weight": torch.tensor([88.0]),
            "control_head.weight": torch.tensor([77.0]),
        }
        seq_state = {
            "backbone.weight": torch.tensor([4.0, 8.0]),
            "sid_embedding.weight": torch.tensor([1.0, 2.0]),
            "code_heads.0.weight": torch.tensor([3.0]),
            "control_head.weight": torch.tensor([4.0]),
        }
        merged = merge_pag_set_sequence_states(set_state, seq_state, seq_state)

        self.assertTrue(torch.equal(merged["backbone.weight"], torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.equal(merged["sid_embedding.weight"], seq_state["sid_embedding.weight"]))
        self.assertTrue(torch.equal(merged["code_heads.0.weight"], seq_state["code_heads.0.weight"]))
        self.assertTrue(torch.equal(merged["control_head.weight"], seq_state["control_head.weight"]))

    def test_scoreless_ranked_candidates_are_not_silent_teacher_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scoreless.jsonl"
            path.write_text(
                json.dumps({"index": "q", "ranked_skill_ids": ["skill/neg"]}) + "\n",
                encoding="utf-8",
            )
            candidate_map, score_map = load_ranked_skill_candidates_with_scores(path)

        self.assertTrue(candidate_map)
        self.assertTrue(all(not values for values in score_map.values()))
        with self.assertRaisesRegex(ValueError, "teacher scores"):
            build_teacher_rank_examples(
                [{"index": "q", "query": "q", "selected_skill_ids": ["skill/pos"]}],
                [
                    {"skill_id": "skill/pos", "skillrouter_text": "positive"},
                    {"skill_id": "skill/neg", "skillrouter_text": "negative"},
                ],
                candidate_map,
                score_map,
                max_negatives=1,
                allow_weak_teacher=False,
            )

    def test_teacher_rank_examples_do_not_repeat_single_sid_sibling_positives(self) -> None:
        examples = build_teacher_rank_examples(
            [
                {
                    "index": "q",
                    "query": "query",
                    "selected_skill_ids": ["skill/a"],
                    "source_selected_skill_ids": ["skill/a", "skill/b"],
                }
            ],
            [
                {"skill_id": "skill/a", "skillrouter_text": "a"},
                {"skill_id": "skill/b", "skillrouter_text": "b"},
                {"skill_id": "skill/c", "skillrouter_text": "c"},
            ],
            {"index:q": ["skill/b", "skill/c"]},
            {"index:q": {"skill/a": 0.9, "skill/b": 0.8, "skill/c": 0.1}},
            max_negatives=1,
            allow_weak_teacher=False,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["positive_skill_id"], "skill/a")
        self.assertEqual(examples[0]["negative_skill_id"], "skill/c")

    def test_pag_set_manifest_requires_fixed_m_when_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "set.json"
            path.write_text(
                json.dumps({"vocab_size": 20, "max_tokens": 2, "skill_token_ids": {"a": [3, 4]}}),
                encoding="utf-8",
            )
            token_ids, vocab_size = load_pag_set_token_ids(path)
        self.assertEqual(token_ids, {"a": [3, 4]})
        self.assertEqual(vocab_size, 20)

    def test_pag_flops_regularization_matches_official_quadratic_ramp(self) -> None:
        self.assertAlmostEqual(pag_quadratic_regularization_weight(0.01, 0, 300), 0.0)
        self.assertAlmostEqual(pag_quadratic_regularization_weight(0.01, 50, 300), 0.0025)
        self.assertAlmostEqual(pag_quadratic_regularization_weight(0.01, 100, 300), 0.01)
        self.assertAlmostEqual(pag_quadratic_regularization_weight(0.01, 250, 300), 0.01)
        with self.assertRaisesRegex(ValueError, "total_steps must be positive"):
            pag_quadratic_regularization_weight(0.01, 1, 0)

    def test_beam_skill_scores_snapshot_at_sid_completion(self) -> None:
        beam = {
            "decoder_input_ids": [772],
            "current_prefix": [],
            "completed": [],
            "completed_records": [],
            "score": 0.0,
            "model_score": 0.0,
            "planner_score": 0.0,
            "stopped": False,
        }
        code = advance_structured_beam(
            beam,
            0,
            1,
            2.0,
            codebook_size=256,
            depth=1,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )
        completed = advance_structured_beam(
            code,
            1,
            1,
            3.0,
            codebook_size=256,
            depth=1,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )
        continued = {**completed, "score": 100.0}
        self.assertEqual(completed["completed_records"][0]["score"], 5.0)
        self.assertEqual(continued["completed_records"][0]["score"], 5.0)
        self.assertEqual(completed["completed_records"][0]["decoder_length"], 3)

    def test_multi_sid_ranking_uses_completion_snapshots_globally(self) -> None:
        beams = [
            {
                "score": 0.0,
                "completed": [(1,)],
                "completed_records": [
                    {"semantic_id": (1,), "score": 2.0, "decoder_length": 2}
                ],
            },
            {
                "score": 100.0,
                "completed": [(2,)],
                "completed_records": [
                    {"semantic_id": (2,), "score": 1.0, "decoder_length": 2}
                ],
            },
        ]
        sequence_to_skill_id = {(1,): "skill/one", (2,): "skill/two"}
        self.assertEqual(
            ranked_skill_ids_from_structured_beams(beams, sequence_to_skill_id, max_rank=2),
            ["skill/one", "skill/two"],
        )
        ranked, scores = ranked_skill_ids_from_marginal_structured_beams(
            beams, sequence_to_skill_id, max_rank=2, length_normalize=True
        )
        self.assertEqual(ranked, ["skill/one", "skill/two"])
        self.assertGreater(scores["skill/one"], scores["skill/two"])

    def test_pag_planner_excludes_completed_skill_from_shared_prefix(self) -> None:
        descendants = {(1,): ["skill/done", "skill/sibling"]}
        scores = {"skill/done": 10.0, "skill/sibling": 1.0}
        self.assertEqual(
            pag_prefix_planner_score([1], scores, descendants, "max", excluded_skill_ids={"skill/done"}),
            1.0,
        )

    def test_pag_zero_weight_preserves_baseline_beam_score(self) -> None:
        beam = {
            "decoder_input_ids": [772],
            "current_prefix": [],
            "completed": [],
            "score": -2.0,
            "stopped": False,
        }
        baseline = advance_structured_beam(
            beam, 0, 1, -0.5, codebook_size=256, depth=3, dedup_size=4
        )
        pag_zero = advance_structured_beam(
            beam,
            0,
            1,
            -0.5,
            codebook_size=256,
            depth=3,
            dedup_size=4,
            planner_increment=100.0,
            planner_weight=0.0,
        )
        self.assertEqual(pag_zero["decoder_input_ids"], baseline["decoder_input_ids"])
        self.assertEqual(pag_zero["completed"], baseline["completed"])
        self.assertEqual(pag_zero["score"], baseline["score"])

    def test_ripor_prefix_alphas_are_concave_and_full_margin_is_one(self) -> None:
        values = ripor_prefix_alphas(4)
        self.assertEqual(len(values), 4)
        self.assertAlmostEqual(values[-1], 1.0)
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        increments = [right - left for left, right in zip(values, values[1:])]
        self.assertTrue(all(left >= right for left, right in zip(increments, increments[1:])))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_pag_decoder_query_prefix_uses_official_length_without_truncating_encoder(self) -> None:
        import torch

        input_ids = torch.arange(160).reshape(2, 80)
        attention_mask = torch.ones_like(input_ids)
        prefix_ids, prefix_mask = truncate_pag_query_prefix(
            input_ids,
            attention_mask,
            DEFAULT_PAG_QUERY_PREFIX_MAX_LENGTH,
        )

        self.assertEqual(tuple(prefix_ids.shape), (2, 64))
        self.assertEqual(tuple(prefix_mask.shape), (2, 64))
        self.assertTrue(torch.equal(prefix_ids, input_ids[:, :64]))
        self.assertEqual(tuple(input_ids.shape), (2, 80))

    def test_native_semantic_id_text_uses_base_vocab_words_and_digits(self) -> None:
        self.assertEqual(
            native_semantic_id_text([1, 3, 5, 0], code_width=3),
            "code 0 001 code 1 003 code 2 005 duplicate 000",
        )

    def test_decoder_target_can_emit_native_vocab_text_for_diagnostics(self) -> None:
        semantic_ids = {
            "skill/b": [2, 3, 4, 0],
            "skill/a": [1, 3, 5, 0],
        }

        target = build_decoder_target(
            ["skill/b", "skill/a"],
            semantic_ids,
            code_width=3,
            target_mode="native",
        )

        self.assertEqual(
            target,
            "code 0 001 code 1 003 code 2 005 duplicate 000 | "
            "code 0 002 code 1 003 code 2 004 duplicate 000",
        )

    def test_decoder_target_defaults_to_legacy_special_tokens(self) -> None:
        semantic_ids = {
            "skill/b": [2, 3, 4, 0],
            "skill/a": [1, 3, 5, 0],
        }

        target = build_decoder_target(["skill/b", "skill/a"], semantic_ids, code_width=3)

        self.assertEqual(
            target,
            "<RQ0_001> <RQ1_003> <RQ2_005> <DUP_000> "
            "<SKILL_SEP> "
            "<RQ0_002> <RQ1_003> <RQ2_004> <DUP_000>",
        )

    def test_stratified_split_keeps_balanced_k_buckets(self) -> None:
        rows = [
            {"id": f"{k}-{i}", "target_skill_count": k}
            for k in [1, 2, 3, 4]
            for i in range(500)
        ]

        train, dev, test, manifest = stratified_split_rows(rows, dev_ratio=0.05, test_ratio=0.05, seed=7)

        self.assertEqual(len(train), 1800)
        self.assertEqual(len(dev), 100)
        self.assertEqual(len(test), 100)
        for key in ["1", "2", "3", "4"]:
            self.assertEqual(manifest["split_counts"][key], {"train": 450, "dev": 25, "test": 25, "total": 500})
        all_ids = {row["id"] for row in train + dev + test}
        self.assertEqual(len(all_ids), 2000)

    def test_build_prefix_trie_preserves_shared_prefixes(self) -> None:
        trie = build_prefix_trie([[1, 2, 3], [1, 2, 4], [5, 6, 7]])

        self.assertIn("1", trie)
        self.assertIn("5", trie)
        self.assertIn("3", trie["1"]["2"])
        self.assertIn("4", trie["1"]["2"])
        self.assertEqual(trie["1"]["2"]["3"]["$"], [0])
        self.assertEqual(trie["1"]["2"]["4"]["$"], [1])
        self.assertEqual(trie["5"]["6"]["7"]["$"], [2])

    def test_allowed_next_tokens_reuses_single_skill_trie_after_separator(self) -> None:
        trie = build_prefix_trie([[10, 11], [10, 12], [20, 21]])

        self.assertEqual(
            sorted(allowed_next_tokens_for_skill_trie([], trie, sep_id=30, eos_id=1, max_skills=2)),
            [10, 20],
        )
        self.assertEqual(
            sorted(allowed_next_tokens_for_skill_trie([10], trie, sep_id=30, eos_id=1, max_skills=2)),
            [11, 12],
        )
        self.assertEqual(
            sorted(allowed_next_tokens_for_skill_trie([10, 11], trie, sep_id=30, eos_id=1, max_skills=2)),
            [1, 30],
        )
        self.assertEqual(
            sorted(allowed_next_tokens_for_skill_trie([10, 11, 30], trie, sep_id=30, eos_id=1, max_skills=2)),
            [10, 20],
        )
        self.assertEqual(
            allowed_next_tokens_for_skill_trie([10, 11, 30, 10], trie, sep_id=30, eos_id=1, max_skills=2),
            [12],
        )
        self.assertEqual(
            allowed_next_tokens_for_skill_trie([10, 11, 30, 20, 21], trie, sep_id=30, eos_id=1, max_skills=2),
            [1],
        )

    def test_compute_retrieval_metrics_reports_hits_and_recall_by_k(self) -> None:
        rows = [
            {"target_skill_count": 2, "selected_skill_ids": ["a", "b"]},
            {"target_skill_count": 1, "selected_skill_ids": ["c"]},
        ]
        predictions = [
            {"predicted_skill_ids": ["b", "x"]},
            {"predicted_skill_ids": ["c"]},
        ]

        metrics = compute_retrieval_metrics(rows, predictions)

        self.assertEqual(metrics["overall"]["rows"], 2)
        self.assertEqual(metrics["overall"]["total_hits"], 2)
        self.assertEqual(metrics["overall"]["total_gold"], 3)
        self.assertAlmostEqual(metrics["overall"]["micro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["overall"]["macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["overall"]["hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["overall"]["exact_match_rate"], 0.5)
        self.assertEqual(metrics["by_target_skill_count"]["1"]["total_hits"], 1)
        self.assertEqual(metrics["by_target_skill_count"]["2"]["total_hits"], 1)

    def test_compute_at_k_metrics_reports_hit_recall_and_full_coverage(self) -> None:
        rows = [
            {"target_skill_count": 2, "selected_skill_ids": ["a", "b"]},
            {"target_skill_count": 1, "selected_skill_ids": ["c"]},
        ]
        predictions = [
            {"ranked_skill_ids": ["x", "a", "b"]},
            {"ranked_skill_ids": ["z", "y", "c"]},
        ]

        metrics = compute_at_k_metrics(rows, predictions, k_values=[1, 2, 3])

        self.assertEqual(metrics["overall"]["@1"]["total_hits"], 0)
        self.assertAlmostEqual(metrics["overall"]["@1"]["hit"], 0.0)
        self.assertAlmostEqual(metrics["overall"]["@1"]["recall"], 0.0)
        self.assertAlmostEqual(metrics["overall"]["@1"]["fc"], 0.0)
        self.assertEqual(metrics["overall"]["@2"]["total_hits"], 1)
        self.assertAlmostEqual(metrics["overall"]["@2"]["hit"], 0.5)
        self.assertAlmostEqual(metrics["overall"]["@2"]["recall"], 0.25)
        self.assertAlmostEqual(metrics["overall"]["@2"]["macro_recall"], 0.25)
        self.assertAlmostEqual(metrics["overall"]["@2"]["micro_recall"], 1 / 3)
        self.assertAlmostEqual(metrics["overall"]["@2"]["fc"], 0.0)
        self.assertEqual(metrics["overall"]["@3"]["total_hits"], 3)
        self.assertAlmostEqual(metrics["overall"]["@3"]["hit"], 1.0)
        self.assertAlmostEqual(metrics["overall"]["@3"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["overall"]["@3"]["fc"], 1.0)
        self.assertAlmostEqual(metrics["by_target_skill_count"]["2"]["@3"]["fc"], 1.0)
        self.assertEqual(metrics["recall_aggregation"], "macro")

    def test_rescore_ranked_can_filter_multi_skill_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions_path = tmp_path / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps({"gold_skill_ids": ["a"], "ranked_skill_ids": ["a"]}),
                        json.dumps({"gold_skill_ids": ["b", "c"], "ranked_skill_ids": ["b", "x"]}),
                    ]
                )
                + "\n"
            )
            output_path = tmp_path / "multi.json"
            args = argparse.Namespace(
                predictions=predictions_path,
                output=output_path,
                subset="multi",
                k_values="1",
            )

            command_rescore_ranked(args)

            metrics = json.loads(output_path.read_text())
            self.assertEqual(metrics["overall"]["rows"], 1)
            self.assertAlmostEqual(metrics["overall"]["@1"]["recall"], 0.5)
            self.assertAlmostEqual(metrics["overall"]["@1"]["micro_recall"], 0.5)

    def test_expand_single_sid_query_rows_creates_one_training_example_per_positive_skill(self) -> None:
        rows = [
            {
                "index": 7,
                "decoder_input": "skill routing query: do the thing",
                "query": "do the thing",
                "target_skill_count": 2,
                "selected_skill_ids": ["skill/b", "skill/a"],
            }
        ]
        semantic_ids = {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 6, 0]}

        expanded = expand_single_sid_query_rows(rows, semantic_ids)

        self.assertEqual([row["selected_skill_ids"] for row in expanded], [["skill/a"], ["skill/b"]])
        self.assertEqual([row["positive_skill_id"] for row in expanded], ["skill/a", "skill/b"])
        self.assertEqual(expanded[0]["decoder_input"], "skill routing query: do the thing")
        self.assertEqual(expanded[0]["source_target_skill_count"], 2)

    def test_single_sid_expansion_and_source_support_raw_generated_schema(self) -> None:
        row = {
            "index": 7,
            "generated": {
                "query": "find the best audio extraction tool",
                "selected_skill_ids": ["skill/b", "skill/a"],
                "target_skill_count": 2,
            },
        }
        semantic_ids = {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 6, 0]}

        expanded = expand_single_sid_query_rows([row], semantic_ids)

        self.assertEqual([item["positive_skill_id"] for item in expanded], ["skill/a", "skill/b"])
        self.assertTrue(decoder_source_text(expanded[0]).endswith("find the best audio extraction tool"))
        self.assertEqual(expanded[0]["source_selected_skill_ids"], ["skill/a", "skill/b"])

    def test_single_sid_fallback_split_keeps_each_source_query_in_one_split(self) -> None:
        rows = [
            {
                "index": index,
                "query": f"query {index}",
                "selected_skill_ids": ["skill/a", "skill/b"],
                "target_skill_count": 2,
            }
            for index in range(10)
        ]
        semantic_ids = {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 6, 0]}

        train_rows, dev_rows = split_and_expand_single_sid_query_rows(rows, semantic_ids, dev_ratio=0.2, seed=7)

        train_indices = {row["index"] for row in train_rows}
        dev_indices = {row["index"] for row in dev_rows}
        self.assertFalse(train_indices & dev_indices)
        self.assertEqual(len(train_rows), 16)
        self.assertEqual(len(dev_rows), 4)

    def test_rank_capacity_rejects_k_larger_than_decoding_capacity(self) -> None:
        validate_rank_capacity(num_beams=50, max_skills=1, max_rank=50)
        with self.assertRaisesRegex(ValueError, "at most 40"):
            validate_rank_capacity(num_beams=10, max_skills=4, max_rank=50)

    def test_build_single_sid_catalog_rows_uses_skill_document_text_as_input(self) -> None:
        skills = [
            {"skill_id": "skill/b", "skill_name": "B", "description": "desc b", "body": "body b"},
            {"skill_id": "skill/a", "skill_name": "A", "description": "desc a", "body": "body a"},
            {"skill_id": "skill/missing", "skill_name": "M", "description": "desc", "body": "body"},
        ]
        semantic_ids = {"skill/a": [1, 2, 3, 0], "skill/b": [4, 5, 6, 0]}

        catalog_rows = build_single_sid_catalog_rows(skills, semantic_ids)

        self.assertEqual([row["positive_skill_id"] for row in catalog_rows], ["skill/a", "skill/b"])
        self.assertEqual(catalog_rows[0]["selected_skill_ids"], ["skill/a"])
        self.assertEqual(catalog_rows[0]["decoder_input"], "skill document: A | desc a | body a")
        self.assertEqual(catalog_rows[1]["decoder_input"], "skill document: B | desc b | body b")

    def test_exact_document_aliases_are_canonicalized_without_test_dependent_priority(self) -> None:
        skills = [
            {"skill_id": "repo/b", "skillrouter_text": "Weather | forecast | body"},
            {"skill_id": "repo/a", "skillrouter_text": "Weather   | forecast\n| body"},
            {"skill_id": "repo/c", "skillrouter_text": "Calendar | schedule | body"},
        ]

        canonical, aliases, alias_map, manifest = canonicalize_exact_skill_documents(skills)

        self.assertEqual([row["skill_id"] for row in canonical], ["repo/a", "repo/c"])
        self.assertEqual(alias_map, {"repo/a": "repo/a", "repo/b": "repo/a", "repo/c": "repo/c"})
        self.assertEqual(manifest["duplicate_document_groups"], 1)
        self.assertEqual(manifest["aliased_skills"], 1)
        self.assertEqual(len(aliases), 3)

    def test_training_rows_are_remapped_to_canonical_aliases_and_deduplicated(self) -> None:
        row = {
            "target_skill_count": 3,
            "selected_skill_ids": ["repo/b", "repo/c", "repo/a"],
            "positive_skill_id": "repo/b",
            "generated": {"selected_skill_ids": ["repo/b", "repo/a"], "target_skill_count": 2},
        }
        alias_map = {"repo/a": "repo/a", "repo/b": "repo/a", "repo/c": "repo/c"}

        remapped, audit = remap_row_skill_aliases(row, alias_map)

        self.assertEqual(remapped["selected_skill_ids"], ["repo/a", "repo/c"])
        self.assertEqual(remapped["positive_skill_id"], "repo/a")
        self.assertEqual(remapped["generated"]["selected_skill_ids"], ["repo/a"])
        self.assertEqual(remapped["target_skill_count"], 2)
        self.assertEqual(remapped["pre_alias_target_skill_count"], 3)
        self.assertEqual(remapped["canonicalized_skill_aliases"], {"repo/b": "repo/a"})
        self.assertTrue(audit["changed"])
        self.assertEqual(audit["removed_duplicate_mentions"], 2)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is available in the remote training environment")
    def test_minimum_cost_last_code_assignment_guarantees_unique_codes(self) -> None:
        codes = [(0, 0), (0, 0), (0, 1), (1, 0)]
        costs = [
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
            [0.0, 9.0, 9.0],
        ]

        resolved, audit = minimum_cost_unique_last_codes(codes, costs)

        self.assertEqual(resolved, [(0, 0), (0, 1), (0, 2), (1, 0)])
        self.assertEqual(len(set(resolved)), len(resolved))
        self.assertEqual(audit["repaired_prefixes"], 1)
        self.assertEqual(audit["repaired_items"], 3)
        self.assertEqual(audit["changed_items"], 2)
        self.assertEqual(audit["post_collision_groups"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_vectorized_letter_diversity_loss_is_bitwise_equivalent(self) -> None:
        import torch
        import torch.nn.functional as F

        def official_diversity_loss(self, x, x_q, indices, labels):
            indices_cluster = [labels[index.item()] for index in indices]
            indices_list = {
                target: [index for index, cluster in enumerate(labels) if cluster == target]
                for target in range(10)
            }
            pos_list = [indices_list[cluster] for cluster in indices_cluster]
            pos_sample = []
            for batch_index, positives in enumerate(pos_list):
                sampled = random.choice(positives)
                while sampled == indices[batch_index]:
                    sampled = random.choice(positives)
                pos_sample.append(sampled)
            y_true = torch.tensor(pos_sample, device=x_q.device)
            similarities = torch.matmul(x_q, self.embedding.weight.t())
            self_mask = torch.zeros_like(similarities)
            for batch_index, row in enumerate(self_mask):
                row[indices[batch_index]] = 1e12
            return F.cross_entropy((similarities - self_mask) / 1, y_true)

        torch.manual_seed(7)
        labels = [index % 10 for index in range(30)]
        indices = torch.tensor([0, 11, 22, 3, 14, 25], dtype=torch.long)
        old_embedding = torch.nn.Embedding(30, 4)
        new_embedding = torch.nn.Embedding(30, 4)
        new_embedding.load_state_dict(old_embedding.state_dict())
        old_quantizer = types.SimpleNamespace(embedding=old_embedding)
        new_quantizer = types.SimpleNamespace(embedding=new_embedding)
        old_input = torch.randn(6, 4, requires_grad=True)
        new_input = old_input.detach().clone().requires_grad_(True)

        random.seed(19)
        old_loss = official_diversity_loss(old_quantizer, old_input, old_input, indices, labels)
        random.seed(19)
        new_loss = vectorized_letter_diversity_loss_main_entry(
            new_quantizer,
            new_input,
            new_input,
            indices,
            labels,
        )
        old_loss.backward()
        new_loss.backward()

        self.assertTrue(torch.equal(old_loss, new_loss))
        self.assertTrue(torch.equal(old_input.grad, new_input.grad))
        self.assertTrue(torch.equal(old_embedding.weight.grad, new_embedding.weight.grad))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_last_code_costs_use_rq_residual_after_previous_code_levels(self) -> None:
        import torch

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.rows = torch.tensor([[1.0, 0.0], [1.0, 2.0]])

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                return self.rows[index], index

        class DummyVQ(torch.nn.Module):
            def __init__(self, values) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(values, dtype=torch.float32))

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Identity()
                self.rq = torch.nn.Module()
                self.rq.vq_layers = torch.nn.ModuleList(
                    [
                        DummyVQ([[1.0, 0.0], [0.0, 1.0]]),
                        DummyVQ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                    ]
                )

        costs = compute_last_code_costs(
            model=DummyModel(),
            dataset=DummyDataset(),
            all_indices=[(0, 0), (1, 1)],
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(costs.tolist(), [[0.0, 2.0, 8.0], [2.0, 0.0, 2.0]])

    def test_single_sid_catalog_commands_are_exposed(self) -> None:
        parser = build_arg_parser()

        structured_args = parser.parse_args(
            [
                "train-structured-decoder",
                "--rows",
                "train.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--output-dir",
                "out",
                "--skills-jsonl",
                "skills.jsonl",
                "--catalog-epochs",
                "3",
                "--eval-batch-size",
                "128",
                "--wandb-project",
                "SKillGR",
                "--wandb-run-name",
                "multi-long",
                "--init-model-dir",
                "multi_last",
                "--step-offset",
                "10000",
                "--initial-best-dev-loss",
                "2.1",
            ]
        )
        train_args = parser.parse_args(
            [
                "train-single-sid-decoder",
                "--rows",
                "train.jsonl",
                "--dev-rows",
                "dev.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--skills-jsonl",
                "skills.jsonl",
                "--output-dir",
                "out",
                "--init-model-dir",
                "single_best",
                "--step-offset",
                "10000",
                "--initial-best-dev-loss",
                "1.269171",
                "--eval-batch-size",
                "128",
            ]
        )
        eval_args = parser.parse_args(
            [
                "eval-single-sid-decoder",
                "--rows",
                "dev.jsonl",
                "--model-dir",
                "model",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--output-dir",
                "eval",
            ]
        )

        self.assertEqual(structured_args.command, "train-structured-decoder")
        self.assertEqual(structured_args.skills_jsonl, Path("skills.jsonl"))
        self.assertEqual(structured_args.catalog_epochs, 3)
        self.assertEqual(structured_args.eval_batch_size, 128)
        self.assertEqual(structured_args.wandb_project, "SKillGR")
        self.assertEqual(structured_args.wandb_run_name, "multi-long")
        self.assertEqual(structured_args.init_model_dir, Path("multi_last"))
        self.assertEqual(structured_args.step_offset, 10000)
        self.assertAlmostEqual(structured_args.initial_best_dev_loss, 2.1)
        self.assertEqual(structured_args.max_source_length, DEFAULT_GR_MAX_SOURCE_LENGTH)
        self.assertEqual(train_args.command, "train-single-sid-decoder")
        self.assertEqual(train_args.init_model_dir, Path("single_best"))
        self.assertEqual(train_args.step_offset, 10000)
        self.assertAlmostEqual(train_args.initial_best_dev_loss, 1.269171)
        self.assertEqual(train_args.eval_batch_size, 128)
        self.assertEqual(eval_args.command, "eval-single-sid-decoder")
        self.assertEqual(eval_args.max_skills, 1)
        self.assertIsNone(eval_args.max_source_length)

        structured_eval_args = parser.parse_args(
            [
                "eval-structured-decoder",
                "--rows",
                "dev.jsonl",
                "--model-dir",
                "model",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--output-dir",
                "eval",
            ]
        )
        prepare_args = parser.parse_args(
            [
                "prepare",
                "--dataset",
                "dataset.jsonl",
                "--skills",
                "skills.jsonl",
                "--output-dir",
                "prepared",
            ]
        )
        alias_args = parser.parse_args(
            [
                "canonicalize-skill-aliases",
                "--skills-jsonl",
                "skills.jsonl",
                "--rows",
                "train.jsonl",
                "--output-dir",
                "canonical",
            ]
        )
        self.assertEqual(structured_eval_args.max_skills, DEFAULT_GR_MAX_SKILLS)
        self.assertEqual(structured_eval_args.num_beams, DEFAULT_GR_NUM_BEAMS)
        self.assertEqual(prepare_args.query_max_chars, SKILLROUTER_QUERY_MAX_CHARS)
        self.assertEqual(alias_args.rows, [Path("train.jsonl")])


if __name__ == "__main__":
    unittest.main()
