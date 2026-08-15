from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    SKILLROUTER_QUERY_MAX_CHARS,
    SEMANTIC_ID_MODE_DEDUP,
    SEMANTIC_ID_MODE_LEARNED,
    advance_structured_beam,
    append_loss_history,
    audit_source_token_lengths,
    allowed_next_tokens_for_skill_trie,
    balanced_trie_multilabel_binary_cross_entropy,
    build_pag_prefix_to_skill_ids,
    build_single_sid_optimizer,
    build_decoder_target,
    build_arg_parser,
    build_query_balanced_multi_positive_targets,
    build_query_balanced_trie_multilabel_states,
    build_single_sid_catalog_rows,
    expand_single_sid_query_rows,
    build_prefix_trie,
    collect_query_skill_interactions,
    canonicalize_exact_skill_documents,
    compute_at_k_metrics,
    compute_ranked_subset_metrics,
    compute_retrieval_metrics,
    count_query_balanced_trie_multilabel_states,
    command_rescore_ranked,
    deduplicate_code_tuples,
    decoder_source_text,
    decoupled_multi_positive_cross_entropy,
    encode_structured_sid_target,
    format_skillrouter_query,
    format_skillrouter_skill,
    global_sid_token_id,
    infer_structured_sid_config,
    load_semantic_ids_with_mode,
    load_structured_decoder_checkpoint,
    make_structured_t5_sid_model,
    mean_active_loss_per_source_query_and_depth,
    mean_loss_per_source_query,
    native_semantic_id_text,
    pag_candidate_lexical_scores,
    pag_augmented_beam_score,
    pag_lexical_potential,
    ranked_skill_ids_from_pag_beams,
    prepared_selected_skill_ids,
    query_balanced_standard_ce_structured_sid_batch,
    remap_row_skill_aliases,
    resolve_eval_max_source_length,
    save_structured_decoder_checkpoint,
    semantic_ids_sha256,
    single_sid_stage_batch_objective,
    split_and_expand_single_sid_query_rows,
    stratified_split_rows,
    structured_allowed_values,
    structured_decoder_allowed_scores,
    structured_decoder_log_scores,
    summarize_source_token_lengths,
    trie_multi_positive_set_cross_entropy,
    validate_checkpoint_semantic_ids,
    validate_rank_capacity,
)
from train_letter_rqvae_official import (  # noqa: E402
    compute_last_code_costs,
    minimum_cost_unique_last_codes,
    vectorized_letter_diversity_loss_main_entry,
)


class SkillGRPipelineTest(unittest.TestCase):
    def test_query_objectives_never_replace_catalog_cross_entropy(self) -> None:
        flags = [False, False, False, True]
        self.assertEqual(single_sid_stage_batch_objective("catalog", *flags), "standard_ce")
        self.assertEqual(single_sid_stage_batch_objective("query", *flags), "query_balanced_trie_set_ce")

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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_single_sid_optimizer_assigns_encoder_decoder_and_sid_groups_once(self) -> None:
        import torch

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.encoder = torch.nn.Module()
                self.backbone.encoder.block = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.backbone.encoder.final_layer_norm = torch.nn.LayerNorm(2)
                self.backbone.shared = torch.nn.Embedding(3, 2)
                self.backbone.decoder = torch.nn.Linear(2, 2)
                self.sid_embedding = torch.nn.Embedding(4, 2)
                self.code_heads = torch.nn.ModuleList([torch.nn.Linear(2, 3)])

        model = DummyModel()
        optimizer, metadata = build_single_sid_optimizer(
            model,
            lr=3e-4,
            weight_decay=0.0,
            encoder_lr=1e-5,
            decoder_lr=3e-5,
            sid_lr=3e-4,
        )

        self.assertEqual([group["group_name"] for group in optimizer.param_groups], ["encoder", "decoder", "sid"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 3e-5, 3e-4])
        assigned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(expected))
        self.assertTrue(metadata["uses_module_learning_rates"])
        self.assertEqual(metadata["groups"]["encoder"]["first_parameter"], "backbone.encoder.block.0.weight")
        self.assertEqual(metadata["groups"]["decoder"]["first_parameter"], "backbone.shared.weight")
        self.assertEqual(metadata["groups"]["sid"]["first_parameter"], "sid_embedding.weight")

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            build_single_sid_optimizer(model, 3e-4, 0.0, encoder_lr=1e-5)

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

    def test_query_balanced_targets_share_positive_labels_only_at_matching_prefixes(self) -> None:
        semantic_ids = {
            "skill/a": [1, 3, 5, 7],
            "skill/b": [1, 4, 6, 8],
        }

        targets, source_group_ids, positive_labels = build_query_balanced_multi_positive_targets(
            [{"index": 9, "selected_skill_ids": ["skill/b", "skill/a"]}],
            semantic_ids,
            codebook_size=256,
            depth=4,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )

        self.assertEqual(len(targets), 2)
        self.assertEqual(source_group_ids, [0, 0])
        self.assertEqual([target["skill_ids"] for target in targets], [["skill/a"], ["skill/b"]])
        self.assertEqual(positive_labels[0][0], [1])
        self.assertEqual(positive_labels[1][0], [1])
        self.assertEqual(positive_labels[0][1], [3, 4])
        self.assertEqual(positive_labels[1][1], [3, 4])
        self.assertEqual(positive_labels[0][2], [5])
        self.assertEqual(positive_labels[1][2], [6])
        self.assertEqual(positive_labels[0][-1], [1])
        self.assertEqual(positive_labels[1][-1], [1])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_decoupled_multi_positive_ce_does_not_push_alternative_positives_down(self) -> None:
        import torch

        logits = torch.tensor([[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        positive_labels = torch.tensor([[0, 1], [0, 1]])

        losses = decoupled_multi_positive_cross_entropy(logits, labels, positive_labels)
        losses.sum().backward()

        expected = torch.log1p(torch.exp(torch.tensor(-2.0)))
        self.assertTrue(torch.allclose(losses, torch.full_like(losses, expected)))
        self.assertEqual(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[1, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)
        self.assertGreater(float(logits.grad[1, 2]), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_grouped_loss_gives_each_source_query_equal_weight(self) -> None:
        import torch

        row_losses = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
        source_group_ids = torch.tensor([0, 0, 1])

        loss = mean_loss_per_source_query(row_losses, source_group_ids)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 3.5)
        self.assertTrue(torch.allclose(row_losses.grad, torch.tensor([0.25, 0.25, 0.5])))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_trie_set_ce_marginalizes_positives_and_ignores_invalid_children(self) -> None:
        import torch

        logits = torch.tensor(
            [[2.0, 2.0, 0.0, 10.0], [3.0, -1.0, 8.0, 9.0]],
            requires_grad=True,
        )
        positive_labels = torch.tensor([[0, 1], [0, -1]])
        candidate_labels = torch.tensor([[0, 1, 2], [0, -1, -1]])

        losses, active = trie_multi_positive_set_cross_entropy(
            logits,
            positive_labels,
            candidate_labels,
        )
        losses.sum().backward()

        expected = torch.logsumexp(torch.tensor([2.0, 2.0, 0.0]), dim=0) - torch.logsumexp(
            torch.tensor([2.0, 2.0]), dim=0
        )
        self.assertTrue(torch.allclose(losses, torch.tensor([expected, 0.0])))
        self.assertEqual(active.tolist(), [True, False])
        self.assertAlmostEqual(float(logits.grad[0, 0]), float(logits.grad[0, 1]), places=7)
        self.assertEqual(float(logits.grad[0, 3]), 0.0)
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(4)))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_active_trie_loss_averages_states_then_depths_then_queries(self) -> None:
        import torch

        state_losses = torch.tensor([1.0, 3.0, 10.0, 5.0], requires_grad=True)
        source_group_ids = torch.tensor([0, 0, 0, 1])
        target_types = torch.tensor([0, 0, 1, 0])
        active_states = torch.tensor([True, True, False, True])

        loss = mean_active_loss_per_source_query_and_depth(
            state_losses,
            source_group_ids,
            target_types,
            active_states,
            semantic_length=2,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 3.5)
        self.assertTrue(torch.allclose(state_losses.grad, torch.tensor([0.25, 0.25, 0.0, 0.5])))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_trie_local_softmax_normalizes_only_allowed_children(self) -> None:
        import torch

        logits = torch.tensor([8.0, 2.0, 1.0, -3.0])
        raw = structured_decoder_log_scores(logits, "trie_local_softmax")
        allowed = structured_decoder_allowed_scores(raw, torch.tensor([1, 2]), "trie_local_softmax")

        self.assertTrue(torch.allclose(allowed.exp().sum(), torch.tensor(1.0)))
        self.assertGreater(float(allowed[0]), float(allowed[1]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_conditional_logit_scores_preserve_raw_logits(self) -> None:
        import torch

        logits = torch.tensor([[8.0, 2.0, 1.0, -3.0]])
        scores = structured_decoder_log_scores(logits, "conditional_logit")
        allowed = structured_decoder_allowed_scores(
            scores[0], torch.tensor([1, 2]), "conditional_logit"
        )

        self.assertTrue(torch.equal(scores, logits))
        self.assertTrue(torch.equal(allowed, torch.tensor([2.0, 1.0])))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_query_balanced_standard_ce_averages_tokens_then_targets_then_queries(self) -> None:
        import torch

        class DummyTokenizer:
            def __call__(self, texts, **_kwargs):
                return {
                    "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
                }

        class DummyEncoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                hidden = torch.zeros((*input_ids.shape, 4), dtype=torch.float32, device=input_ids.device)
                return types.SimpleNamespace(last_hidden_state=hidden)

        class DummyDecoder(torch.nn.Module):
            def forward(self, inputs_embeds, **_kwargs):
                return types.SimpleNamespace(last_hidden_state=inputs_embeds)

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(d_model=4)
                self.encoder = DummyEncoder()
                self.decoder = DummyDecoder()

        class DummyAutoModel:
            @staticmethod
            def from_pretrained(_base_model: str):
                return DummyBackbone()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = make_structured_t5_sid_model(
                "dummy",
                codebook_size=4,
                depth=2,
                dedup_size=0,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            )

        semantic_ids = {
            "skill/a": [0, 1],
            "skill/b": [1, 2],
            "skill/c": [2, 3],
        }
        rows = [
            {"query": "first", "selected_skill_ids": ["skill/a", "skill/b"]},
            {"query": "second", "selected_skill_ids": ["skill/c"]},
        ]

        def batch(batch_rows):
            return query_balanced_standard_ce_structured_sid_batch(
                batch_rows,
                DummyTokenizer(),
                semantic_ids,
                codebook_size=4,
                depth=2,
                dedup_size=0,
                max_source_length=16,
                device=torch.device("cpu"),
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            )

        combined_batch = batch(rows)
        combined_loss = model(**combined_batch)
        first_query_loss = model(**batch([rows[0]]))
        second_query_loss = model(**batch([rows[1]]))

        self.assertEqual(combined_batch["input_ids"].shape[0], 2)
        self.assertEqual(combined_batch["decoder_input_ids"].shape[0], 3)
        self.assertEqual(combined_batch["source_group_ids"].tolist(), [0, 0, 1])
        self.assertTrue(torch.allclose(combined_loss, (first_query_loss + second_query_loss) / 2))
        combined_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_query_balanced_trie_states_merge_gold_children_at_shared_prefix(self) -> None:
        semantic_ids = {
            "skill/a": [1, 3, 5, 7],
            "skill/b": [1, 4, 6, 8],
            "skill/c": [2, 9, 10, 11],
        }
        rows = [{"index": 1, "selected_skill_ids": ["skill/b", "skill/a"]}]

        states, source_group_ids = build_query_balanced_trie_multilabel_states(
            rows,
            semantic_ids,
            codebook_size=256,
            depth=4,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
        )

        self.assertEqual(len(states), 8)
        self.assertEqual(count_query_balanced_trie_multilabel_states(rows, semantic_ids), 8)
        self.assertEqual(source_group_ids, [0] * 8)
        root = next(state for state in states if state["target_type"] == 0)
        shared_prefix = next(
            state for state in states if state["target_type"] == 1 and state["prefix"] == [1]
        )
        controls = [state for state in states if state["target_type"] == 4]
        self.assertEqual(root["positive_labels"], [1])
        self.assertEqual(root["candidate_labels"], [1, 2])
        self.assertEqual(shared_prefix["positive_labels"], [3, 4])
        self.assertEqual(shared_prefix["candidate_labels"], [3, 4])
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(state["candidate_labels"] == [0, 1] for state in controls))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_balanced_trie_multilabel_bce_has_independent_positive_gradients(self) -> None:
        import torch

        logits = torch.tensor([[2.0, 2.0, 0.0]], requires_grad=True)
        positive_labels = torch.tensor([[0, 1]])
        candidate_labels = torch.tensor([[0, 1, 2]])

        loss = balanced_trie_multilabel_binary_cross_entropy(
            logits,
            positive_labels,
            candidate_labels,
            negative_weight=1.0,
        ).mean()
        loss.backward()

        expected = torch.nn.functional.softplus(torch.tensor(-2.0)) + torch.nn.functional.softplus(torch.tensor(0.0))
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertAlmostEqual(float(logits.grad[0, 0]), float(logits.grad[0, 1]), places=7)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_sigmoid_decoder_scores_do_not_normalize_across_positive_children(self) -> None:
        import torch

        logits = torch.tensor([[2.0, 0.0]])
        sigmoid_scores = structured_decoder_log_scores(logits, "sigmoid")
        softmax_scores = structured_decoder_log_scores(logits, "softmax")

        self.assertAlmostEqual(float(sigmoid_scores[0, 1]), -0.693147, places=5)
        self.assertLess(float(softmax_scores[0, 1]), -2.0)

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
        self.assertIsNone(model.dedup_head)
        self.assertEqual(model.control_type, 4)
        self.assertEqual(model.sid_embedding.num_embeddings, 1027)
        self.assertFalse(any(name.startswith("dedup_head") for name in model.state_dict()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_codebook_initialization_ties_decoder_inputs_and_code_outputs(self) -> None:
        import torch

        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(d_model=8)
                self.shared = torch.nn.Embedding(16, 8)
                with torch.no_grad():
                    self.shared.weight.fill_(1.0)

        class DummyAutoModel:
            @staticmethod
            def from_pretrained(_base_model: str):
                return DummyBackbone()

        initial = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8) / 10.0
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForSeq2SeqLM = DummyAutoModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = make_structured_t5_sid_model(
                "dummy",
                codebook_size=4,
                depth=2,
                dedup_size=0,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
                codebook_embeddings=initial,
                tie_code_input_output=True,
            )
            restored = make_structured_t5_sid_model(
                "dummy",
                codebook_size=4,
                depth=2,
                dedup_size=0,
                semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
                tie_code_input_output=True,
            )

        self.assertTrue(model.tie_code_input_output)
        self.assertTrue(model.sid_codebook_initialized)
        self.assertIs(model.code_heads[0].embedding, model.sid_embedding)
        expected = initial / initial.square().mean().sqrt()
        self.assertTrue(torch.allclose(model.sid_embedding.weight[:4], expected[0]))
        self.assertTrue(torch.allclose(model.sid_embedding.weight[4:8], expected[1]))
        self.assertAlmostEqual(model.sid_codebook_init_scale, 1.0 / float(initial.square().mean().sqrt()))

        hidden = torch.ones((1, 8), requires_grad=True)
        model.code_heads[1](hidden).sum().backward()
        self.assertIsNotNone(model.sid_embedding.weight.grad)
        self.assertEqual(int(torch.count_nonzero(model.sid_embedding.weight.grad[:4])), 0)
        self.assertGreater(int(torch.count_nonzero(model.sid_embedding.weight.grad[4:8])), 0)

        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(restored.sid_embedding.weight, model.sid_embedding.weight))

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

    def test_pag_prefix_potential_uses_best_unseen_descendant(self) -> None:
        semantic_ids = {
            "skill-a": [1, 2],
            "skill-b": [1, 3],
            "skill-c": [4, 5],
        }
        prefix_index = build_pag_prefix_to_skill_ids(semantic_ids)
        sequence_to_skill_id = {tuple(values): sid for sid, values in semantic_ids.items()}
        lexical_scores = {"skill-a": 2.0, "skill-b": 5.0, "skill-c": 9.0}
        beam = {
            "current_prefix": [1],
            "completed": [],
            "score": -3.0,
            "stopped": False,
        }

        self.assertEqual(
            pag_lexical_potential(beam, prefix_index, sequence_to_skill_id, lexical_scores),
            5.0,
        )
        self.assertEqual(
            pag_augmented_beam_score(
                beam,
                prefix_index,
                sequence_to_skill_id,
                lexical_scores,
                alpha=0.25,
            ),
            -1.75,
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available in the remote training environment")
    def test_pag_candidate_lexical_scores_max_pools_query_positions(self) -> None:
        import torch

        query_hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]])
        query_mask = torch.tensor([[1, 1, 0]])
        vocabulary_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        candidate_ids = torch.tensor([[[0, 1], [2, 0]]])
        candidate_mask = torch.tensor([[[1, 1], [1, 0]]])

        scores = pag_candidate_lexical_scores(
            query_hidden,
            query_mask,
            vocabulary_weight,
            candidate_ids,
            candidate_mask,
        )

        expected = torch.tensor(
            [[math.log1p(1.0) + math.log1p(2.0), math.log1p(2.0)]]
        )
        self.assertTrue(torch.allclose(scores, expected))

    def test_pag_prefix_potential_does_not_reuse_completed_skill(self) -> None:
        semantic_ids = {"skill-a": [1, 2], "skill-b": [1, 3]}
        prefix_index = build_pag_prefix_to_skill_ids(semantic_ids)
        sequence_to_skill_id = {tuple(values): sid for sid, values in semantic_ids.items()}
        lexical_scores = {"skill-a": 7.0, "skill-b": 2.0}
        continuing = {
            "current_prefix": [],
            "completed": [(1, 2)],
            "score": -1.0,
            "stopped": False,
        }
        stopped = {**continuing, "stopped": True}

        self.assertEqual(
            pag_lexical_potential(continuing, prefix_index, sequence_to_skill_id, lexical_scores),
            9.0,
        )
        self.assertEqual(
            pag_lexical_potential(stopped, prefix_index, sequence_to_skill_id, lexical_scores),
            7.0,
        )

    def test_pag_final_ranking_uses_semantic_scores_only(self) -> None:
        sequence_to_skill_id = {(0,): "semantic-best", (1,): "lexical-best"}
        beams = [
            {"score": -1.0, "completed": [[0]], "stopped": True},
            {"score": -2.0, "completed": [[1]], "stopped": True},
        ]

        ranked, scores = ranked_skill_ids_from_pag_beams(
            beams,
            sequence_to_skill_id,
            {(0,): ("semantic-best",), (1,): ("lexical-best",)},
            {"semantic-best": 0.0, "lexical-best": 100.0},
            alpha=10.0,
            max_rank=2,
        )

        self.assertEqual(ranked, ["semantic-best", "lexical-best"])
        self.assertEqual(scores, {"semantic-best": -1.0, "lexical-best": -2.0})

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

    def test_compute_ranked_subset_metrics_reports_single_and_multi(self) -> None:
        rows = [
            {"selected_skill_ids": ["a"]},
            {"selected_skill_ids": ["b", "c"]},
        ]
        predictions = [
            {"ranked_skill_ids": ["a", "x"]},
            {"ranked_skill_ids": ["b", "x"]},
        ]

        metrics = compute_ranked_subset_metrics(rows, predictions, [1, 2])

        self.assertEqual(metrics["overall"]["overall"]["rows"], 2)
        self.assertEqual(metrics["single"]["overall"]["rows"], 1)
        self.assertEqual(metrics["multi"]["overall"]["rows"], 1)
        self.assertAlmostEqual(metrics["single"]["overall"]["@1"]["macro_recall"], 1.0)
        self.assertAlmostEqual(metrics["multi"]["overall"]["@1"]["macro_recall"], 0.5)

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
                "--periodic-document-epochs",
                "1",
                "--interleave-document-batches",
                "--query-balanced-multi-positive-ce",
                "--query-batch-size",
                "32",
                "--document-batch-size",
                "16",
                "--disable-dev-eval",
                "--official-eval-rows",
                "official.jsonl",
            ]
        )
        differential_lr_args = parser.parse_args(
            [
                "train-single-sid-decoder",
                "--rows",
                "train.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--skills-jsonl",
                "skills.jsonl",
                "--output-dir",
                "differential-lr-out",
                "--init-model-dir",
                "catalog-final",
                "--reset-optimizer-on-init",
                "--encoder-lr",
                "1e-5",
                "--decoder-lr",
                "3e-5",
                "--sid-lr",
                "3e-4",
            ]
        )
        trie_train_args = parser.parse_args(
            [
                "train-single-sid-decoder",
                "--rows",
                "train.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--skills-jsonl",
                "skills.jsonl",
                "--output-dir",
                "trie-out",
                "--query-balanced-trie-multilabel-bce",
                "--trie-multilabel-negative-weight",
                "0.5",
            ]
        )
        balanced_ce_args = parser.parse_args(
            [
                "train-single-sid-decoder",
                "--rows",
                "train.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--skills-jsonl",
                "skills.jsonl",
                "--output-dir",
                "balanced-ce-out",
                "--query-balanced-standard-ce",
            ]
        )
        trie_set_ce_args = parser.parse_args(
            [
                "train-single-sid-decoder",
                "--rows",
                "train.jsonl",
                "--semantic-ids",
                "semantic_ids.jsonl",
                "--skills-jsonl",
                "skills.jsonl",
                "--output-dir",
                "trie-set-ce-out",
                "--query-balanced-trie-set-ce",
                "--disable-dev-eval",
                "--official-eval-rows",
                "official.jsonl",
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
        self.assertEqual(train_args.periodic_document_epochs, 1)
        self.assertTrue(train_args.interleave_document_batches)
        self.assertTrue(train_args.query_balanced_multi_positive_ce)
        self.assertEqual(train_args.query_batch_size, 32)
        self.assertEqual(train_args.document_batch_size, 16)
        self.assertTrue(train_args.disable_dev_eval)
        self.assertEqual(train_args.official_eval_rows, Path("official.jsonl"))
        self.assertEqual(train_args.official_eval_every_epochs, 2)
        self.assertEqual(train_args.official_selection_subset, "overall")
        self.assertEqual(train_args.official_selection_k, 1)
        self.assertTrue(differential_lr_args.reset_optimizer_on_init)
        self.assertEqual(differential_lr_args.encoder_lr, 1e-5)
        self.assertEqual(differential_lr_args.decoder_lr, 3e-5)
        self.assertEqual(differential_lr_args.sid_lr, 3e-4)
        self.assertTrue(trie_train_args.query_balanced_trie_multilabel_bce)
        self.assertAlmostEqual(trie_train_args.trie_multilabel_negative_weight, 0.5)
        self.assertTrue(balanced_ce_args.query_balanced_standard_ce)
        self.assertTrue(trie_set_ce_args.query_balanced_trie_set_ce)
        self.assertTrue(trie_set_ce_args.disable_dev_eval)
        self.assertEqual(eval_args.command, "eval-single-sid-decoder")
        self.assertEqual(eval_args.max_skills, 1)
        self.assertIsNone(eval_args.max_source_length)
        self.assertEqual(eval_args.decoder_score_mode, "auto")

        conditional_eval_args = parser.parse_args(
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
                "--split-name",
                "dev",
                "--decoder-score-mode",
                "conditional_logit",
            ]
        )
        self.assertEqual(conditional_eval_args.decoder_score_mode, "conditional_logit")

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
