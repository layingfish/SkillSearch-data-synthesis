import math
import random
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from skill_gr_dgr import (  # noqa: E402
    _load_candidate_shards,
    _load_teacher_query_embedding_shards,
    build_parser,
    build_semantic_trie_children,
    build_trie_decode_tables,
    candidate_retrieval_audit,
    candidate_prefix_logit_scores,
    candidate_prefix_logprob_scores,
    candidate_sequence_log_scores,
    distilled_ranknet_loss,
    dgr_row_fingerprint,
    dgr_rows_fingerprint,
    loss_gradient_alignment,
    parse_trainable_parameter_groups,
    prefix_margin_mse_loss,
    progressive_prefix_indices,
    select_dgr_candidate_list,
    select_prefix_candidate_list,
    teacher_best_gold_indices_for_rows,
    teacher_margins_for_candidates,
    topk_single_sid_scores_from_encoder,
    structured_parameter_group,
)


def test_build_semantic_trie_children_tracks_valid_prefixes():
    children, sequence_to_skill, prefix_counts = build_semantic_trie_children(
        {"a": [0, 0], "b": [0, 1], "c": [1, 0]}
    )
    assert children[()] == (0, 1)
    assert children[(0,)] == (0, 1)
    assert children[(1,)] == (0,)
    assert sequence_to_skill[(0, 1)] == "b"
    assert prefix_counts == [2, 3]


def test_select_dgr_candidate_list_filters_all_gold_and_is_seeded():
    candidates = [1, 2, 3, 4, 5, 6, 7]
    ranks = [5, 4, 0, 6, 2, 1, 3]
    first = select_dgr_candidate_list(
        candidates,
        ranks,
        gold_indices={1, 2},
        positive_index=2,
        list_size=5,
        rng=random.Random(17),
    )
    second = select_dgr_candidate_list(
        candidates,
        ranks,
        gold_indices={1, 2},
        positive_index=2,
        list_size=5,
        rng=random.Random(17),
    )
    assert first == second
    selected_indices, selected_ranks = first
    assert selected_indices[0] == 2
    assert selected_ranks[0] == -1
    assert selected_indices[1] == 3  # teacher rank zero
    assert not ({1, 2} & set(selected_indices[1:]))
    assert len(selected_indices) == 5


def test_select_prefix_candidate_list_uses_student_order_and_filters_all_gold():
    selected = select_prefix_candidate_list(
        [4, 2, 1, 5, 3, 6],
        gold_indices={1, 2, 3},
        positive_index=2,
        list_size=3,
    )
    assert selected == [2, 4, 5]


def test_candidate_retrieval_audit_separates_student_and_teacher_top1():
    audit = candidate_retrieval_audit(
        [
            {"index": "single", "selected_skill_ids": ["a"]},
            {"index": "multi", "selected_skill_ids": ["b", "c"]},
        ],
        torch.tensor([[0, 1, 2], [0, 1, 3]]),
        torch.tensor([[0, 1, 2], [2, 0, 1]]),
        {"a": 0, "b": 1, "c": 2, "d": 3},
    )
    assert audit["overall"]["student_top1_recall"] == 0.5
    assert audit["overall"]["teacher_top1_within_student_pool_recall"] == 1.0
    assert audit["single"]["student_top_n_macro_recall"] == 1.0
    assert audit["multi"]["student_top_n_macro_recall"] == 0.5


def test_student_only_candidate_shards_load_with_disabled_teacher_ranks(tmp_path):
    rows = [{"index": 1, "query": "q", "selected_skill_ids": ["a"]}]
    (tmp_path / "shards").mkdir()
    torch.save(
        {
            "start": 0,
            "end": 1,
            "candidate_indices": torch.tensor([[2, 3]], dtype=torch.int32),
            "row_fingerprints": [dgr_row_fingerprint(rows[0])],
        },
        tmp_path / "shards" / "part-000000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": 1,
                "teacher_score": "disabled",
                "shards": ["part-000000.pt"],
            }
        ),
        encoding="utf-8",
    )

    candidates, teacher_ranks, manifest = _load_candidate_shards(tmp_path, rows)
    assert candidates.tolist() == [[2, 3]]
    assert teacher_ranks.tolist() == [[-1, -1]]
    assert manifest["teacher_score"] == "disabled"


def test_teacher_query_embedding_shards_are_bound_to_ordered_rows(tmp_path):
    rows = [
        {"index": 1, "query": "q1", "selected_skill_ids": ["a"]},
        {"index": 2, "query": "q2", "selected_skill_ids": ["b"]},
    ]
    (tmp_path / "shards").mkdir()
    torch.save(
        {
            "start": 0,
            "end": 2,
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        tmp_path / "shards" / "part-000000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": 2,
                "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
                "embedding_dim": 2,
                "shards": ["part-000000.pt"],
            }
        ),
        encoding="utf-8",
    )

    embeddings, manifest = _load_teacher_query_embedding_shards(tmp_path, rows)
    assert embeddings.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert manifest["rows"] == 2

    try:
        _load_teacher_query_embedding_shards(tmp_path, list(reversed(rows)))
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("reordered teacher-query rows must be rejected")


def test_mine_parser_allows_student_only_candidate_mining():
    args = build_parser().parse_args(
        [
            "mine",
            "--rows",
            "rows.jsonl",
            "--semantic-ids",
            "semantic_ids.jsonl",
            "--skills-jsonl",
            "skills.jsonl",
            "--checkpoint-dir",
            "checkpoint",
            "--output-dir",
            "candidates",
            "--skip-teacher-scores",
            "--student-score-mode",
            "conditional_logit",
        ]
    )
    assert args.skip_teacher_scores
    assert args.teacher_skill_embeddings is None
    assert args.checkpoint_dir == Path("checkpoint")
    assert args.student_score_mode == "conditional_logit"


def test_train_parser_accepts_teacher_margin_ripor_controls():
    args = build_parser().parse_args(
        [
            "train",
            "--rows",
            "rows.jsonl",
            "--dev-rows",
            "dev.jsonl",
            "--semantic-ids",
            "semantic_ids.jsonl",
            "--skills-jsonl",
            "skills.jsonl",
            "--candidates-dir",
            "candidates",
            "--init-model-dir",
            "checkpoint",
            "--output-dir",
            "output",
            "--rank-objective",
            "ripor_prefix_margin_mse",
            "--prefix-target-mode",
            "teacher_margin",
            "--prefix-positive-policy",
            "teacher_best_gold",
            "--teacher-query-embeddings-dir",
            "teacher_queries",
            "--teacher-skill-embeddings",
            "teacher_skills.npy",
            "--teacher-margin-scale",
            "12.5",
            "--teacher-positive-margins-only",
            "--prefix-score-mode",
            "softmax_stop",
            "--generation-weight",
            "1.0",
            "--rank-weight",
            "0.05",
            "--rank-gradient-target-ratio",
            "0.5",
            "--prefix-eval-standard-softmax",
            "--include-init-in-checkpoint-selection",
            "--trainable-parameter-groups",
            "sid_heads",
        ]
    )
    assert args.prefix_target_mode == "teacher_margin"
    assert args.prefix_positive_policy == "teacher_best_gold"
    assert args.teacher_query_embeddings_dir == Path("teacher_queries")
    assert args.teacher_skill_embeddings == Path("teacher_skills.npy")
    assert args.teacher_margin_scale == 12.5
    assert args.teacher_positive_margins_only
    assert args.prefix_score_mode == "softmax_stop"
    assert args.generation_weight == 1.0
    assert args.rank_weight == 0.05
    assert args.rank_gradient_target_ratio == 0.5
    assert args.prefix_eval_standard_softmax
    assert args.include_init_in_checkpoint_selection
    assert args.trainable_parameter_groups == "sid_heads"


def test_parse_encode_teacher_skills_configuration() -> None:
    args = build_parser().parse_args(
        [
            "encode-teacher-skills",
            "--skills-jsonl",
            "skills.jsonl",
            "--output",
            "teacher_skills.npy",
            "--teacher-model",
            "pipizhao/SkillRouter-Embedding-0.6B",
            "--batch-size",
            "4",
        ]
    )
    assert args.teacher_model == "pipizhao/SkillRouter-Embedding-0.6B"
    assert args.batch_size == 4
    assert args.desc_max == 300
    assert args.body_max == 2500


def test_parser_accepts_official_ripor_conditional_logit_eval():
    args = build_parser().parse_args(
        [
            "eval-retrieval",
            "--rows",
            "dev.jsonl",
            "--semantic-ids",
            "semantic_ids.jsonl",
            "--model-dir",
            "checkpoint",
            "--output-dir",
            "eval",
            "--expected-rows",
            "1000",
            "--score-mode",
            "conditional_logit",
            "--no-include-stop",
        ]
    )

    assert args.command == "eval-retrieval"
    assert args.expected_rows == 1000
    assert args.score_mode == "conditional_logit"
    assert args.include_stop is False


def test_trainable_parameter_groups_partition_backbone_and_sid_heads():
    assert structured_parameter_group("backbone.encoder.block.0.layer.0.weight") == "encoder_blocks"
    assert structured_parameter_group("backbone.encoder.final_layer_norm.weight") == "encoder_blocks"
    assert structured_parameter_group("backbone.decoder.block.0.layer.0.weight") == "decoder_and_shared"
    assert structured_parameter_group("backbone.shared.weight") == "decoder_and_shared"
    assert structured_parameter_group("code_heads.0.weight") == "sid_heads"
    assert parse_trainable_parameter_groups("sid_heads") == {"sid_heads"}
    assert parse_trainable_parameter_groups("encoder_blocks,sid_heads") == {
        "encoder_blocks",
        "sid_heads",
    }
    assert parse_trainable_parameter_groups("all") == {
        "encoder_blocks",
        "decoder_and_shared",
        "sid_heads",
    }


def test_distilled_ranknet_loss_matches_incremental_margin_hinge():
    zero_loss = distilled_ranknet_loss(
        torch.tensor([[10.0, 5.0, 7.0]]),
        torch.tensor([[-1, 1, 0]]),
        base_margin=2.0,
        gap_margin=1.0,
    )
    assert float(zero_loss) == 0.0

    scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    loss = distilled_ranknet_loss(
        scores,
        torch.tensor([[-1, 1, 0]]),
        base_margin=2.0,
        gap_margin=1.0,
    )
    assert math.isclose(float(loss.detach()), 3.0, rel_tol=0.0, abs_tol=1e-6)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


class _FakeCandidateScoreModel:
    dedup_head = None

    def _decoder_hidden_from_encoder(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        return decoder_input_ids.float().unsqueeze(-1)

    def logits_for_hidden_type(self, hidden, target_type):
        rows = hidden.shape[0]
        return torch.tensor([[0.0, 1.0, 2.0]]).repeat(rows, 1)


def test_candidate_sequence_score_is_sum_of_full_head_log_probs():
    model = _FakeCandidateScoreModel()
    codes = torch.tensor([[[0, 1], [2, 0]]])
    scores = candidate_sequence_log_scores(
        model,
        torch.zeros((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.long),
        codes,
        codebook_size=3,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
    )
    log_probs = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    expected = torch.tensor([[log_probs[0] + log_probs[1], log_probs[2] + log_probs[0]]])
    assert torch.allclose(scores, expected)


def test_candidate_prefix_scores_use_raw_logits_and_cumulative_sum():
    model = _FakeCandidateScoreModel()
    codes = torch.tensor([[[0, 1], [2, 0]]])
    scores = candidate_prefix_logit_scores(
        model,
        torch.zeros((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.long),
        codes,
        codebook_size=3,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
    )
    assert torch.equal(scores, torch.tensor([[[0.0, 1.0], [2.0, 2.0]]]))


def test_candidate_prefix_logprob_scores_match_softmax_inference_and_final_stop():
    model = _FakeCandidateScoreModel()
    codes = torch.tensor([[[0, 1], [2, 0]]])
    scores = candidate_prefix_logprob_scores(
        model,
        torch.zeros((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.long),
        codes,
        codebook_size=3,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
        include_final_stop=True,
    )
    log_probs = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    expected = torch.tensor(
        [
            [
                [log_probs[0], log_probs[0] + 2 * log_probs[1]],
                [log_probs[2], log_probs[2] + log_probs[0] + log_probs[1]],
            ]
        ]
    )
    assert torch.allclose(scores, expected)


def test_loss_gradient_alignment_reports_component_norms_and_conflict():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    generation_component = parameter.square().sum()
    rank_loss = -3.0 * parameter.sum()
    measured = loss_gradient_alignment(generation_component, rank_loss, [parameter])
    assert math.isclose(measured["generation_component_grad_norm"], 4.0)
    assert math.isclose(measured["unweighted_rank_grad_norm"], 3.0)
    assert math.isclose(measured["gradient_cosine"], -1.0)


def test_prefix_margin_mse_uses_all_active_prefixes_and_negatives():
    scores = torch.tensor(
        [[[3.0, 5.0], [1.0, 2.0], [2.0, 4.0]]], requires_grad=True
    )
    loss = prefix_margin_mse_loss(scores, active_prefixes=2, target_margin=1.0)
    expected_margins = torch.tensor([2.0, 3.0, 1.0, 1.0])
    assert torch.allclose(loss, ((expected_margins - 1.0) ** 2).mean())
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_prefix_margin_mse_uses_ripor_concave_prefix_weights():
    scores = torch.zeros((1, 2, 4))
    loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=4,
        target_margin=1.0,
        margin_beta=0.5,
    )
    expected_alpha = torch.tensor([4 / 7, 6 / 7, 20 / 21, 1.0])
    assert torch.allclose(loss, expected_alpha.square().mean())


def test_prefix_margin_mse_accepts_signed_per_pair_teacher_margins():
    scores = torch.tensor([[[2.0, 4.0], [1.0, 2.0], [3.0, 5.0]]])
    teacher_margins = torch.tensor([[0.5, -0.25]])
    loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=2,
        target_margin=teacher_margins,
    )
    predicted_margins = torch.tensor([[1.0, 2.0], [-1.0, -1.0]])
    targets = torch.tensor([[0.5, 0.5], [-0.25, -0.25]])
    assert torch.allclose(loss, (predicted_margins - targets).square().mean())


def test_prefix_margin_mse_can_skip_teacher_margins_that_contradict_gold():
    scores = torch.tensor([[[2.0, 4.0], [1.0, 2.0], [3.0, 5.0]]])
    loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=2,
        target_margin=torch.tensor([[0.5, -0.25]]),
        positive_target_margins_only=True,
    )
    predicted_positive_pair = torch.tensor([1.0, 2.0])
    assert torch.allclose(loss, ((predicted_positive_pair - 0.5) ** 2).mean())


def test_prefix_margin_mse_can_mask_nonselected_expanded_gold_rows():
    scores = torch.tensor(
        [
            [[2.0, 4.0], [1.0, 2.0]],
            [[4.0, 8.0], [1.0, 2.0]],
        ]
    )
    loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=2,
        target_margin=torch.tensor([[0.5], [10.0]]),
        example_mask=torch.tensor([True, False]),
    )
    assert torch.allclose(loss, torch.tensor(((1.0 - 0.5) ** 2 + (2.0 - 0.5) ** 2) / 2))


def test_teacher_margins_use_positive_minus_each_negative_cosine_score():
    query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    skill_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]]
    )
    margins = teacher_margins_for_candidates(
        query_embeddings,
        skill_embeddings,
        source_positions=torch.tensor([1, 0]),
        selected_indices=torch.tensor([[1, 2, 0], [0, 2, 3]]),
        scale=2.0,
    )
    assert torch.allclose(margins, torch.tensor([[0.4, 2.0], [0.8, 4.0]]))


def test_teacher_best_gold_selects_one_labeled_skill_per_source_query():
    rows = [
        {"selected_skill_ids": ["a", "b"]},
        {"selected_skill_ids": ["b", "c"]},
    ]
    query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    skill_embeddings = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    selected = teacher_best_gold_indices_for_rows(
        rows,
        query_embeddings,
        skill_embeddings,
        {"a": 0, "b": 1, "c": 2},
        batch_size=2,
    )
    assert selected.tolist() == [0, 2]


def test_prefix_margin_mse_masks_pairs_before_their_paths_diverge():
    scores = torch.tensor([[[3.0, 5.0], [3.0, 2.0], [1.0, 4.0]]])
    codes = torch.tensor([[[0, 0], [0, 1], [1, 0]]])

    first_prefix_loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=1,
        target_margin=1.0,
        candidate_codes=codes,
    )
    assert torch.allclose(first_prefix_loss, torch.tensor(1.0))

    both_prefixes_loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=2,
        target_margin=1.0,
        candidate_codes=codes,
    )
    expected_margins = torch.tensor([3.0, 2.0, 1.0])
    assert torch.allclose(both_prefixes_loss, ((expected_margins - 1.0) ** 2).mean())


def test_prefix_margin_mse_can_select_noncontiguous_prefix_levels():
    scores = torch.tensor([[[0.0, 0.0, 0.0, 4.0], [1.0, 2.0, 3.0, 2.0]]])
    loss = prefix_margin_mse_loss(
        scores,
        active_prefixes=2,
        active_prefix_indices=[0, 3],
        target_margin=1.0,
    )
    predicted_margins = torch.tensor([-1.0, 2.0])
    assert torch.allclose(loss, ((predicted_margins - 1.0) ** 2).mean())


def test_progressive_prefix_indices_keep_full_sequence_anchor():
    assert progressive_prefix_indices(depth=4, epoch=1, full_sequence_warmup=True) == [3]
    assert progressive_prefix_indices(depth=4, epoch=2, full_sequence_warmup=True) == [0, 3]
    assert progressive_prefix_indices(depth=4, epoch=3, full_sequence_warmup=True) == [0, 1, 3]
    assert progressive_prefix_indices(depth=4, epoch=4, full_sequence_warmup=True) == [0, 1, 2, 3]


class _FakeTopKModel:
    def logits_for_next_type_from_encoder(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        target_type,
    ):
        if target_type == 0:
            return torch.tensor([[0.0, 1.0]]).repeat(decoder_input_ids.shape[0], 1)
        rows = []
        for previous_code in decoder_input_ids[:, -1].tolist():
            rows.append([2.0, 0.0] if previous_code == 1 else [0.0, 1.0])
        return torch.tensor(rows)


def test_topk_single_sid_search_uses_full_head_sequence_probability():
    semantic_ids = {"a": [0, 0], "b": [0, 1], "c": [1, 0]}
    children, _, prefix_counts = build_semantic_trie_children(semantic_ids)
    codes, scores = topk_single_sid_scores_from_encoder(
        _FakeTopKModel(),
        torch.zeros((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.long),
        children,
        prefix_counts,
        codebook_size=2,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
        top_k=2,
        decoder_batch_size=4,
        include_stop=False,
    )
    assert codes.tolist() == [[[1, 0], [0, 1]]]
    assert scores[0, 0] > scores[0, 1]


def test_topk_single_sid_search_supports_ripor_conditional_logits():
    semantic_ids = {"a": [0, 0], "b": [0, 1], "c": [1, 0]}
    children, _, prefix_counts = build_semantic_trie_children(semantic_ids)
    codes, scores = topk_single_sid_scores_from_encoder(
        _FakeTopKModel(),
        torch.zeros((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.long),
        children,
        prefix_counts,
        codebook_size=2,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
        top_k=2,
        decoder_batch_size=4,
        include_stop=False,
        score_mode="conditional_logit",
    )
    assert codes.tolist() == [[[1, 0], [0, 1]]]
    assert torch.equal(scores, torch.tensor([[3.0, 1.0]]))


def test_cached_gpu_trie_tables_preserve_exact_beam_output():
    semantic_ids = {"a": [0, 0], "b": [0, 1], "c": [1, 0]}
    children, _, prefix_counts = build_semantic_trie_children(semantic_ids)
    encoder = torch.zeros((2, 1, 1))
    mask = torch.ones((2, 1), dtype=torch.long)
    uncached = topk_single_sid_scores_from_encoder(
        _FakeTopKModel(),
        encoder,
        mask,
        children,
        prefix_counts,
        codebook_size=2,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
        top_k=2,
        decoder_batch_size=4,
        include_stop=False,
        score_mode="conditional_logit",
    )
    tables = build_trie_decode_tables(
        children, codebook_size=2, device=encoder.device
    )
    cached = topk_single_sid_scores_from_encoder(
        _FakeTopKModel(),
        encoder,
        mask,
        children,
        prefix_counts,
        codebook_size=2,
        depth=2,
        dedup_size=0,
        semantic_id_mode="learned_codes",
        top_k=2,
        decoder_batch_size=4,
        include_stop=False,
        score_mode="conditional_logit",
        trie_decode_tables=tables,
    )
    assert torch.equal(cached[0], uncached[0])
    assert torch.equal(cached[1], uncached[1])
