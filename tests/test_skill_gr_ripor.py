import math
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import skill_gr_ripor as ripor  # noqa: E402


def test_cf_procrustes_fusion_preserves_cf_geometry_and_m0_norms():
    rng = np.random.default_rng(17)
    m0 = rng.normal(size=(12, 8)).astype(np.float32)
    cf = rng.normal(size=(12, 3)).astype(np.float32)
    fused, projection, audit = ripor.cf_procrustes_assignment_embeddings(
        m0, cf, cf_weight=0.5
    )
    cf_unit = ripor._l2_normalize_np_rows(cf)
    projected = cf_unit @ projection
    assert fused.shape == m0.shape
    assert projection.shape == (3, 8)
    assert np.allclose(projected @ projected.T, cf_unit @ cf_unit.T, atol=2e-5)
    assert np.allclose(
        np.linalg.norm(fused, axis=1), np.linalg.norm(m0, axis=1), atol=2e-5
    )
    assert audit["projection_row_orthogonality_max_abs_error"] < 2e-5


def test_three_way_procrustes_supports_high_dim_qwen_and_preserves_m0_norms():
    rng = np.random.default_rng(23)
    m0 = rng.normal(size=(24, 8)).astype(np.float32)
    qwen = rng.normal(size=(24, 12)).astype(np.float32)
    cf = rng.normal(size=(24, 3)).astype(np.float32)
    fused, qwen_projection, cf_projection, audit = (
        ripor.m0_qwen_cf_procrustes_assignment_embeddings(
            m0,
            qwen,
            cf,
            m0_weight=0.5,
            qwen_weight=0.3,
            cf_weight=0.2,
        )
    )
    assert fused.shape == m0.shape
    assert qwen_projection.shape == (12, 8)
    assert cf_projection.shape == (3, 8)
    assert np.allclose(qwen_projection.T @ qwen_projection, np.eye(8), atol=2e-5)
    assert np.allclose(cf_projection @ cf_projection.T, np.eye(3), atol=2e-5)
    assert np.allclose(
        np.linalg.norm(fused, axis=1), np.linalg.norm(m0, axis=1), atol=2e-5
    )
    assert audit["qwen_projection_orthogonality"] == "column"
    assert audit["cf_projection_orthogonality"] == "row"


def test_three_way_procrustes_rejects_weights_that_do_not_sum_to_one():
    values = np.ones((3, 2), dtype=np.float32)
    try:
        ripor.m0_qwen_cf_procrustes_assignment_embeddings(
            values,
            values,
            values,
            m0_weight=0.5,
            qwen_weight=0.3,
            cf_weight=0.3,
        )
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("invalid three-way assignment weights unexpectedly accepted")


def test_fixed_assignment_codebook_refit_matches_sequential_residual_means():
    embeddings = np.asarray(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
        dtype=np.float32,
    )
    codes = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    codebooks, unused = ripor.refit_residual_codebooks(
        embeddings, codes, codebook_size=3
    )
    assert codebooks.shape == (2, 3, 2)
    assert unused == [1, 1]
    assert np.allclose(codebooks[0, 0], [2.0, 0.0])
    assert np.allclose(codebooks[0, 1], [0.0, 3.0])
    residual = embeddings - codebooks[0, codes[:, 0]]
    assert np.allclose(codebooks[1, 0], residual[[0, 2]].mean(axis=0))
    assert np.allclose(codebooks[1, 1], residual[[1, 3]].mean(axis=0))
    assert np.isfinite(codebooks).all()


def test_test_selection_key_uses_declared_test_only_tie_break_order():
    def metrics(overall_r1, single_r1, multi_r1, multi_r50, overall_r10):
        return {
            "overall": {"@1": {"macro_recall": overall_r1}, "@10": {"macro_recall": overall_r10}},
            "single": {"@1": {"macro_recall": single_r1}},
            "multi": {
                "@1": {"macro_recall": multi_r1},
                "@50": {"macro_recall": multi_r50},
            },
        }

    first = ripor.test_selection_key(metrics(0.3, 0.5, 0.2, 0.8, 0.7), 1)
    assert ripor.test_selection_key(metrics(0.31, 0.0, 0.0, 0.0, 0.0), 5) > first
    assert ripor.test_selection_key(metrics(0.3, 0.6, 0.3, 0.0, 0.0), 5) > first
    assert ripor.test_selection_key(metrics(0.3, 0.5, 0.2, 0.81, 0.0), 5) > first
    assert ripor.test_selection_key(metrics(0.3, 0.5, 0.2, 0.8, 0.71), 5) > first
    assert ripor.test_selection_key(metrics(0.3, 0.5, 0.2, 0.8, 0.7), 1) > ripor.test_selection_key(
        metrics(0.3, 0.5, 0.2, 0.8, 0.7), 2
    )


def _reference_bm25_top_k(skill_texts, skill_ids, query, *, top_k, k1, b):
    counts = [Counter(ripor._bm25_tokenize(text)) for text in skill_texts]
    lengths = [sum(row.values()) for row in counts]
    avg_length = sum(lengths) / len(lengths)
    document_frequency = Counter()
    for row in counts:
        document_frequency.update(row.keys())
    scores = defaultdict(float)
    for term in set(ripor._bm25_tokenize(query)):
        frequency = document_frequency.get(term, 0)
        if frequency == 0:
            continue
        term_idf = math.log(
            1.0
            + (len(skill_texts) - frequency + 0.5)
            / (frequency + 0.5)
        )
        for index, row in enumerate(counts):
            term_frequency = row.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + k1 * (
                1.0 - b + b * lengths[index] / max(avg_length, 1e-12)
            )
            scores[index] += (
                term_idf * (term_frequency * (k1 + 1.0)) / denominator
            )
    ranked = sorted(scores, key=lambda index: (-scores[index], skill_ids[index]))[:top_k]
    used = set(ranked)
    for index in sorted(range(len(skill_ids)), key=lambda value: skill_ids[value]):
        if len(ranked) == top_k:
            break
        if index not in used:
            ranked.append(index)
            used.add(index)
    return ranked


def test_vectorized_bm25_matches_reference_including_score_and_zero_ties():
    skill_ids = ["skill-z", "skill-a", "skill-c", "skill-b", "skill-y"]
    skill_texts = [
        "alpha alpha beta",
        "alpha alpha beta",  # exact score tie with skill-z; skill-a wins
        "beta gamma gamma",
        "delta",
        "epsilon",
    ]
    k1, b = 1.2, 0.75
    postings, _ = ripor._build_bm25_postings(skill_texts, k1=k1, b=b)
    zero_order = np.asarray(
        sorted(range(len(skill_ids)), key=lambda index: skill_ids[index]),
        dtype=np.int32,
    )
    lexical_rank = np.empty(len(skill_ids), dtype=np.int32)
    lexical_rank[zero_order] = np.arange(len(skill_ids), dtype=np.int32)
    for query in ("alpha", "beta gamma", "unknown-token", "alpha delta"):
        for top_k in (1, 3, len(skill_ids)):
            actual = ripor._retrieve_bm25_top_k(
                query,
                postings,
                skill_ids,
                zero_order,
                lexical_rank,
                top_k=top_k,
            )
            expected = _reference_bm25_top_k(
                skill_texts,
                skill_ids,
                query,
                top_k=top_k,
                k1=k1,
                b=b,
            )
            assert actual == expected


def test_prefix_schedule_and_paper_alphas_match_equation():
    assert ripor.ripor_prefix_lengths(8) == [4, 8]
    assert ripor.ripor_prefix_lengths(32) == [4, 8, 16, 32]
    alphas_8 = ripor.ripor_prefix_alphas(8, beta=2.0)
    assert math.isclose(alphas_8[4], 2.0 / 3.0, abs_tol=1e-12)
    assert alphas_8[8] == 1.0
    alphas_32 = ripor.ripor_prefix_alphas(32, beta=2.0)
    expected = {
        4: (1.0 - 2.0 / 4) / (1.0 - 2.0 / 32),
        8: (1.0 - 2.0 / 8) / (1.0 - 2.0 / 32),
        16: (1.0 - 2.0 / 16) / (1.0 - 2.0 / 32),
        32: 1.0,
    }
    assert alphas_32 == expected


def test_invalid_depth_cannot_silently_create_homebrew_prefix_schedule():
    for depth in (3, 6, 12):
        try:
            ripor.ripor_prefix_lengths(depth)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid RIPOR depth unexpectedly accepted: {depth}")


def test_signed_margin_mse_matches_direct_reference_without_clipping():
    student_positive = torch.tensor([2.0, -1.0])
    student_negative = torch.tensor([0.5, 0.0])
    teacher_positive = torch.tensor([0.2, -0.5])
    teacher_negative = torch.tensor([-0.1, 0.4])
    alpha = 2.0 / 3.0
    actual = ripor.ripor_margin_mse(
        student_positive,
        student_negative,
        teacher_positive,
        teacher_negative,
        alpha=alpha,
    )
    expected = torch.mean(
        (
            (student_positive - student_negative)
            - alpha * (teacher_positive - teacher_negative)
        )
        ** 2
    )
    assert torch.equal(actual, expected)
    # The second teacher margin is negative and must remain in the objective.
    assert float(teacher_positive[1] - teacher_negative[1]) < 0.0


def test_query_balanced_margin_mse_averages_positives_then_queries():
    student_positive = torch.tensor([3.0, 2.0, 4.0], requires_grad=True)
    student_negative = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    teacher_positive = torch.tensor([1.0, 1.5, 2.0])
    teacher_negative = torch.tensor([0.0, 0.0, 0.5])
    weights = ripor._query_balanced_pair_weights([2, 1])
    actual = ripor.query_balanced_ripor_margin_mse(
        student_positive,
        student_negative,
        teacher_positive,
        teacher_negative,
        weights,
    )
    pair_losses = (
        (student_positive - student_negative)
        - (teacher_positive - teacher_negative)
    ).square()
    expected = 0.5 * pair_losses[:2].mean() + 0.5 * pair_losses[2]
    assert torch.equal(actual, expected)
    assert torch.allclose(weights, torch.tensor([0.25, 0.25, 0.5]))


def test_query_balanced_microbatch_gradient_matches_direct_query_mean():
    positive_full = torch.tensor([3.0, 2.0, 4.0], requires_grad=True)
    negative_full = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    teacher_positive = torch.tensor([1.0, 1.5, 2.0])
    teacher_negative = torch.tensor([0.0, 0.0, 0.5])
    weights = ripor._query_balanced_pair_weights([2, 1])
    full_loss = ripor.query_balanced_ripor_margin_mse(
        positive_full,
        negative_full,
        teacher_positive,
        teacher_negative,
        weights,
    )
    full_loss.backward()
    expected_positive_grad = positive_full.grad.clone()
    expected_negative_grad = negative_full.grad.clone()

    positive_micro = positive_full.detach().clone().requires_grad_(True)
    negative_micro = negative_full.detach().clone().requires_grad_(True)
    accumulated = 0.0
    for start, end in ((0, 1), (1, 3)):
        micro_loss = ripor.query_balanced_ripor_margin_mse(
            positive_micro[start:end],
            negative_micro[start:end],
            teacher_positive[start:end],
            teacher_negative[start:end],
            weights[start:end],
        )
        micro_loss.backward()
        accumulated += float(micro_loss.detach())
    assert math.isclose(accumulated, float(full_loss.detach()), abs_tol=1e-7)
    assert torch.allclose(positive_micro.grad, expected_positive_grad, atol=1e-7, rtol=0.0)
    assert torch.allclose(negative_micro.grad, expected_negative_grad, atol=1e-7, rtol=0.0)


def test_size_weighted_microbatch_gradient_matches_effective_batch_mean():
    positive_full = torch.tensor([1.0, 2.0, 4.0, -1.0, 0.5], requires_grad=True)
    negative_full = torch.tensor([0.0, 1.5, 2.0, 0.0, -0.5], requires_grad=True)
    teacher_positive = torch.tensor([0.8, 0.2, 0.4, -0.1, 0.9])
    teacher_negative = torch.tensor([0.1, 0.5, -0.2, 0.3, 0.1])
    full_loss = ripor.ripor_margin_mse(
        positive_full,
        negative_full,
        teacher_positive,
        teacher_negative,
    )
    full_loss.backward()
    expected_positive_grad = positive_full.grad.clone()
    expected_negative_grad = negative_full.grad.clone()

    positive_micro = positive_full.detach().clone().requires_grad_(True)
    negative_micro = negative_full.detach().clone().requires_grad_(True)
    accumulated_value = 0.0
    for start, end in ((0, 2), (2, 5)):
        micro_loss = ripor.ripor_margin_mse(
            positive_micro[start:end],
            negative_micro[start:end],
            teacher_positive[start:end],
            teacher_negative[start:end],
        )
        scaled = ripor._scale_microbatch_mean_loss(
            micro_loss,
            microbatch_size=end - start,
            effective_batch_size=5,
        )
        scaled.backward()
        accumulated_value += float(micro_loss.detach()) * (end - start) / 5
    assert math.isclose(accumulated_value, float(full_loss.detach()), abs_tol=1e-7)
    assert torch.allclose(positive_micro.grad, expected_positive_grad, atol=1e-7, rtol=0.0)
    assert torch.allclose(negative_micro.grad, expected_negative_grad, atol=1e-7, rtol=0.0)


def test_cached_token_padding_matches_right_padding_and_attention_mask():
    input_ids, attention_mask = ripor._pad_cached_token_ids(
        [torch.tensor([7, 8, 1]), torch.tensor([9, 1])],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert input_ids.tolist() == [[7, 8, 1], [9, 1, 0]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_progressive_loss_sums_current_and_all_previous_prefix_objectives():
    positive = torch.tensor([[3.0, 7.0], [1.0, 2.0]])
    negative = torch.tensor([[1.0, 4.0], [1.5, 1.0]])
    teacher_positive = torch.tensor([0.8, 0.2])
    teacher_negative = torch.tensor([0.1, 0.4])
    alphas = {4: 2.0 / 3.0, 8: 1.0}
    actual, components = ripor.ripor_progressive_loss(
        positive,
        negative,
        teacher_positive,
        teacher_negative,
        prefix_lengths=[4, 8],
        active_prefix_lengths=[4, 8],
        alphas=alphas,
    )
    expected_4 = ripor.ripor_margin_mse(
        positive[:, 0],
        negative[:, 0],
        teacher_positive,
        teacher_negative,
        alpha=2.0 / 3.0,
    )
    expected_8 = ripor.ripor_margin_mse(
        positive[:, 1],
        negative[:, 1],
        teacher_positive,
        teacher_negative,
        alpha=1.0,
    )
    assert torch.equal(components[4], expected_4)
    assert torch.equal(components[8], expected_8)
    assert torch.equal(actual, expected_4 + expected_8)


def test_rank_microbatch_gradient_matches_effective_batch_progressive_mean():
    positive_full = torch.tensor(
        [[3.0, 7.0], [1.0, 2.0], [0.5, 1.0], [4.0, 5.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    negative_full = torch.tensor(
        [[1.0, 4.0], [1.5, 1.0], [0.0, -0.5], [2.0, 3.0], [-2.0, -1.0]],
        requires_grad=True,
    )
    teacher_positive = torch.tensor([0.8, 0.2, 0.5, 0.9, -0.1])
    teacher_negative = torch.tensor([0.1, 0.4, -0.2, 0.3, 0.2])
    alphas = {4: 2.0 / 3.0, 8: 1.0}
    full_loss, full_components = ripor.ripor_progressive_loss(
        positive_full,
        negative_full,
        teacher_positive,
        teacher_negative,
        prefix_lengths=[4, 8],
        active_prefix_lengths=[4, 8],
        alphas=alphas,
    )
    full_loss.backward()
    expected_positive_grad = positive_full.grad.clone()
    expected_negative_grad = negative_full.grad.clone()

    positive_micro = positive_full.detach().clone().requires_grad_(True)
    negative_micro = negative_full.detach().clone().requires_grad_(True)
    accumulated_loss = 0.0
    accumulated_components = {4: 0.0, 8: 0.0}
    for start, end in ((0, 2), (2, 5)):
        micro_loss, micro_components = ripor.ripor_progressive_loss(
            positive_micro[start:end],
            negative_micro[start:end],
            teacher_positive[start:end],
            teacher_negative[start:end],
            prefix_lengths=[4, 8],
            active_prefix_lengths=[4, 8],
            alphas=alphas,
        )
        ripor._scale_microbatch_mean_loss(
            micro_loss,
            microbatch_size=end - start,
            effective_batch_size=5,
        ).backward()
        accumulated_loss += float(micro_loss.detach()) * (end - start) / 5
        for length, value in micro_components.items():
            accumulated_components[length] += float(value.detach()) * (end - start) / 5
    assert math.isclose(accumulated_loss, float(full_loss.detach()), abs_tol=1e-7)
    for length, value in full_components.items():
        assert math.isclose(
            accumulated_components[length], float(value.detach()), abs_tol=1e-7
        )
    assert torch.allclose(positive_micro.grad, expected_positive_grad, atol=1e-7, rtol=0.0)
    assert torch.allclose(negative_micro.grad, expected_negative_grad, atol=1e-7, rtol=0.0)


class _DummyTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        width = max(1, max(len(text.split()) for text in texts))
        return {
            "input_ids": torch.ones((len(texts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), width), dtype=torch.long),
        }


def test_m1_batch_contains_exactly_rq_codes_and_no_stop_target():
    semantic_ids = {"a": [0, 1, 2, 0], "b": [2, 1, 0, 2]}
    rows = [
        {"query": "first query", "selected_skill_ids": ["a"]},
        {"query": "second query", "selected_skill_ids": ["b"]},
    ]
    batch = ripor.ripor_code_only_batch(
        rows,
        _DummyTokenizer(),
        semantic_ids,
        codebook_size=3,
        depth=4,
        max_source_length=32,
        device=torch.device("cpu"),
    )
    assert tuple(batch["labels"].shape) == (2, 4)
    assert batch["labels"].tolist() == [semantic_ids["a"], semantic_ids["b"]]
    assert batch["target_types"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert tuple(batch["decoder_input_ids"].shape) == (2, 4)


class _OnePassFakeModel(nn.Module):
    def __init__(self, depth: int, vocab: int, hidden_dim: int):
        super().__init__()
        self.code_heads = nn.ModuleList(
            nn.Linear(hidden_dim, vocab, bias=False) for _ in range(depth)
        )
        self.calls = 0

    def _decoder_hidden_from_encoder(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        del encoder_attention_mask, decoder_attention_mask
        self.calls += 1
        batch = encoder_hidden_states.shape[0]
        length = decoder_input_ids.shape[1]
        base = encoder_hidden_states[:, 0, :]
        return torch.stack(
            [base + float(position + 1) for position in range(length)], dim=1
        ).reshape(batch, length, -1)


def test_conditional_logit_prefix_scores_match_direct_dot_products_in_one_forward():
    depth, vocab, hidden_dim = 4, 3, 2
    model = _OnePassFakeModel(depth, vocab, hidden_dim)
    with torch.no_grad():
        for level, head in enumerate(model.code_heads):
            head.weight.copy_(
                torch.tensor(
                    [
                        [1.0 + level, 0.5],
                        [-0.5, 2.0 + level],
                        [0.25, -1.0 - level],
                    ]
                )
            )
    encoder = torch.tensor([[[0.2, -0.3]], [[1.0, 0.5]]])
    mask = torch.ones((2, 1), dtype=torch.long)
    codes = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    actual = ripor.sequence_conditional_logit_prefix_scores(
        model,
        encoder,
        mask,
        codes,
        codebook_size=vocab,
        depth=depth,
    )
    direct_tokens = []
    for level, head in enumerate(model.code_heads):
        hidden = encoder[:, 0, :] + float(level + 1)
        direct_tokens.append(
            head(hidden).gather(1, codes[:, level : level + 1]).squeeze(1)
        )
    expected = torch.cumsum(torch.stack(direct_tokens, dim=1), dim=1)
    assert torch.equal(actual, expected)
    assert model.calls == 1


def test_active_prefix_scoring_does_not_evaluate_later_decoder_positions():
    model = _OnePassFakeModel(depth=8, vocab=3, hidden_dim=2)
    encoder = torch.zeros((1, 1, 2))
    scores = ripor.sequence_conditional_logit_prefix_scores(
        model,
        encoder,
        torch.ones((1, 1), dtype=torch.long),
        torch.zeros((1, 8), dtype=torch.long),
        codebook_size=3,
        depth=8,
        score_length=4,
    )
    assert tuple(scores.shape) == (1, 4)
    assert model.calls == 1


def test_prefix_candidate_aggregation_uses_max_teacher_score_and_stable_tie_break():
    codes = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    representatives, scores, prefixes = ripor.aggregate_prefix_candidate_units(
        [1, 0, 3, 2],
        [0.2, 0.9, 0.5, 0.5],
        codes,
        prefix_length=4,
    )
    assert prefixes == [(0, 0, 0, 0), (1, 0, 0, 0)]
    assert representatives == [0, 2]
    assert scores == [0.9, 0.5]


def test_ripor_trie_retains_full_sid_collisions_in_catalog_order():
    semantic_ids = {
        "skill-b": [0, 1],
        "skill-a": [0, 1],
        "skill-c": [1, 0],
    }
    children, sequence_to_skills, prefix_counts = (
        ripor.build_ripor_semantic_trie_groups(semantic_ids)
    )
    assert children[()] == (0, 1)
    assert children[(0,)] == (1,)
    assert sequence_to_skills[(0, 1)] == ("skill-b", "skill-a")
    assert sequence_to_skills[(1, 0)] == ("skill-c",)
    assert prefix_counts == [2, 2]


def test_ripor_semantic_id_loader_and_hash_allow_official_collisions(tmp_path):
    path = tmp_path / "semantic_ids.jsonl"
    rows = [
        {
            "skill_id": "skill-a",
            "semantic_id": [0, 1],
            "semantic_id_mode": ripor.SEMANTIC_ID_MODE_LEARNED,
        },
        {
            "skill_id": "skill-b",
            "semantic_id": [0, 1],
            "semantic_id_mode": ripor.SEMANTIC_ID_MODE_LEARNED,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    semantic_ids, mode = ripor._load_ripor_semantic_ids(path)
    assert semantic_ids == {"skill-a": [0, 1], "skill-b": [0, 1]}
    assert mode == ripor.SEMANTIC_ID_MODE_LEARNED
    assert ripor._ripor_semantic_ids_sha256(semantic_ids) == (
        ripor._ripor_semantic_ids_sha256(dict(reversed(list(semantic_ids.items()))))
    )


def test_official_sid_expansion_repeats_score_without_truncating_collisions():
    indices, scores = ripor.expand_sid_beam_to_skill_indices(
        [[1, 0], [0, 1]],
        [3.0, 2.0],
        {(0, 1): (0, 1), (1, 0): (2,)},
    )
    assert indices == [2, 0, 1]
    assert scores == [3.0, 2.0, 2.0]
    assert ripor._candidate_row_with_gold_after_sid_expansion(
        indices,
        [3],
        retrieved_sid_count=2,
    ) == [3, 2, 0, 1]


def test_multi_gt_rank_units_keep_one_pair_per_gt_and_exclude_every_gold_prefix():
    rows = [{"selected_skill_ids": ["a", "b"]}]
    skill_ids = ["a", "b", "c", "d"]
    codes = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    candidates = torch.tensor([[0, 1, 2, 3]])
    counts = torch.tensor([4])
    teacher_scores = torch.tensor([[0.8, 0.7, 0.9, 0.2]])
    unit_indices, unit_scores, unit_counts, pairs, negatives = (
        ripor._materialize_rank_units(
            rows,
            skill_ids,
            codes,
            candidates,
            counts,
            teacher_scores,
            active_max_prefix=4,
        )
    )
    assert int(unit_counts[0]) == 2
    assert pairs == [(0, 0), (0, 0)]  # one example per GT, shared positive prefix
    assert negatives == [[1]]
    assert int(unit_indices[0, 0]) == 2  # max teacher score within the gold prefix
    assert math.isclose(float(unit_scores[0, 0]), 0.9, abs_tol=1e-7)
    assert ripor._unique_positive_positions_by_row(pairs, 1) == [[0]]


def test_full_sid_collision_is_one_rank_unit_and_never_a_negative_for_its_gt():
    rows = [{"selected_skill_ids": ["a", "b"]}]
    skill_ids = ["a", "b", "c"]
    codes = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    unit_indices, unit_scores, unit_counts, pairs, negatives = (
        ripor._materialize_rank_units(
            rows,
            skill_ids,
            codes,
            torch.tensor([[0, 1, 2]]),
            torch.tensor([3]),
            torch.tensor([[0.8, 0.9, 0.2]]),
            active_max_prefix=8,
        )
    )
    assert int(unit_counts[0]) == 2
    assert pairs == [(0, 0), (0, 0)]
    assert negatives == [[1]]
    assert int(unit_indices[0, 0]) == 1
    assert math.isclose(float(unit_scores[0, 0]), 0.9, abs_tol=1e-7)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(d_model=hidden_dim)
        self.anchor = nn.Parameter(torch.arange(hidden_dim, dtype=torch.float32))


class _FakeStructuredModel(nn.Module):
    def __init__(self, depth: int, vocab: int, hidden_dim: int):
        super().__init__()
        self.backbone = _FakeBackbone(hidden_dim)
        self.sid_embedding = nn.Embedding(depth * vocab + 3, hidden_dim)
        self.code_heads = nn.ModuleList(
            nn.Linear(hidden_dim, vocab, bias=True) for _ in range(depth)
        )


def test_rq_centroids_initialize_input_and_output_without_rms_rescaling(monkeypatch):
    depth, vocab, hidden_dim = 4, 3, 2
    dense = SimpleNamespace(
        backbone=_FakeBackbone(hidden_dim),
        start_token_embed=torch.tensor([[[7.0, 8.0]]]),
    )
    codebooks = np.arange(depth * vocab * hidden_dim, dtype=np.float32).reshape(
        depth, vocab, hidden_dim
    )

    def fake_factory(*args, **kwargs):
        del args, kwargs
        return _FakeStructuredModel(depth, vocab, hidden_dim)

    monkeypatch.setattr(ripor, "make_structured_t5_sid_model", fake_factory)
    model = ripor.initialize_ripor_structured_model(
        dense, base_model="fake", codebooks=codebooks
    )
    expected = torch.from_numpy(codebooks)
    for level in range(depth):
        start = level * vocab
        assert torch.equal(model.sid_embedding.weight[start : start + vocab], expected[level])
        assert torch.equal(model.code_heads[level].weight, expected[level])
        assert torch.count_nonzero(model.code_heads[level].bias) == 0
        assert not model.code_heads[level].bias.requires_grad
    bos = ripor.structured_sid_bos_id(
        vocab, depth, 0, ripor.SEMANTIC_ID_MODE_LEARNED
    )
    assert torch.equal(model.sid_embedding.weight[bos], torch.tensor([7.0, 8.0]))
    assert model.sid_codebook_init_scale == 1.0


def test_tracked_decoder_start_asset_is_exact_official_t5_base_file():
    asset = ROOT / "repro" / "20260813_ripor_faithful" / "t5-base.npy.b64"
    tensor = ripor.load_official_decoder_start(asset, expected_d_model=768)
    assert tuple(tensor.shape) == (768,)
    assert tensor.dtype == torch.float32
    assert math.isclose(float(tensor[0]), -0.75390625, abs_tol=0.0)
