import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.legacy_ce_prefix_ranking import (
    FrozenC1Candidates,
    attach_prefix_ranking_forward,
    cumulative_prefix_scores,
    prefix_ranking_loss,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenC1CandidatesTest(unittest.TestCase):
    def test_manifest_is_hash_bound_and_validates_c1_divergence(self):
        semantic_ids = {
            "gold": (1, 2, 3, 4),
            "negative": (1, 7, 8, 9),
        }
        manifest_row = {
            "index": 0,
            "query_sha256": hashlib.sha256(b"query").hexdigest(),
            "gold_skill_ids": ["gold"],
            "gold_candidates": [
                {
                    "gold_skill_id": "gold",
                    "gold_semantic_id": [1, 2, 3, 4],
                    "selected_c1_negative": {
                        "skill_id": "negative",
                        "semantic_id": [1, 7, 8, 9],
                        "teacher_rank": 3,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            manifest_hash = sha256(manifest)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "counts": {
                            "rows": 1,
                            "gold_mentions": 1,
                            "selected_c1_gold_mentions": 1,
                        },
                        "inputs": {
                            "rows_sha256": "rows-hash",
                            "semantic_ids_sha256": "semantic-hash",
                        },
                        "outputs": {"candidate_manifest_sha256": manifest_hash},
                        "protocol": {
                            "semantic_depth": 4,
                            "candidate_limit": 50,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = FrozenC1Candidates.load(
                summary_path=summary,
                manifest_path=manifest,
                semantic_ids=semantic_ids,
                expected_summary_sha256=sha256(summary),
                expected_manifest_sha256=manifest_hash,
                expected_rows_sha256="rows-hash",
                expected_semantic_ids_sha256="semantic-hash",
            )
        self.assertEqual(candidates.source_rows, 1)
        self.assertEqual(candidates.gold_mentions, 1)
        self.assertEqual(candidates.selected_mentions, 1)


@unittest.skipIf(torch is None, "torch is unavailable")
class PrefixRankingTensorTest(unittest.TestCase):
    class TinyModel(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.depth = 4
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.code_heads = torch.nn.ModuleList(
                [torch.nn.Linear(3, 5, bias=False) for _ in range(self.depth)]
            )
            self.dropout = torch.nn.Dropout(p=0.25)

        def forward(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_types,
            labels,
        ):
            del attention_mask, decoder_input_ids, decoder_attention_mask, target_types
            features = self.dropout(input_ids.float())
            return ((features.mean() * self.weight) - labels.float().mean()).square()

        def encode_source(self, input_ids, attention_mask):
            del attention_mask
            return self.dropout(input_ids.float()).unsqueeze(1)

        def _decoder_hidden_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        ):
            del encoder_attention_mask, decoder_attention_mask
            base = encoder_hidden_states.expand(
                -1, decoder_input_ids.shape[1], -1
            )
            return self.dropout(
                base + decoder_input_ids.float().unsqueeze(-1) * 0.01
            )

        def logits_for_next_type_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type,
        ):
            hidden = self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
            )
            return self.code_heads[target_type](hidden[:, -1])

    def test_gate_zero_forward_gradient_and_update_match_base(self):
        torch.manual_seed(7)
        base = self.TinyModel()
        student = copy.deepcopy(base)
        attach_prefix_ranking_forward(
            student,
            weight=0.0,
            scope="cumulative",
            temperature=1.0,
            trace={},
        )
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "decoder_input_ids": torch.ones(2, 5, dtype=torch.long),
            "decoder_attention_mask": torch.ones(2, 5, dtype=torch.long),
            "target_types": torch.arange(5).repeat(2, 1),
            "labels": torch.tensor([[1, 2, 3, 4, 1], [2, 3, 4, 0, 1]]),
        }
        initial_rng = torch.random.get_rng_state()
        base_loss = base(**batch)
        base_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng)
        student_loss = student(**batch)
        student_rng = torch.random.get_rng_state()
        torch.testing.assert_close(student_loss, base_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(student_rng, base_rng, rtol=0.0, atol=0.0)
        base_loss.backward()
        student_loss.backward()
        for base_parameter, student_parameter in zip(
            base.parameters(), student.parameters()
        ):
            torch.testing.assert_close(
                student_parameter.grad,
                base_parameter.grad,
                rtol=0.0,
                atol=0.0,
            )
        base_optimizer = torch.optim.SGD(base.parameters(), lr=0.1)
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        base_optimizer.step()
        student_optimizer.step()
        for base_parameter, student_parameter in zip(
            base.parameters(), student.parameters()
        ):
            torch.testing.assert_close(
                student_parameter, base_parameter, rtol=0.0, atol=0.0
            )

    def test_cumulative_scores_use_each_requested_depth(self):
        torch.manual_seed(11)
        model = self.TinyModel()
        hidden = torch.randn(2, 4, 3)
        labels = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        observed = cumulative_prefix_scores(model, hidden, labels, (2, 3, 4))
        token_scores = []
        for level, head in enumerate(model.code_heads):
            log_probs = torch.log_softmax(head(hidden[:, level]).float(), dim=-1)
            token_scores.append(log_probs.gather(1, labels[:, level, None]).squeeze(1))
        cumulative = torch.stack(token_scores, dim=1).cumsum(dim=1)
        expected = cumulative[:, [1, 2, 3]]
        torch.testing.assert_close(observed, expected)

    def _assert_zero_contribution_invariant(self, device_name):
        device = torch.device(device_name)
        torch.manual_seed(17)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(17)
        base = self.TinyModel().to(device)
        student = copy.deepcopy(base)
        trace = {}
        attach_prefix_ranking_forward(
            student,
            weight=0.0,
            scope="cumulative",
            temperature=1.0,
            trace=trace,
            execute_when_zero=True,
        )
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "decoder_input_ids": torch.tensor(
                [[1, 1, 2, 3, 4], [1, 2, 3, 4, 0]]
            ),
            "decoder_attention_mask": torch.ones(2, 5, dtype=torch.long),
            "target_types": torch.arange(5).repeat(2, 1),
            "labels": torch.tensor([[1, 2, 3, 4, 1], [2, 3, 4, 0, 1]]),
        }
        auxiliary = {
            "prefix_rank_eligible_indices": torch.tensor([0]),
            "prefix_rank_negative_decoder_input_ids": torch.tensor(
                [[1, 1, 4, 3, 2]]
            ),
            "prefix_rank_negative_decoder_attention_mask": torch.ones(
                1, 5, dtype=torch.long
            ),
            "prefix_rank_negative_labels": torch.tensor([[1, 4, 3, 2, 1]]),
            "prefix_rank_query_step": 1,
        }
        batch = {key: value.to(device) for key, value in batch.items()}
        auxiliary = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in auxiliary.items()
        }
        if device.type == "cuda":
            def get_rng_state():
                return torch.cuda.get_rng_state(device)

            def set_rng_state(state):
                torch.cuda.set_rng_state(state, device)
        else:
            get_rng_state = torch.random.get_rng_state
            set_rng_state = torch.random.set_rng_state
        initial_rng = get_rng_state()

        base_optimizer = torch.optim.SGD(base.parameters(), lr=0.1)
        base_optimizer.zero_grad()
        base_loss = base(**batch)
        base_loss.backward()
        base_optimizer.step()
        base_rng_after_update = get_rng_state()
        base_next_loss = base(**batch)
        base_rng_after_next = get_rng_state()

        set_rng_state(initial_rng)
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        student_optimizer.zero_grad()
        student_loss = student(**batch, **auxiliary)
        student_loss.backward()
        student_optimizer.step()
        student_rng_after_update = get_rng_state()
        student_next_loss = student(**batch)
        student_rng_after_next = get_rng_state()

        torch.testing.assert_close(student_loss, base_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            student_rng_after_update, base_rng_after_update, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            student_next_loss, base_next_loss, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            student_rng_after_next, base_rng_after_next, rtol=0.0, atol=0.0
        )
        for base_parameter, student_parameter in zip(
            base.parameters(), student.parameters()
        ):
            torch.testing.assert_close(
                student_parameter, base_parameter, rtol=0.0, atol=0.0
            )

    def test_zero_contribution_aux_branch_preserves_rng_gradient_update_and_next_ce(self):
        self._assert_zero_contribution_invariant("cpu")

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA is unavailable"
    )
    def test_cuda_zero_contribution_aux_branch_preserves_main_rng(self):
        self._assert_zero_contribution_invariant("cuda")

    def test_loss_means_softplus_over_rows_and_depths(self):
        positive = torch.tensor([[2.0, 1.0, 0.0], [0.5, 0.0, -0.5]])
        negative = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 0.5]])
        observed = prefix_ranking_loss(positive, negative, temperature=1.0)
        expected = torch.nn.functional.softplus(negative - positive).mean()
        wrong_order = torch.nn.functional.softplus((negative - positive).mean())
        torch.testing.assert_close(observed, expected)
        self.assertNotEqual(float(observed), float(wrong_order))


if __name__ == "__main__":
    unittest.main()
