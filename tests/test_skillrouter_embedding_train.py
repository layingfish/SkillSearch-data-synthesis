from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is available in the remote training environment")

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from skillrouter_embedding_train import (  # noqa: E402
    backprop_tokenized_text_chunks,
    build_parser,
    encode_texts_train,
    encode_tokenized_text_chunks,
    ensure_gradcache_deterministic,
    text_batch_order,
    tokenize_text_chunks,
)


class TinyTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.calls += 1
        token_rows = [[(ord(char) % 31) + 1 for char in text][:max_length] for text in texts]
        width = max(len(row) for row in token_rows)
        input_ids = []
        attention_mask = []
        for row in token_rows:
            pad = width - len(row)
            input_ids.append([0] * pad + row)
            attention_mask.append([0] * pad + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = torch.tanh(self.projection(self.embedding(input_ids)))
        return types.SimpleNamespace(last_hidden_state=hidden)


class SkillRouterEmbeddingTrainTest(unittest.TestCase):
    def test_gradcache_matches_standard_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        standard_model = TinyEncoder()
        gradcache_model = copy.deepcopy(standard_model)
        queries = ["short query", "a substantially longer query"]
        docs = ["doc a", "longer document b", "doc c", "document d", "tiny", "the longest document here"]
        labels = torch.tensor([0, 3], dtype=torch.long)
        device = torch.device("cpu")

        standard_tokenizer = TinyTokenizer()
        query_reps = encode_texts_train(queries, standard_tokenizer, standard_model, device, 64, 2, True)
        doc_reps = encode_texts_train(docs, standard_tokenizer, standard_model, device, 64, 3, True)
        standard_loss = F.cross_entropy((query_reps @ doc_reps.T) / 0.2, labels)
        standard_loss.backward()

        gradcache_tokenizer = TinyTokenizer()
        query_chunks = tokenize_text_chunks(queries, gradcache_tokenizer, 64, 2, True)
        doc_chunks = tokenize_text_chunks(docs, gradcache_tokenizer, 64, 3, True)
        with torch.no_grad():
            cached_queries = encode_tokenized_text_chunks(
                query_chunks, len(queries), gradcache_model, device, detach=True
            ).requires_grad_(True)
            cached_docs = encode_tokenized_text_chunks(
                doc_chunks, len(docs), gradcache_model, device, detach=True
            ).requires_grad_(True)
        gradcache_loss = F.cross_entropy((cached_queries @ cached_docs.T) / 0.2, labels)
        gradcache_loss.backward()
        backprop_tokenized_text_chunks(query_chunks, cached_queries.grad, gradcache_model, device)
        backprop_tokenized_text_chunks(doc_chunks, cached_docs.grad, gradcache_model, device)

        self.assertTrue(torch.allclose(standard_loss, gradcache_loss, atol=1e-7, rtol=1e-7))
        self.assertEqual(gradcache_tokenizer.calls, 3)
        for standard_parameter, gradcache_parameter in zip(
            standard_model.parameters(), gradcache_model.parameters(), strict=True
        ):
            self.assertTrue(
                torch.allclose(standard_parameter.grad, gradcache_parameter.grad, atol=1e-6, rtol=1e-5)
            )

    def test_length_sort_restores_original_representation_order(self) -> None:
        torch.manual_seed(11)
        model = TinyEncoder()
        texts = ["medium", "x", "this is longest", "short"]
        device = torch.device("cpu")

        unsorted = encode_texts_train(texts, TinyTokenizer(), model, device, 64, 2, False)
        sorted_reps = encode_texts_train(texts, TinyTokenizer(), model, device, 64, 2, True)

        self.assertEqual(text_batch_order(texts, 2, True), [2, 0, 3, 1])
        self.assertTrue(torch.allclose(unsorted, sorted_reps, atol=1e-7, rtol=1e-7))

    def test_gradcache_rejects_positive_dropout(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))

        with self.assertRaisesRegex(ValueError, "positive dropout"):
            ensure_gradcache_deterministic(model)

    def test_train_defaults_keep_strict_recipe(self) -> None:
        args = build_parser().parse_args(["train", "--pairs", "pairs.jsonl", "--output-dir", "model"])

        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.grad_accum, 4)
        self.assertEqual(args.negatives_per_pair, 10)
        self.assertEqual(args.max_length, 2048)
        self.assertTrue(args.length_sort)
        self.assertFalse(args.fused_optimizer)


if __name__ == "__main__":
    unittest.main()
