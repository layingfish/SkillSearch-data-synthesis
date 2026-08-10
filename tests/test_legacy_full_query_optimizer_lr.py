import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_full_query_optimizer_lr import (
    ARM_SPECS,
    audit_workspace_catalog,
    ranking_key,
    selection_gate,
)
from scripts.legacy_full_query_optimizer_lr import (
    FullQueryObserver,
    checkpoint_state_summary,
    expected_eval_steps,
    pipeline_training_argv,
    validate_loss_history,
)
from scripts.legacy_layerwise_trace_probe import load_module


def metric_summary(
    *, overall: tuple[float, float, float], single_r1: float, multi_r1: float, dev: float
) -> dict:
    def subset(values: tuple[float, float, float]) -> dict:
        return {
            cutoff: {"recall": value}
            for cutoff, value in zip(("@1", "@10", "@50"), values)
        }

    return {
        "overall": subset(overall),
        "single": subset((single_r1, 0.5, 0.5)),
        "multi": subset((multi_r1, 0.8, 0.87)),
        "training": {"query_best_dev_loss": dev},
    }


class LegacyFullQueryMetadataTest(unittest.TestCase):
    def test_arm_matrix_uses_six_selection_arms_and_two_audits(self):
        self.assertEqual(len(ARM_SPECS), 8)
        self.assertEqual(sum(role == "selection" for _, role, _, _, _ in ARM_SPECS), 6)
        self.assertEqual(sum(role == "replica" for _, role, _, _, _ in ARM_SPECS), 1)
        self.assertEqual(sum(role == "anchor" for _, role, _, _, _ in ARM_SPECS), 1)

    def test_eval_cadence_includes_non_multiple_final_step(self):
        steps = expected_eval_steps(38805, 200)
        self.assertEqual(len(steps), 195)
        self.assertEqual(steps[-3:], [38600, 38800, 38805])

    def test_loss_history_requires_stage_specific_learning_rates(self):
        events = [
            {"stage": "catalog", "event": "train", "step": 50, "lr": 3e-4},
            {"stage": "query", "event": "train", "step": 50, "lr": 2e-4},
            {"stage": "query", "event": "eval", "step": 200, "dev_loss": 1.0},
            {"stage": "query", "event": "eval", "step": 205, "dev_loss": 0.9},
        ]
        summary = validate_loss_history(
            events, query_lr=2e-4, query_steps=205, eval_steps=200
        )
        self.assertEqual(summary["best_step"], 205)
        self.assertEqual(summary["query_logged_lrs"], [2e-4])

        events[1]["lr"] = 3e-4
        with self.assertRaisesRegex(ValueError, "query lr history mismatch"):
            validate_loss_history(
                events, query_lr=2e-4, query_steps=205, eval_steps=200
            )

    def test_training_argv_is_accepted_by_archived_parser(self):
        root = Path(__file__).resolve().parents[1]
        pipeline_path = (
            root
            / "repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py"
        )
        args = Namespace(
            pipeline=pipeline_path,
            rows=Path("rows.jsonl"),
            dev_rows=Path("dev.jsonl"),
            semantic_ids=Path("semantic_ids.json"),
            skills_jsonl=Path("skills.jsonl"),
            model=Path("model"),
            catalog_steps=2730,
            query_steps=38805,
            batch_size=16,
            max_source_length=2048,
            eval_steps=200,
            seed=20260626,
        )
        argv = pipeline_training_argv(args, Path("output/train"))
        pipeline = load_module(pipeline_path, "full_query_parser_contract_test")
        parsed = pipeline.build_arg_parser().parse_args(argv[1:])
        self.assertEqual(parsed.catalog_max_steps, 2730)
        self.assertEqual(parsed.max_steps, 38805)
        self.assertEqual(parsed.batch_size, 16)
        self.assertTrue(parsed.load_best_model_at_end)

    def test_checkpoint_policy_suppresses_last_and_strips_optimizer(self):
        observer = FullQueryObserver(
            optimizer_mode="carry",
            query_lr=3e-4,
            catalog_steps=2730,
            query_steps=38805,
            eval_steps=200,
            batch_size=16,
        )
        calls = []

        def saver(*args, **kwargs):
            calls.append((Path(args[0]).name, kwargs.get("optimizer")))

        controlled = observer.wrap_checkpoint_saver(saver)
        positional = [None] * 10
        positional[9] = {"best_step": 200}
        controlled(Path("best"), *positional[1:], optimizer=object())
        controlled(Path("last"), *positional[1:], optimizer=object())
        controlled(Path("final"), *positional[1:], optimizer=object())

        self.assertEqual(calls, [("best", None), ("final", None)])
        self.assertEqual(
            [row["name"] for row in observer.checkpoint_saves],
            ["best", "last", "final"],
        )
        self.assertTrue(observer.checkpoint_saves[1]["suppressed"])

    def test_selection_gate_and_tie_breaking_prioritize_top1(self):
        passing = metric_summary(
            overall=(0.28, 0.70, 0.76), single_r1=0.42, multi_r1=0.22, dev=0.8
        )
        self.assertTrue(selection_gate(passing)["passed"])

        higher_top1 = metric_summary(
            overall=(0.29, 0.70, 0.76), single_r1=0.42, multi_r1=0.22, dev=0.9
        )
        lower_top1 = metric_summary(
            overall=(0.28, 0.72, 0.77), single_r1=0.50, multi_r1=0.30, dev=0.7
        )
        self.assertGreater(ranking_key(higher_top1), ranking_key(lower_top1))

    def test_catalog_audit_detects_query_gradient_change(self):
        payload = {
            "runtime": {"torch": "fixed"},
            "precision": {"bf16": True},
            "configured_precision": {"bf16": True},
            "file_sha256": {"pipeline": "same"},
            "initial": {"model": "same"},
            "batches": {"catalog_step_1": "same", "query_step_1": "same"},
            "catalog_boundary": {"model": "same", "optimizer": "same"},
            "losses": {
                "catalog_step_1": 4.5,
                "catalog_step_50": 4.2,
                "catalog_step_2730": 3.0,
                "query_step_1": 4.1,
            },
            "query_step_1": {"gradient": {"sha256": "same"}},
            "dropout": {"requested_rate": 0.1},
        }
        self.assertTrue(audit_workspace_catalog([payload, payload])["passed"])
        changed = json.loads(json.dumps(payload))
        changed["query_step_1"]["gradient"]["sha256"] = "changed"
        result = audit_workspace_catalog([payload, changed])
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["query_step_1_gradient"])


@unittest.skipIf(torch is None, "torch is unavailable")
class LegacyFullQueryTensorTest(unittest.TestCase):
    def test_model_only_checkpoint_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.save(
                {"state_dict": {"weight": torch.tensor([1.0, 2.0])}},
                checkpoint_dir / "structured_decoder.pt",
            )
            summary = checkpoint_state_summary(checkpoint_dir)
        self.assertFalse(summary["optimizer_state_present"])
        self.assertIn("full", summary["model_state_sha256"])


if __name__ == "__main__":
    unittest.main()
