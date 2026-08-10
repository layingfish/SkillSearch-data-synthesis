import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_optimizer_boundary_matrix import (
    catalog_invariant,
    query_update_distance,
)
from scripts.legacy_optimizer_boundary_probe import (
    EXPECTED_CATALOG_FIRST_BATCH_SHA256,
    EXPECTED_MANIFEST_ROW_COUNTS,
    EXPECTED_QUERY_FIRST_BATCH_SHA256,
    recursive_state_sha256,
    reset_optimizer_in_place,
    row_ids_payload,
)


class LegacyOptimizerBoundaryMetadataTest(unittest.TestCase):
    def test_manifest_row_counts_match_archived_training_manifest(self):
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "repro/20260714_letter_base_single_sid/archive/train/"
            "single_sid_decoder_train_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = {
            key: manifest.get(key) for key in EXPECTED_MANIFEST_ROW_COUNTS
        }
        self.assertEqual(observed, EXPECTED_MANIFEST_ROW_COUNTS)

    def test_archived_catalog_and_query_row_hashes(self):
        catalog_ids = [
            "catalog:other/services-layer",
            "catalog:other/go-oss-maintainer",
            "catalog:other/album-conceptualizer",
            "catalog:other/early-stopping-callback-jeremylongshore-claude-code-plugins-",
            "catalog:development/meta",
            "catalog:other/osc52-clipboard",
            "catalog:other/vendix-frontend-module-rzyfront-vendix-2",
            "catalog:other/typescript-api",
            "catalog:other/gpu-memory-analysis",
            "catalog:data/dbt-model-generator",
            "catalog:other/solid-nextjs-fusengine-agents",
            "catalog:development/recursive-improvement",
            "catalog:other/file-validator-cisco-ai-defense-skill-scanner",
            "catalog:other/terraform-dralgorhythm-claude-agentic-frame",
            "catalog:development/communication-protocols-pluginagentmarketpla-custom-plugin-server",
            "catalog:other/read-github",
        ]
        query_ids = [
            "pseudo:catalog_task:other/read-inspect-eval:4|other/read-inspect-eval",
            "pseudo:catalog_task:other/replit-incident-runbook:3|other/replit-incident-runbook",
            "15784|other/supabase-patterns",
            "pseudo:catalog_task:other/serena-takeaki0817-hibioru:3|other/serena-takeaki0817-hibioru",
            "pseudo:catalog_task:development/app-accessibility:4|development/app-accessibility",
            "12963|data/chroma-client",
            "pseudo:catalog_task:other/subagent-best-practices-opensourcesam-v2-heras-garden:3|other/subagent-best-practices-opensourcesam-v2-heras-garden",
            "pseudo:catalog_task:other/hook-development-aiskillstore-marketplace:4|other/hook-development-aiskillstore-marketplace",
            "pseudo:catalog_task:other/regulatory-deadline-tracker:1|other/regulatory-deadline-tracker",
            "10969|other/frontend-design-grupous-gpus-2",
            "19366|other/ui-ux-pro-max",
            "8626|other/animation-review",
            "8567|marketing/heading-tree",
            "pseudo:catalog_task:development/long-prompt-guide-rfxlamia-flow:3|development/long-prompt-guide-rfxlamia-flow",
            "14816|other/cell-segmentation-gptomics-bioskills",
            "pseudo:catalog_task:other/designer:4|other/designer",
        ]
        catalog_rows = [
            {
                "index": row_id,
                "positive_skill_id": row_id.removeprefix("catalog:"),
            }
            for row_id in catalog_ids
        ]
        query_rows = [
            {
                "index": row_id.rsplit("|", 1)[0],
                "positive_skill_id": row_id.rsplit("|", 1)[1],
            }
            for row_id in query_ids
        ]
        self.assertEqual(
            row_ids_payload(
                catalog_rows, include_positive_skill_id=False
            )["row_ids_sha256"],
            EXPECTED_CATALOG_FIRST_BATCH_SHA256,
        )
        self.assertEqual(
            row_ids_payload(
                query_rows, include_positive_skill_id=True
            )["row_ids_sha256"],
            EXPECTED_QUERY_FIRST_BATCH_SHA256,
        )
        self.assertNotEqual(
            row_ids_payload(
                catalog_rows, include_positive_skill_id=True
            )["row_ids_sha256"],
            EXPECTED_CATALOG_FIRST_BATCH_SHA256,
        )

    def test_catalog_invariant_covers_pre_query_state_and_gradient(self):
        payload = {
            "initial": {"model": "init"},
            "file_sha256": {"pipeline": "source"},
            "catalog_boundary": {
                "model_state_sha256": {"full": "model"},
                "trainable_state_sha256": "trainable",
                "optimizer_state_sha256": "optimizer",
                "optimizer_state_entries": 10,
                "optimizer_param_groups": [{"lr": 3e-4}],
                "cpu_rng_sha256": "cpu",
                "cuda_rng_sha256": "cuda",
            },
            "losses": {"catalog_step_2730": 4.0, "query_step_1": 3.9},
            "query_step_1": {"gradient": {"sha256": "gradient"}},
            "batches": {"query_step_1": {"row_ids_sha256": "batch"}},
        }
        matched = catalog_invariant([payload, payload])
        self.assertTrue(matched["passed"])

        changed = {**payload, "query_step_1": {"gradient": {"sha256": "changed"}}}
        mismatched = catalog_invariant([payload, changed])
        self.assertFalse(mismatched["passed"])
        self.assertFalse(mismatched["checks"]["query_step_1_gradient"])


@unittest.skipIf(torch is None, "torch is unavailable")
class LegacyOptimizerBoundaryTensorTest(unittest.TestCase):
    def test_recursive_state_hash_is_deterministic_and_sensitive(self):
        left = {"state": {0: {"step": torch.tensor(1.0)}}, "groups": [1, 2]}
        right = {"groups": [1, 2], "state": {0: {"step": torch.tensor(1.0)}}}
        changed = {"state": {0: {"step": torch.tensor(2.0)}}, "groups": [1, 2]}
        self.assertEqual(recursive_state_sha256(left), recursive_state_sha256(right))
        self.assertNotEqual(recursive_state_sha256(left), recursive_state_sha256(changed))

    def test_reset_optimizer_matches_fresh_adamw_state(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        model(torch.randn(4, 3)).sum().backward()
        optimizer.step()
        self.assertGreater(len(optimizer.state), 0)

        reset_optimizer_in_place(optimizer, 2e-4)
        fresh = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(
            recursive_state_sha256(optimizer.state_dict()),
            recursive_state_sha256(fresh.state_dict()),
        )

    def test_query_update_distance_uses_per_backend_catalog_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {}
            states = {
                "default": (torch.tensor([1.0]), torch.tensor([3.0])),
                "workspace_16": (torch.tensor([10.0]), torch.tensor([13.0])),
            }
            for path, (catalog, query) in states.items():
                arm = f"{path}_carry_lr2e4"
                arm_root = root / "arms" / arm
                arm_root.mkdir(parents=True)
                torch.save({"weight": catalog}, arm_root / "catalog.pt")
                torch.save({"weight": query}, arm_root / "query.pt")
                payloads[path] = {
                    "condition": {"arm": arm},
                    "catalog_boundary": {"trainable_state_artifact": "catalog.pt"},
                    "query_step_50": {"trainable_state_artifact": "query.pt"},
                }
            metrics = query_update_distance(
                root, payloads["default"], payloads["workspace_16"]
            )
            self.assertEqual(metrics["default_update_squared"], 4.0)
            self.assertEqual(metrics["workspace_update_squared"], 9.0)
            self.assertEqual(metrics["update_difference_squared"], 1.0)
            self.assertEqual(metrics["D_update"], 0.5)


if __name__ == "__main__":
    unittest.main()
