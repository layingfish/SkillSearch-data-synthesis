from __future__ import annotations

import unittest
from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_multigt_loss_catalog_20260805.sh"
)


class MultiGtCatalogLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def test_fresh_e60_catalog_has_isolated_outputs_and_fixed_protocol(self) -> None:
        self.assertIn('FRESH_CATALOG_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_fresh_e60"', self.script)
        self.assertIn('run_catalog_fresh_e60() {', self.script)
        self.assertIn('--catalog-epochs "${SKILLGR_EXTENDED_CATALOG_EPOCHS}"', self.script)
        self.assertIn(
            '--save-steps "$((SKILLGR_LONG_CATALOG_EVAL_INTERVAL * '
            'SKILLGR_CATALOG_STEPS_PER_EPOCH))"',
            self.script,
        )
        self.assertIn('catalog_fresh_e60) run_catalog_fresh_e60 ;;', self.script)

    def test_fresh_query_selection_uses_only_fresh_convergence_artifacts(self) -> None:
        self.assertIn('run_query_from_selected_fresh_catalog() {', self.script)
        self.assertIn('catalog_fresh_e60_convergence.json', self.script)
        self.assertIn('"$(fresh_catalog_checkpoint "${selected_epoch}")"', self.script)
        self.assertIn(
            'query_expanded_ce_fresh_e60_selected) '
            'run_query_from_selected_fresh_catalog ;;',
            self.script,
        )

    def test_non_converged_fresh_endpoint_has_an_explicit_query_stage(self) -> None:
        self.assertIn('run_query_from_fresh_e60_endpoint() {', self.script)
        self.assertIn(
            'endpoint query stage requires the observed non-converged catalog result',
            self.script,
        )
        self.assertIn(
            'query_expanded_ce_from_catalog_fresh_e60_endpoint',
            self.script,
        )
        self.assertIn(
            'query_expanded_ce_fresh_e60_endpoint) '
            'run_query_from_fresh_e60_endpoint ;;',
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
