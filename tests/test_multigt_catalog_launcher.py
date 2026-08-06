from __future__ import annotations

import unittest
from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_multigt_loss_catalog_20260805.sh"
)
CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "multigt_loss_catalog_20260805.env"
)


class MultiGtCatalogLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = LAUNCHER.read_text(encoding="utf-8")
        cls.config = CONFIG.read_text(encoding="utf-8")

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

    def test_fresh_e120_catalog_is_continuous_and_isolated(self) -> None:
        self.assertIn("SKILLGR_FRESH_LONG_CATALOG_EPOCHS=120", self.config)
        self.assertIn(
            'FRESH_E120_CATALOG_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_fresh_e120"',
            self.script,
        )
        self.assertIn('run_catalog_fresh_e120() {', self.script)
        self.assertIn(
            '--catalog-epochs "${SKILLGR_FRESH_LONG_CATALOG_EPOCHS}"',
            self.script,
        )
        self.assertIn(
            'catalog_fresh_e120_epoch60_parity.json',
            self.script,
        )
        self.assertIn(
            '"$(fresh_e120_catalog_checkpoint 60)/structured_decoder.pt"',
            self.script,
        )
        self.assertIn(
            'catalog_fresh_e120) run_catalog_fresh_e120 ;;',
            self.script,
        )

    def test_fresh_e120_query_uses_only_selected_converged_checkpoint(self) -> None:
        self.assertIn('run_query_from_selected_fresh_e120_catalog() {', self.script)
        self.assertIn('catalog_fresh_e120_convergence.json', self.script)
        self.assertIn(
            'fresh epoch-120 catalog has no checkpoint satisfying the convergence criterion',
            self.script,
        )
        self.assertIn(
            '"$(fresh_e120_catalog_checkpoint "${selected_epoch}")"',
            self.script,
        )
        self.assertIn(
            'query_expanded_ce_fresh_e120_selected) '
            'run_query_from_selected_fresh_e120_catalog ;;',
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
