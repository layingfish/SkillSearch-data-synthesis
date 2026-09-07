import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_ROOT = PROJECT_ROOT / "repro/20260903_letter_opq_cooldown_calibration"
CANDIDATE_RUNNER = RUNNER_ROOT / "run_candidate.sh"
TOP_RUNNER = RUNNER_ROOT / "run_1node_4x4.sh"
DDP_RUNNER = PROJECT_ROOT / "repro/20260903_letter_opq_ceiling/run_ddp_command.sh"
REAL_DATA_ROOT = Path(
    "/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/"
    "skill_gr_full_catalog_task5_gpt4o_20260713/"
    "canonical_alias_core75_learned_d4_inputs"
)


def write_metrics(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "recall_aggregation": "macro",
                "overall": {f"@{k}": {"macro_recall": score} for k in (1, 5, 10, 50)},
            }
        ),
        encoding="utf-8",
    )


class LetterOpqCooldownRunnerTest(unittest.TestCase):
    def test_two_rank_gloo_calibration_gradient_is_synchronized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "ddp_calibration_smoke.py"
            result = root / "result.json"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    import torch

                    sys.path.insert(0, os.environ["LETTER_OPQ_SCRIPTS_ROOT"])
                    import skill_gr_opq_parallel as parallel
                    from distributed_utils import (
                        all_gather_object,
                        destroy_distributed,
                        global_normalized_loss,
                        initialize_distributed,
                        unwrap_model,
                        wrap_ddp,
                    )

                    class Toy(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.logits = torch.nn.Parameter(
                                torch.tensor([[4.0, 0.0], [0.0, 2.0]])
                            )
                            self.position_weight_logits = torch.nn.Parameter(torch.zeros(2))

                        def forward(self, unused, *, return_position_weights=False):
                            logits = self.logits.unsqueeze(0)
                            weights = torch.softmax(self.position_weight_logits, dim=0)
                            if return_position_weights:
                                return logits, weights
                            return logits

                    context = initialize_distributed(cpu=True)
                    try:
                        model = wrap_ddp(Toy(), context)
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                        logits, weights = model(None, return_position_weights=True)
                        loss_sum, count, _ = parallel.exact_catalog_nll_components(
                            logits,
                            torch.tensor([[0, 0], [1, 1]]),
                            torch.tensor([0]),
                            position_weights=weights,
                        )
                        loss = global_normalized_loss(loss_sum, count, context)
                        loss.backward()
                        grad = unwrap_model(model).position_weight_logits.grad.detach().clone()
                        if not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
                            raise SystemExit("calibration weight gradient is missing")
                        optimizer.step()
                        learned = unwrap_model(model).position_weight_logits.detach().tolist()
                        gathered = all_gather_object(learned, context)
                        if any(values != gathered[0] for values in gathered[1:]):
                            raise SystemExit("DDP calibration parameters diverged")
                        if context.is_main:
                            Path(os.environ["LETTER_OPQ_DDP_RESULT"]).write_text(
                                json.dumps({"world_size": context.world_size, "weights": learned})
                            )
                    finally:
                        destroy_distributed(context)
                    """
                ),
                encoding="utf-8",
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = os.environ.copy()
            env.update(
                {
                    "LETTER_OPQ_WORLD_SIZE": "2",
                    "LETTER_OPQ_MASTER_PORT": str(port),
                    "LETTER_OPQ_SCRIPTS_ROOT": str(PROJECT_ROOT / "scripts"),
                    "LETTER_OPQ_DDP_RESULT": str(result),
                    "RIPOR_PYTHON_BIN": sys.executable,
                }
            )
            env.pop("CUDA_VISIBLE_DEVICES", None)
            subprocess.run(
                ["bash", str(DDP_RUNNER), str(script)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["world_size"], 2)
            self.assertNotEqual(payload["weights"], [0.0, 0.0])

    def test_candidate_runner_freezes_query_only_cooldown_and_is_dev_only(self):
        source = CANDIDATE_RUNNER.read_text(encoding="utf-8")
        for fragment in (
            "--catalog-epochs 0",
            "--epochs 3",
            "--lr 1e-4",
            "--min-lr 1e-5",
            "--lr-scheduler cosine",
            "--max-grad-norm 1",
            "--epoch-offset 10",
            "--initial-checkpoint",
            "LETTER_OPQ_EVAL_SPLIT=dev_rows_001",
            '"official_evaluation": False',
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("rows_003", source)
        self.assertNotIn("official_core75", source)

    def test_top_runner_has_disjoint_4x4_candidates_and_delayed_official(self):
        source = TOP_RUNNER.read_text(encoding="utf-8")
        self.assertIn("cf20_rpg_cooldown_e3", source)
        self.assertIn("cf20_rpg_calibrated_e3", source)
        self.assertIn("rpg_cosine_residual 0 0 0,1,2,3 29801", source)
        self.assertIn("rpg_cosine_residual_calibrated 0.1 0.01 4,5,6,7 29802", source)
        selection_call = source.index('--output "$selection"')
        official_path = source.index('official_rows="$data_root/rows_003')
        self.assertLess(selection_call, official_path)
        self.assertEqual(source.count("LETTER_OPQ_EVAL_SPLIT=official_core75"), 1)
        self.assertNotIn("private_test1000", source)
        self.assertNotIn("rows_002.decoder_rows", source)

    def _fake_environment(self, root: Path, baseline_score: float):
        fake_candidate = root / "fake-candidate.sh"
        fake_eval = root / "fake-eval.sh"
        record = root / "events.tsv"
        fake_candidate.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'root="$LETTER_OPQ_VARIANT_EXPERIMENT"\n'
            'mkdir -p "$root/decoder/final" "$root/dev_exact"\n'
            'printf checkpoint > "$root/decoder/final/parallel_decoder.pt"\n'
            "score=0.80\n"
            '[[ "$LETTER_OPQ_VARIANT" == cf20_rpg_calibrated_e3 ]] && score=0.90\n'
            'python3 - "$root/dev_exact/ranked_metrics.json" "$score" <<\'PY\'\n'
            "import json, sys\n"
            "from pathlib import Path\n"
            "score=float(sys.argv[2])\n"
            "Path(sys.argv[1]).write_text(json.dumps({'recall_aggregation':'macro','overall':{f'@{k}':{'macro_recall':score} for k in (1,5,10,50)}}))\n"
            "PY\n"
            'printf manifest > "$root/dev_exact/eval_manifest.json"\n'
            "python3 - \"$root/completion.json\" <<'PY'\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "Path(sys.argv[1]).write_text(json.dumps({'status':'complete','official_evaluation':False}))\n"
            "PY\n"
            'printf \'candidate\\t%s\\t%s\\t%s\\n\' "$LETTER_OPQ_VARIANT" "$CUDA_VISIBLE_DEVICES" "$LETTER_OPQ_MASTER_PORT" >> "$LETTER_OPQ_TEST_RECORD"\n',
            encoding="utf-8",
        )
        fake_eval.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'mkdir -p "$LETTER_OPQ_EVAL_OUTPUT"\n'
            "python3 - \"$LETTER_OPQ_EVAL_OUTPUT/ranked_metrics.json\" <<'PY'\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "Path(sys.argv[1]).write_text(json.dumps({'recall_aggregation':'macro','overall':{f'@{k}':{'macro_recall':0.95} for k in (1,5,10,50)}}))\n"
            "PY\n"
            'printf manifest > "$LETTER_OPQ_EVAL_OUTPUT/eval_manifest.json"\n'
            'printf \'eval\\t%s\\t%s\\n\' "$LETTER_OPQ_EVAL_SPLIT" "$CUDA_VISIBLE_DEVICES" >> "$LETTER_OPQ_TEST_RECORD"\n',
            encoding="utf-8",
        )
        fake_candidate.chmod(0o755)
        fake_eval.chmod(0o755)

        baseline = root / "baseline"
        baseline_candidate = baseline / "candidates/semantic_cf20_rpg_e10"
        write_metrics(
            baseline_candidate / "dev_exact/ranked_metrics.json", baseline_score
        )
        checkpoint = baseline_candidate / "decoder/final/parallel_decoder.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("baseline checkpoint", encoding="utf-8")
        write_metrics(baseline / "official_winner_exact/ranked_metrics.json", 0.75)
        (baseline / "official_winner_exact/eval_manifest.json").write_text(
            "manifest", encoding="utf-8"
        )
        cf20 = root / "cf20"
        cf20.mkdir()
        (cf20 / "semantic_ids.jsonl").write_text("ids", encoding="utf-8")

        output = root / "output"
        env = os.environ.copy()
        env.update(
            {
                "RIPOR_EXPERIMENT": str(output),
                "RIPOR_PROJECT_ROOT": str(PROJECT_ROOT),
                "RIPOR_PYTHON_BIN": "python3",
                "LETTER_OPQ_RUN_TAG": "unit-test",
                "LETTER_OPQ_CANDIDATE_RUNNER": str(fake_candidate),
                "LETTER_OPQ_EVAL_RUNNER": str(fake_eval),
                "LETTER_OPQ_SELECTOR": str(
                    PROJECT_ROOT / "scripts/select_letter_opq_ceiling.py"
                ),
                "LETTER_OPQ_LOG_ROOT": str(root / "logs"),
                "LETTER_OPQ_TEST_RECORD": str(record),
                "LETTER_OPQ_DATA_ROOT": str(REAL_DATA_ROOT),
                "LETTER_OPQ_CF20_ROOT": str(cf20),
                "LETTER_OPQ_BASELINE_ROOT": str(baseline),
                "LETTER_OPQ_MODEL_ROOT": str(root / "models"),
            }
        )
        return output, record, env

    @unittest.skipUnless(
        (REAL_DATA_ROOT / "rows_003.skillsbench_core75.rows.jsonl").is_file(),
        "Official SHA fixture is not mounted",
    )
    def test_new_dev_winner_runs_exactly_one_official_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, record, env = self._fake_environment(root, baseline_score=0.70)
            subprocess.run(
                ["bash", str(TOP_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            events = record.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                len([line for line in events if line.startswith("candidate")]), 2
            )
            self.assertEqual(events[-1], "eval\tofficial_core75\t0,1,2,3")
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual(completion["winner"], "cf20_rpg_calibrated_e3")
            self.assertEqual(completion["official_source"], "new_winner_only_eval")

    def test_baseline_dev_winner_reuses_frozen_official_without_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, record, env = self._fake_environment(root, baseline_score=0.99)
            subprocess.run(
                ["bash", str(TOP_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            events = record.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(events), 2)
            self.assertTrue(all(line.startswith("candidate") for line in events))
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual(completion["winner"], "baseline_cf20_rpg_e10")
            self.assertEqual(completion["official_source"], "reused_frozen_phase12")


if __name__ == "__main__":
    unittest.main()
