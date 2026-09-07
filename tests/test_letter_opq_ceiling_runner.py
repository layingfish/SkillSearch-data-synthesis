import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_ROOT = PROJECT_ROOT / "repro/20260903_letter_opq_ceiling"
DDP_RUNNER = RUNNER_ROOT / "run_ddp_command.sh"
CANDIDATE_RUNNER = RUNNER_ROOT / "run_candidate.sh"
TOP_RUNNER = RUNNER_ROOT / "run_1node_4x4.sh"


class LetterOpqCeilingRunnerTest(unittest.TestCase):
    def test_direct_rank_launcher_clears_elastic_and_assigns_four_local_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_python = root / "fake-python"
            record = root / "ranks.tsv"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "
                '"$RANK" "$LOCAL_RANK" "$WORLD_SIZE" "$MASTER_PORT" '
                '"${TORCHELASTIC_RUN_ID-unset}" >> "$LETTER_OPQ_TEST_RECORD"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "RIPOR_PYTHON_BIN": str(fake_python),
                    "LETTER_OPQ_WORLD_SIZE": "4",
                    "LETTER_OPQ_MASTER_PORT": "29801",
                    "LETTER_OPQ_TEST_RECORD": str(record),
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "TORCHELASTIC_RUN_ID": "must-be-cleared",
                }
            )
            subprocess.run(
                ["bash", str(DDP_RUNNER), "ignored.py"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                sorted(record.read_text(encoding="utf-8").splitlines()),
                [
                    "0\t0\t4\t29801\tunset",
                    "1\t1\t4\t29801\tunset",
                    "2\t2\t4\t29801\tunset",
                    "3\t3\t4\t29801\tunset",
                ],
            )

    def test_candidate_runner_is_dev_only_with_frozen_training_budget(self):
        source = CANDIDATE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("--catalog-epochs 3", source)
        self.assertIn("--epochs 10", source)
        self.assertIn("--expected-global-batch-size 16", source)
        self.assertIn("--split-name dev_rows_001", source)
        self.assertIn('"official_evaluation": False', source)
        self.assertNotIn("rows_003", source)
        self.assertNotIn("official_core75", source)

    def test_top_runner_has_two_disjoint_waves_and_delays_single_official_eval(self):
        source = TOP_RUNNER.read_text(encoding="utf-8")
        for name in (
            "semantic_linear_e10",
            "semantic_rpg_e10",
            "semantic_cf20_linear_e10",
            "semantic_cf20_rpg_e10",
        ):
            self.assertIn(name, source)
        self.assertIn("0,1,2,3 29701", source)
        self.assertIn("4,5,6,7 29702", source)
        self.assertIn("0,1,2,3 29711", source)
        self.assertIn("4,5,6,7 29712", source)
        selection_call = source.index('--output "$selection"')
        official_path = source.index('official_rows="$data_root/rows_003')
        self.assertLess(selection_call, official_path)
        self.assertEqual(source.count("LETTER_OPQ_EVAL_SPLIT=official_core75"), 1)
        self.assertNotIn("private_test1000", source)
        self.assertNotIn("rows_002.decoder_rows", source)

    def test_top_runner_executes_dev_selection_before_one_official_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_candidate = root / "fake-candidate.sh"
            fake_eval = root / "fake-eval.sh"
            record = root / "events.tsv"
            fake_candidate.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'root="$LETTER_OPQ_VARIANT_EXPERIMENT"\n'
                'mkdir -p "$root/decoder/final" "$root/dev_exact"\n'
                'printf checkpoint > "$root/decoder/final/parallel_decoder.pt"\n'
                "score=0.60\n"
                '[[ "$LETTER_OPQ_VARIANT" == semantic_cf20_rpg_e10 ]] && score=0.90\n'
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
                "score=0.50\n"
                '[[ "$LETTER_OPQ_EVAL_SPLIT" == official_core75 ]] && score=0.80\n'
                'python3 - "$LETTER_OPQ_EVAL_OUTPUT/ranked_metrics.json" "$score" <<\'PY\'\n'
                "import json, sys\n"
                "from pathlib import Path\n"
                "score=float(sys.argv[2])\n"
                "Path(sys.argv[1]).write_text(json.dumps({'recall_aggregation':'macro','overall':{f'@{k}':{'macro_recall':score} for k in (1,5,10,50)}}))\n"
                "PY\n"
                'printf manifest > "$LETTER_OPQ_EVAL_OUTPUT/eval_manifest.json"\n'
                'printf \'eval\\t%s\\t%s\\n\' "$LETTER_OPQ_EVAL_SPLIT" "$CUDA_VISIBLE_DEVICES" >> "$LETTER_OPQ_TEST_RECORD"\n',
                encoding="utf-8",
            )
            fake_candidate.chmod(0o755)
            fake_eval.chmod(0o755)

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
                    "LETTER_OPQ_DATA_ROOT": "/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs",
                    "LETTER_OPQ_SEMANTIC_ROOT": str(
                        PROJECT_ROOT
                        / ".tmp/letter_opq_m64_semantic_c256_seed42_20260901"
                    ),
                    "LETTER_OPQ_CF20_ROOT": "/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/letter_opq_m64_semantic_cf20_c256_seed42_20260903",
                    "LETTER_OPQ_REFERENCE_ROOT": "/models_rw/SkillSearch-data-synthesis/letter_opq_parallel_runs/rudder_r2_parallel_mtp_4gpu_20260902_v3/opq_m64_parallel_t5base_seed20260626",
                }
            )
            subprocess.run(
                ["bash", str(TOP_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            events = record.read_text(encoding="utf-8").splitlines()
            candidate_events = sorted(
                event for event in events if event.startswith("candidate")
            )
            self.assertEqual(len(candidate_events), 4)
            self.assertEqual(events[-1], "eval\tofficial_core75\t0,1,2,3")
            self.assertEqual(events[-2], "eval\tdev_rows_001\t0,1,2,3")
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual(completion["winner"], "semantic_cf20_rpg_e10")
            self.assertEqual(
                completion["official_scope"], "official_core75_winner_only"
            )


if __name__ == "__main__":
    unittest.main()
