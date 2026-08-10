import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DropoutZeroCatalogProbeTest(unittest.TestCase):
    def test_wrapper_overrides_instantiated_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "probe.py"
            audit = root / "audit.json"
            observed = root / "observed.json"
            probe.write_text(
                "import json, sys, torch\n"
                "from pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps({'p': torch.nn.Dropout(0.9).p}))\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_dropout_zero_catalog_probe.py",
                    "--probe",
                    str(probe),
                    "--audit",
                    str(audit),
                    str(observed),
                ],
                check=True,
            )
            self.assertEqual(json.loads(observed.read_text(encoding="utf-8")), {"p": 0.0})
            payload = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(payload["dropout_module_count"], 1)
            self.assertEqual(payload["dropout_module_rates"], [0.0])


if __name__ == "__main__":
    unittest.main()
