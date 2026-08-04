from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class OperatorSupportDocumentationTests(unittest.TestCase):
    def test_generated_operator_table_is_current(self) -> None:
        generator_root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [sys.executable, str(generator_root / "examples" / "generate_operator_support.py"), "--check"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
