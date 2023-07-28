import unittest

from opencompass.datasets.humaneval import humaneval_postprocess

class TestHumaneval(unittest.TestCase):

    def setUp(self) -> None:
        self.gt = "    return x - int(x)"

    def test_vanilla(self):
        raw = self.gt
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_quote(self):
        lines = [
            "```python",
            "    return x - int(x)",
            "```",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_bare_quote(self):
        lines = [
            "```",
            "    return x - int(x)",
            "```",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_error_space_quote(self):
        lines = [
            "```",
            "  return x - int(x)",
            "```",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_import_1(self):
        lines = [
            "import numpy as np",
            "import math",
            "from typing import List",
            "",
            "def func(x):",
            "    return x - int(x)",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_import_2(self):
        lines = [
            "from typing import List",
            "import numpy as np",
            "import math",
            "def func(x):",
            "    return x - int(x)",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)


    def test_import_3(self):
        lines = [
            "import math",
            "",
            "",
            "def func(x):",
            "    return x - int(x)",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_comment(self):
        lines = [
            "def func(x: float) -> float:",
            "    '''",
            "    blah blah blah",
            "    blah blah blah",
            "    '''",
            "    return x - int(x)",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)

    def test_additional(self):
        lines = [
            "    return x - int(x)",
            "",
            "",
            "def func(x: float) -> float:",
            "    '''",
            "    blah blah blah",
            "    blah blah blah",
            "    '''",
            "    return x - int(x)",
        ]
        raw = "\n".join(lines)
        self.assertEqual(humaneval_postprocess(raw).rstrip(), self.gt)
