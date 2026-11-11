"""Short GPU integration test for the experimental fracture seam."""
import json
import tempfile
import unittest
from pathlib import Path
from breakable_chip_probe import run


class BreakableChipTest(unittest.TestCase):
    def test_bond_holds_under_light_load_and_separates_under_heavy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            low = Path(directory) / 'low.json'
            high = Path(directory) / 'high.json'
            run(.0001, .03, low)
            run(.03, .03, high)
            a, b = (json.loads(p.read_text()) for p in (low, high))
            self.assertTrue(a['finite'] and b['finite'])
            self.assertFalse(a['broken'])
            self.assertTrue(b['broken'])
            self.assertGreater(b['final_gap_m'], .001)


if __name__ == '__main__':
    unittest.main()
