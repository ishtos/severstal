import sys
sys.path.append('../../')

import unittest
import numpy as np

from utils.mask_functions import *

class MaskFunctionsTest(unittest.TestCase):

    def setUp(self):
        self.rle = '0 4 8 2'
        self.mask = [
            [1, 1, 0, 0, 0, 0, 0, 0,],
            [1, 1, 0, 0, 0, 0, 0, 0,],
            [1, 0, 0, 0, 0, 0, 0, 0,],
            [1, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0, 0,],
        ]

    def test_run_length_encode(self):
        self.assertEqual(np.sum(run_length_decode(self.rle, height=8, width=8) - self.mask), 0)

    def run_length_decode(self):
        self.assertEqual(run_length_encode(self.mask), self.rle)


if __name__ == '__main__':
    unittest.main()
