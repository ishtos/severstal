import sys
sys.path.append('../../../')

import unittest
import numpy as np

from layers.loss_funcs.kaggle_metric import *

class KaggleMetricTest(unittest.TestCase):
    
    def setUp(self):
        self.prob0 = np.zeros(( 2, 256, 1600, 4 ))
        self.prob1 = np.ones((  2, 256, 1600, 4 ))
        self.truth = np.ones((  2, 256, 1600, 4 ))
    
    def test_do_kaggle_metric(self):
        self.assertAlmostEqual(np.round(np.mean(do_kaggle_metric(self.prob0, self.truth)), 2), 0.0)
        self.assertAlmostEqual(np.round(np.mean(do_kaggle_metric(self.prob1, self.truth)), 2), 1.0)


if __name__ == '__main__':
    unittest.main()