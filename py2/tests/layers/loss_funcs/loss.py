import sys
sys.path.append('../../../')

import unittest
import numpy as np

import torch

from layers.loss_funcs.loss import *

class TestLoss(unittest.TestCase):
    
    def setUp(self):
        self.prob0 = torch.zeros( 2, 256, 1600, 4 )
        self.prob1 = torch.ones(  2, 256, 1600, 4 )
        self.truth = torch.ones(  2, 256, 1600, 4 )

    def test_dice_score(self):
        self.assertAlmostEqual(np.round(dice_score(self.prob0, self.truth).numpy(), 2), 0.0)
        self.assertAlmostEqual(np.round(dice_score(self.prob1, self.truth).numpy(), 2), 1.0)

    def test_symmetricLovaszLoss(self):
        slloss = SymmetricLovaszLoss()
        self.assertAlmostEqual(np.round(slloss(self.prob0, self.truth).numpy(), 2), 1.0)
        self.assertAlmostEqual(np.round(slloss(self.prob1, self.truth).numpy(), 2), 0.0)


if __name__ == '__main__':
    unittest.main()

