import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../')

import unittest
import torch

from networks.unet_senet import *

class SEnetUnetTest(unittest.TestCase):

    def test_unet_se_resnext50_32x4d(self):
        model = SEnetUnet(feature_net='se_resnext50_32x4d', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))


    def test_unet_se_resnext101_32x4d(self):
        model = SEnetUnet(feature_net='se_resnext101_32x4d', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_se_resnet50(self):
        model = SEnetUnet(feature_net='se_resnet50', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_se_resnet101(self):
        model = SEnetUnet(feature_net='se_resnet101', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_se_resnet152(self):
        model = SEnetUnet(feature_net='se_resnet152', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_senet154(self):
        model = SEnetUnet(feature_net='senet154', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))




if __name__ == '__main__':
    unittest.main()