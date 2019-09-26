import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../../')

import unittest
import torch

from networks.unet_resnet import *

class ResnetUnetTest(unittest.TestCase):

    def test_unet_resnet18(self):
        model = ResnetUnet(feature_net='resnet18', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))


    def test_unet_resnet34(self):
        model = ResnetUnet(feature_net='resnet34', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_resnet50(self):
        model = ResnetUnet(feature_net='resnet50', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_resnet101(self):
        model = ResnetUnet(feature_net='resnet101', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))

    def test_unet_resnet152(self):
        model = ResnetUnet(feature_net='resnet152', attention_type=None, pretrained_file=None)
        
        input = torch.randn(2, 3, 256, 1600)
        output = model.forward(input)
        
        self.assertEqual(output.shape, (2, 4, 256, 1600))



if __name__ == '__main__':
    unittest.main()