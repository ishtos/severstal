from include import *
from utils import *

sz = 256
bs = 16
nfolds = 0
SEED = 43

TRAIN = os.path.join('..', 'input', 'train_images')

class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn)

def open_mask(fn):
    build_mask(fn)


if __name__ == '__main__':
    seed_everything(SEED)


