
import importlib
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Supported examples
examples = { 
    'mnist-0':   'examples.mnist.mnist0',
    'mnist-1':   'examples.mnist.mnist1',
    'mnist-2':   'examples.mnist.mnist2',
    'mnist-b0':  'examples.mnist.mnist0_imperative',
    'mnist-b1':  'examples.mnist.mnist1_imperative',
    'mnist-b2':  'examples.mnist.mnist2_imperative',
    'cifar-0':   'examples.cifar.basic_net',
    'cifar-1':   'examples.cifar.mlp_net',
    'cifar-2':   'examples.cifar.conv_net',
    'segment-0': 'examples.color.segmentation',
    'regress-0': 'examples.regression.linear_regression',
    'regress-1': 'examples.regression.non_linear_regression',
    'fft':       'examples.fft.fft',
    'data-aug-1':  'examples.data_augmentation.cifar',
    'transfer-learn': 'examples.transfer_learning.cifar',
    'transfer-learn2': 'examples.transfer_learning.cifar2',
}

if __name__ == '__main__':
    
    if(len(sys.argv) > 1):
        module = examples[sys.argv[1]]
        importlib.import_module(module)
    else:
        module = examples['mnist-0']
        importlib.import_module(module)