
import importlib

# Supported examples
examples = { 
    'mnist-0':   'examples.mnist.basic_net',
    'mnist-1':   'examples.mnist.conv_net',
    'mnist-2':   'examples.mnist.mlp_net',
    'cifar-0':   'examples.cifar.basic_net',
    'cifar-1':   'examples.cifar.conv_net',
    'cifar-2':   'examples.cifar.mlp_net',
    'segment-0': 'examples.color.segmentation',
    'regress-0': 'examples.regress.linear_regression',
    'regress-1': 'examples.regress.non_linear_regression',
    'fft':       'examples.fft.fft'
}

if __name__ == '__main__':
    
    module = examples['mnist-0']
    importlib.import_module(module)

