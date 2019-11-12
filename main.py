
import importlib
import sys

# Supported examples
examples = { 
    'keras-mnist-0': 'examples.mnist.keras_mnist_0',
    'keras-mnist-1': 'examples.mnist.keras_mnist_1',
    'keras-mnist-2': 'examples.mnist.keras_mnist_2',
    
    'mnist-0':   'examples.mnist.basic_net',
    'mnist-1':   'examples.mnist.mlp_net',
    'mnist-2':   'examples.mnist.conv_net',
    'cifar-0':   'examples.cifar.basic_net',
    'cifar-1':   'examples.cifar.mlp_net',
    'cifar-2':   'examples.cifar.conv_net',
    'segment-0': 'examples.color.segmentation',
    'regress-0': 'examples.regress.linear_regression',
    'regress-1': 'examples.regress.non_linear_regression',
    'fft':       'examples.fft.fft'
}

if __name__ == '__main__':
    
    if(len(sys.argv) > 1):
        module = examples[sys.argv[1]]
        importlib.import_module(module)
    else:
        module = examples['mnist-0']
        importlib.import_module(module)