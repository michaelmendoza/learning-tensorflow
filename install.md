

# Installing with Anaconda and pip

Take the following steps to install Tensorflow and Keras in an Anaconda environment:

1. Download and install Anaconda 
2. Create a conda environment named 'tf' with the following:
``` conda create -n tf python=3.5```
3. Activate the conda environment
``` activate tf ```
4. Install Tensorflow inside conda environment (GPU-version):
``` pip install --upgrade tensorflow-gpu```
5. Install Keras
``` pip install --upgrade keras```
6. Install Matplotlib, Cairos
``` 
conda install matplotlib
conda install pycairo
```
