
'''
Downloads and extract CiFAR-10 data
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import os
import tarfile
import zipfile
import sys

try: 
	from urllib.request import urlretrieve # Python 3
except ImportError:
	from urllib import urlretrieve # Python 2

def _download_progress(count, block_size, total_size):
    percent_complete = float(count * block_size) / total_size
    msg = "\r- Data download progress: {0:.1%}".format(percent_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_and_extract():
	_dir = "./"
	data_dir = "./cifar-10-batches-py"

	if os.path.exists(data_dir):
		print("CIFAR-10 data already downloaded.")
		return
	else:
		url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
		filename = url.split('/')[-1]
		file_path = os.path.join(_dir, filename)
		zip_cifar_10 = file_path
		file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_download_progress)

		print()
		print("Download finished. Extracting files in " + data_dir + ".")
		if file_path.endswith(".zip"):
			zipfile.ZipFile(file=file_path, mode="r").extractall(_dir)
		elif file_path.endswith((".tar.gz", ".tgz")):
			tarfile.open(name=file_path, mode="r:gz").extractall(_dir)
		print("Done.")

		os.remove(zip_cifar_10)

if __name__ == '__main__':
	download_and_extract()
