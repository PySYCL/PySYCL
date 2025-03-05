import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

# Creating a new device instance
my_gpu = pysycl.device(0, 0)

# create a numpy 1d array of random values
arr = np.random.randn(10)

tensor = pysycl.tensor(my_gpu, arr)