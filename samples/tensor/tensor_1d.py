import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

# Creating a new device instance
my_gpu = pysycl.device(0, 0)

# use numpy array to create a pysycl tensor of float64
arr_double    = np.random.randn(10).astype(np.float64)
tensor_double = pysycl.tensor_types.tensor_d(my_gpu, arr_double)

# use numpy array to create a pysycl tensor of int
arr_int    = np.random.randn(10).astype(np.int32)
tensor_int = pysycl.tensor_types.tensor_i(my_gpu, arr_int)

# use numpy array to create a pysycl tensor of float32
arr_float    = np.random.randn(10).astype(np.float32)
tensor_float = pysycl.tensor_types.tensor_f(my_gpu, arr_int)