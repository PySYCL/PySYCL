import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

# Creating a new device instance
my_gpu = pysycl.device(0, 0)

# create a base pysycl tensor of float64
tensor_f64 = pysycl.tensor(device= my_gpu, dtype= pysycl.float64)

# create a base pysycl tensor of float32
tensor_f32 = pysycl.tensor(device= my_gpu, dtype= pysycl.float32)

# create a base pysycl tensor of int16
tensor_i16 = pysycl.tensor(device= my_gpu, dtype= pysycl.int16)