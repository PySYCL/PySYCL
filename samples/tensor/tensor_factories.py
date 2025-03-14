import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

# Creating a new device instance
my_gpu = pysycl.device(0, 0)

##########################################################
# Constructing tensors with no dimensions
##########################################################

# create a base pysycl tensor of float64
tensor_f64 = pysycl.tensor(device= my_gpu, dtype= pysycl.float64)

# create a base pysycl tensor of float32
tensor_f32 = pysycl.tensor(device= my_gpu, dtype= pysycl.float32)

# create a base pysycl tensor of int16
tensor_i16 = pysycl.tensor(device= my_gpu, dtype= pysycl.int16)

##########################################################
# Constructing tensors with dimensions
##########################################################

# create a pysycl tensor of specific dimensions
tensor = pysycl.tensor(device= my_gpu, dims= (10, 20, 5), dtype= pysycl.float64)

##########################################################
# Constructing tensors with 1D numpy arrays
##########################################################

# create a pysycl tensor with a 1D numpy array

np_arr_1d = np.random.randn(10).astype(np.float64)
ps_arr_1d = pysycl.tensor(device= my_gpu, data= np_arr_1d, dtype= pysycl.float64)

##########################################################
# Constructing tensors with 2D numpy arrays
##########################################################

# create a pysycl tensor with a 2D numpy array

np_arr_2d = np.random.randn(10, 10).astype(np.float64)
ps_arr_2d = pysycl.tensor(device= my_gpu, data= np_arr_2d, dtype= pysycl.float64)