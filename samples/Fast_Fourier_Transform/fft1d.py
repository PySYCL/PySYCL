import sys
import time
import numpy as np

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.get_device(0, 0)

np.random.seed(37)

N = 16

A_np = np.random.rand(N).astype(np.float64)
A_pysycl = pysycl.vector(A_np, device = device, dtype= pysycl.double)

# pysycl timings
start_time_pysycl = time.time()
A_pysycl_fft = pysycl.fft1d(A_pysycl)
end_time_pysycl = time.time()
pysycl_time = end_time_pysycl - start_time_pysycl

# numpy timings
start_time_np = time.time()
A_np_fft = np.fft.fft(A_np)
end_time_np = time.time()
np_time = end_time_np - start_time_np

print(A_np_fft)

for i in A_pysycl_fft:
  print(i)

