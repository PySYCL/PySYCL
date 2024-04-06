import sys
import time
import numpy as np
import torch  # Import PyTorch

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.get_device(0, 0)
np.random.seed(37)

N = 2**3

A_np = np.random.rand(N).astype(np.float64)
A_pysycl = pysycl.vector(A_np, device=device, dtype=pysycl.double)

A_pysycl_fft = pysycl.fft1d(A_pysycl)
A_np_fft = np.fft.fft(A_np)
