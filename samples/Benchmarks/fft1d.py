import sys
import time
import numpy as np
import torch  # Import PyTorch

sys.path.insert(1, '../../build/')
import pysycl

pysycl_device = pysycl.device.get_device(0, 0)
torch_device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(37)

N = 2**25 + 1

print("N = " + str(N))

# numpy vector setup
A_np = np.random.rand(N).astype(np.float64)

# pysycl and pytorch vector setup
A_pysycl = pysycl.vector(A_np, device=pysycl_device, dtype=pysycl.double)
A_torch  = torch.tensor(A_np, device=torch_device, dtype=torch.float64)

# pysycl timings
start_time_pysycl = time.time()
A_pysycl_fft = pysycl.fft1d(A_pysycl)
end_time_pysycl = time.time()
pysycl_time = end_time_pysycl - start_time_pysycl

print(f"PySYCL timings: {pysycl_time} seconds")

# numpy timings
start_time_np = time.time()
A_np_fft = np.fft.fft(A_np)
end_time_np = time.time()
np_time = end_time_np - start_time_np

print(f"NumPy timings: {np_time} seconds")

# pytorch timings
start_time_torch = time.time()
A_torch_fft = torch.fft.fft(A_torch)
end_time_torch = time.time()
torch_time = end_time_torch - start_time_torch

print(f"PyTorch timings: {torch_time} seconds")