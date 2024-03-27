import sys
import time
import numpy as np
import torch

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.get_device(0, 0)

# Matrix dimensions
N = 6400
P = 6400
M = 6400

# numpy
A_np = np.full((M, N), 2.0, dtype=np.float32)
B_np = np.full((N, P), 4.0, dtype=np.float32)
C_np = np.full((M, P), 0.0, dtype=np.float32)

# pytorch
A_pt = torch.full((M, N), 2.0, dtype=torch.float32).to('cuda')
B_pt = torch.full((N, P), 4.0, dtype=torch.float32).to('cuda')
C_pt = torch.full((M, P), 0.0, dtype=torch.float32).to('cuda')

# pysycl
A_ps = pysycl.matrix((M, N), device= device, dtype = pysycl.float)
B_ps = pysycl.matrix((N, P), device= device, dtype = pysycl.float)
C_ps = pysycl.matrix((M, P), device= device, dtype = pysycl.float)

A_ps.fill(2.0)
B_ps.fill(4.0)

# numpy timings
start_time_np = time.time()
C_np = np.matmul(A_np, B_np)
end_time_np = time.time()
numpy_duration = end_time_np - start_time_np

# pysycl timings
start_time_ps = time.time()
pysycl.linalg.matmul(A_ps, B_ps, C_ps, 32)
end_time_ps = time.time()
pysycl_duration = end_time_ps - start_time_ps
C_ps.mem_to_cpu()

# torch timings
start_time_pt = time.time()
C_pt = torch.matmul(A_pt, B_pt)
end_time_pt = time.time()
pytorch_duration = end_time_pt - start_time_pt

print("numpy time: {:.2f} seconds".format(numpy_duration))
print("pysycl time: {:.2f} seconds".format(pysycl_duration))
print("pytorch time: {:.2f} seconds".format(pytorch_duration))

print("C_np[30, 50] = ", C_np[30, 50])
print("C_ps[30, 50] = ", C_ps[30, 50])
print("C_pt[30, 50] = ", C_pt[30, 50].item())