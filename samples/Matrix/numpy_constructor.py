import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

M = 10
N = 12

device = pysycl.device.get_device(0, 0)
A_np = np.random.rand(M, N).astype(np.float64)
A = pysycl.matrix(A_np, device= device, dtype= pysycl.double)

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))

for i in range(M):
  for j in range(N):
    print(f"A[{i}, {j}] = {A[i, j]}")
