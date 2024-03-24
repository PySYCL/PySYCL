import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)

M = 5
N = 3

A = pysycl.matrix((M, N), device= device, dtype= pysycl.double)

for i in range(M):
  for j in range(N):
    A[i, j] = i*j

A.mem_to_gpu()

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))