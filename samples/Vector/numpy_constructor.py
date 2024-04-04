import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

N = 10

device = pysycl.device.get_device(0, 0)
A_np = np.random.rand(N).astype(np.float64)
A = pysycl.vector(A_np, device= device, dtype= pysycl.double)

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))

for i in range(N):
  print("A[" + str(i) + "] = " + str(A[i]))