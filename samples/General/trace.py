import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)
N = 10

A = pysycl.matrix((N, N), device= device, dtype= pysycl.double)
A.fill(8.0)

trace = pysycl.trace(A)

print("Trace of matrix A")
print("----------------------------------------")
print(trace)