import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)

M = 500
N = 300

A = pysycl.matrix((M, N), device= device, dtype= pysycl.double)
B = pysycl.matrix((M, N), device= device, dtype= pysycl.double)

print("Fill A with 1.0 and B with 2.0")
print("Compute C = A - B")
A.fill(1.0)
B.fill(2.0)

C = A - B
C.mem_to_cpu()

print("C[30, 50] = " + str(C[30, 50]))

print("Now compute C -= A")
C -= A
C.mem_to_cpu()

print("C[30, 50] = " + str(C[30, 50]) + "\n")