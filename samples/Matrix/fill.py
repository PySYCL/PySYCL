import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)
M = 12
N = 10

A = pysycl.matrix((M, N), device= device, dtype= pysycl.double)
B = pysycl.matrix((M, N), device= device, dtype= pysycl.double)

# get number of rows
print("Number of rows in A: " + str(A.num_rows()))

# get number of cols
print("Number of cols in A: " + str(A.num_cols()) + "\n")

# set element values
print("Set A[2, 4] = 6.0 and set B[2, 4] = 3.0")
print("----------------------------------------")
A[2, 4] = 6.0
B[2, 4] = 3.0

print("A[2, 4] = " + str(A[2, 4]))
print("B[2, 4] = " + str(B[2, 4]) + "\n")

# fill the matrix with a constant value
print("Fill A with 45.0")
print("----------------------------------------")
A.fill(45.0)
A.mem_to_cpu()
print("A[9, 7] = " + str(A[9, 7]))
