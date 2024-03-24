import sys
sys.path.insert(1, '../../build/')

import pysycl
import numpy as np

rows = 10
cols = 12

device = pysycl.device.get_device(0, 0)
A = pysycl.matrix_type_float(np.random.rand(rows, cols).astype(np.float32), device)
A.mem_to_gpu()

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))

for i in range(rows):
  for j in range(cols):
    print(f"A[{i}, {j}] = {A[i, j]}")
