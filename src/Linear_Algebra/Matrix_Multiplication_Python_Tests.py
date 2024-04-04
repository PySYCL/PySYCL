import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- MATMUL TEST SUITE ----- |\033[0m")

############################################
############## MATMUL TESTS ################
############################################
class Matmul_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATMUL TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATMUL TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-5
    self.tolerance_double = 1e-14
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MATMUL TEST DOUBLE
  def test_matmul_double(self):
    device = pysycl.device.get_device(0, 0)

    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        for P in[8, 16, 44]:
          A_np = np.random.rand(M, N).astype(np.float64)
          B_np = np.random.rand(N, P).astype(np.float64)
          C_np = np.random.rand(M, P).astype(np.float64)

          A_pysycl = pysycl.matrix(A_np, device= device, dtype = pysycl.double)
          B_pysycl = pysycl.matrix(B_np, device= device, dtype = pysycl.double)
          C_pysycl = pysycl.matrix(C_np, device= device, dtype = pysycl.double)

          C_np = np.matmul(A_np, B_np)
          pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

          C_pysycl.mem_to_cpu()

          for i in range(M):
            for j in range(P):
              self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  # MATMUL TEST FLOAT
  def test_matmul_float(self):
    device = pysycl.device.get_device(0, 0)

    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        for P in[8, 16, 44]:
          A_np = np.random.rand(M, N).astype(np.float32)
          B_np = np.random.rand(N, P).astype(np.float32)
          C_np = np.random.rand(M, P).astype(np.float32)

          A_pysycl = pysycl.matrix(A_np, device= device, dtype = pysycl.float)
          B_pysycl = pysycl.matrix(B_np, device= device, dtype = pysycl.float)
          C_pysycl = pysycl.matrix(C_np, device= device, dtype = pysycl.float)

          C_np = np.matmul(A_np, B_np)
          pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

          C_pysycl.mem_to_cpu()

          for i in range(M):
            for j in range(P):
              self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  # MATMUL TEST INT
  def test_matmul_int(self):
    device = pysycl.device.get_device(0, 0)

    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        for P in[8, 16, 44]:
          A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
          B_np = np.random.randint(1, 100, (N, P), dtype=np.int32)
          C_np = np.random.randint(1, 100, (M, P), dtype=np.int32)

          A_pysycl = pysycl.matrix(A_np, device= device, dtype = pysycl.int)
          B_pysycl = pysycl.matrix(B_np, device= device, dtype = pysycl.int)
          C_pysycl = pysycl.matrix(C_np, device= device, dtype = pysycl.int)

          C_np = np.matmul(A_np, B_np)
          pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

          C_pysycl.mem_to_cpu()

          for i in range(M):
            for j in range(P):
              self.assertEqual(C_pysycl[i, j], C_np[i, j])

if __name__ == '__main__':
  unittest.main()