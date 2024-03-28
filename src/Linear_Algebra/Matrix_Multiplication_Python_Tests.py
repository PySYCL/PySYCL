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
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MATMUL TEST DOUBLE
  def test_matmul_double(self):
    device = pysycl.device.get_device(0, 0)

    M = 65
    N = 85
    P = 25

    A_np = np.full((M, N), 2.0, dtype=np.float64)
    B_np = np.full((N, P), 4.0, dtype=np.float64)
    C_np = np.full((M, P), 0.0, dtype=np.float64)

    A_pysycl = pysycl.matrix((M, N), device= device, dtype = pysycl.double)
    B_pysycl = pysycl.matrix((N, P), device= device, dtype = pysycl.double)
    C_pysycl = pysycl.matrix((M, P), device= device, dtype = pysycl.double)

    A_pysycl.fill(2.0)
    B_pysycl.fill(4.0)
    C_pysycl.fill(0.0)

    C_np = np.matmul(A_np, B_np)
    pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

    C_pysycl.mem_to_cpu()

    for i in range(M):
      for j in range(P):
        self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  # MATMUL TEST FLOAT
  def test_matmul_float(self):
    device = pysycl.device.get_device(0, 0)

    M = 65
    N = 85
    P = 25

    A_np = np.full((M, N), 2.0, dtype=np.float32)
    B_np = np.full((N, P), 4.0, dtype=np.float32)
    C_np = np.full((M, P), 0.0, dtype=np.float32)

    A_pysycl = pysycl.matrix((M, N), device= device, dtype = pysycl.float)
    B_pysycl = pysycl.matrix((N, P), device= device, dtype = pysycl.float)
    C_pysycl = pysycl.matrix((M, P), device= device, dtype = pysycl.float)

    A_pysycl.fill(2.0)
    B_pysycl.fill(4.0)
    C_pysycl.fill(0.0)

    C_np = np.matmul(A_np, B_np)
    pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

    C_pysycl.mem_to_cpu()

    for i in range(M):
      for j in range(P):
        self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  # MATMUL TEST INT
  def test_matmul_int(self):
    device = pysycl.device.get_device(0, 0)

    M = 65
    N = 85
    P = 25

    A_np = np.full((M, N), 2, dtype=np.int32)
    B_np = np.full((N, P), 4, dtype=np.int32)
    C_np = np.full((M, P), 0, dtype=np.int32)

    A_pysycl = pysycl.matrix((M, N), device= device, dtype = pysycl.int)
    B_pysycl = pysycl.matrix((N, P), device= device, dtype = pysycl.int)
    C_pysycl = pysycl.matrix((M, P), device= device, dtype = pysycl.int)

    A_pysycl.fill(2)
    B_pysycl.fill(4)
    C_pysycl.fill(0)

    C_np = np.matmul(A_np, B_np)
    pysycl.linalg.matmul(A_pysycl, B_pysycl, C_pysycl, 32)

    C_pysycl.mem_to_cpu()

    for i in range(M):
      for j in range(P):
        self.assertEqual(C_pysycl[i, j], C_np[i, j])

if __name__ == '__main__':
  unittest.main()