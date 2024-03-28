import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- TRACE TEST SUITE ----- |\033[0m")

############################################
############### TRACE TESTS ################
############################################
class Trace_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mTRACE TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mTRACE TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_double = 1e-14
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # TRACE TEST DOUBLE
  def test_trace_double(self):
    for N in [4, 20, 35]:
      A_pysycl = pysycl.matrix((N, N), device= self.device, dtype= pysycl.double)
      A_np = np.random.rand(N, N)

      for i in range(N):
        for j in range(N):
          A_pysycl[i, j] = A_np[i, j]

      A_pysycl.mem_to_gpu()

      trace_pysycl = pysycl.trace(A_pysycl)
      trace_np = np.trace(A_np)

      self.assertAlmostEqual(trace_pysycl, trace_np, delta= self.tolerance_double)

if __name__ == '__main__':
  unittest.main()