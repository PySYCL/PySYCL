import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- FFT 1D TEST SUITE ----- |\033[0m")

############################################
############## FFT 1D TESTS ################
############################################
class FFT1D_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mFFT 1D TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mFFT 1D TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance = 1e-5
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # FFT 1D Powers of 2
  def test_fft1d_power_of_2_double(self):
    device = pysycl.device.get_device(0, 0)

    np.random.seed(37)

    for N in [8, 16, 32, 64]:
      A_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device=device, dtype=pysycl.double)

      A_pysycl_fft = pysycl.fft.fft1d(A_pysycl)
      A_np_fft = np.fft.fft(A_np)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl_fft[i], A_np_fft[i], delta=self.tolerance)

      del(A_pysycl)

  # FFT 1D Non Powers of 2
  def test_fft1d_power_of_2_double(self):
    device = pysycl.device.get_device(0, 0)

    np.random.seed(37)

    for N in [7, 13, 27, 37]:
      A_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device=device, dtype=pysycl.double)

      A_pysycl_fft = pysycl.fft.fft1d(A_pysycl)
      A_np_fft = np.fft.fft(A_np)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl_fft[i], A_np_fft[i], delta=self.tolerance)

      del(A_pysycl)

if __name__ == '__main__':
  unittest.main()