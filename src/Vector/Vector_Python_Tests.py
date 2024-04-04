import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- VECTOR TEST SUITE ----- |\033[0m")

############################################
############## ADDITION TESTS ##############
############################################
class TestVector_Addition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: ADDITION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: ADDITION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # ADDITION DOUBLE TYPE TESTS
  def test_vector_addition_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      C_np = A_np + B_np
      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_addition_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      A_np += B_np
      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # ADDITION FLOAT TYPE TESTS
  def test_vector_addition_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      C_np = A_np + B_np
      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_addition_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      A_np += B_np
      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # ADDITION INTEGER TYPE TESTS
  def test_vector_addition_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      C_np = A_np + B_np
      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_addition_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      A_np += B_np
      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
########### SUBTRACTION TESTS ##############
############################################
class TestVector_Subtraction(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: SUBTRACTION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: SUBTRACTION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SUBTRACTION DOUBLE TYPE TESTS
  def test_vector_subtraction_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      C_np = A_np - B_np
      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_subtraction_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      A_np -= B_np
      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # SUBTRACTION FLOAT TYPE TESTS
  def test_vector_subtraction_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      C_np = A_np - B_np
      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_subtraction_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      A_np -= B_np
      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # SUBTRACTION INTEGER TYPE TESTS
  def test_vector_subtraction_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      C_np = A_np - B_np
      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_subtraction_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      A_np -= B_np
      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
########## MULTIPLICATION TESTS ############
############################################
class TestVector_Multiplication(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: MULTIPLICATION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: MULTIPLICATION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MULTIPLICATION DOUBLE TYPE TESTS
  def test_vector_multiplication_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      C_np = A_np * B_np
      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_multiplication_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      A_np *= B_np
      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # MULTIPLICATION FLOAT TYPE TESTS
  def test_vector_multiplication_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      C_np = A_np * B_np
      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_multiplication_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      A_np *= B_np
      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # MULTIPLICATION INTEGER TYPE TESTS
  def test_vector_multiplication_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      C_np = A_np * B_np
      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_multiplication_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.int32)
      B_np = np.random.rand(N).astype(np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      A_np *= B_np
      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
############## DIVISION TESTS ##############
############################################
class TestVector_Dvision(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: DIVISION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: DIVISION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # DIVISION DOUBLE TYPE TESTS
  def test_vector_division_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      C_np = A_np / B_np
      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_division_double(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      B_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.double)

      A_np /= B_np
      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # DIVISION FLOAT TYPE TESTS
  def test_vector_division_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      C_np = A_np / B_np
      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_division_float(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float32)
      B_np = np.random.rand(N).astype(np.float32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.float)

      A_np /= B_np
      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # DIVISION INTEGER TYPE TESTS
  def test_vector_division_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.randint(1, 100, N, dtype=np.int32)
      B_np = np.random.randint(1, 100, N, dtype=np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      C_np = A_np // B_np
      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_division_int(self):
    for N in [10, 25, 83]:
      A_np = np.random.randint(1, 100, N, dtype=np.int32)
      B_np = np.random.randint(1, 100, N, dtype=np.int32)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.vector(B_np, device= self.device, dtype= pysycl.int)

      A_np //= B_np
      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
############## FILL TESTS ##################
############################################
class TestVector_Fill(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: Fill (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: FILL (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # FILL DOUBLE TYPE TESTS
  def test_vector_fill_double(self):
    for N in [10, 25, 83]:
      A_pysycl = pysycl.vector(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # FILL FLOAT TYPE TESTS
  def test_vector_division_float(self):
    for N in [10, 25, 83]:
      A_pysycl = pysycl.vector(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # FILL INTEGER TYPE TESTS
  def test_vector_division_int(self):
    for N in [10, 25, 83]:
      A_pysycl = pysycl.vector(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
############## SIZE TESTS ##################
############################################
class TestVector_Fill(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: SIZE (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: SIZE (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SIZE TYPE TESTS
  def test_vector_size_double(self):
    for N in [10, 25, 83]:
      A_pysycl = pysycl.vector(N, device= self.device, dtype= pysycl.double)
      self.assertEqual(A_pysycl.get_size(), N)

############################################
######### MAX, MIN, SUM TESTS ##############
############################################
class TestVector_Reductions(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mVECTOR TESTS: MAX, MIN, SUM (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mVECTOR TESTS: MAX, MIN, SUM (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_double = 1e-12
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MAX, MIN, SUM TESTS
  def test_reductions(self):
    for N in [10, 25, 83]:
      A_np = np.random.rand(N).astype(np.float64)
      A_pysycl = pysycl.vector(A_np, device= self.device, dtype= pysycl.double)

      max_pysycl = A_pysycl.max()
      min_pysycl = A_pysycl.min()
      sum_pysycl = A_pysycl.sum()

      max_np = A_np.max()
      min_np = A_np.min()
      sum_np = A_np.sum()

      self.assertAlmostEqual(max_pysycl, max_np, delta= self.tolerance_double)
      self.assertAlmostEqual(min_pysycl, min_np, delta= self.tolerance_double)
      self.assertAlmostEqual(sum_pysycl, sum_np, delta= self.tolerance_double)

if __name__ == '__main__':
  unittest.main()
