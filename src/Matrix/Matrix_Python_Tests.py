import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- MATRIX TEST SUITE ----- |\033[0m")

############################################
############## ADDITION TESTS ##############
############################################
class TestMatrix_Addition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: ADDITION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: ADDITION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # ADDITION DOUBLE TYPE TESTS
  def test_matrix_addition_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        C_np = A_np + B_np
        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  def test_in_place_matrix_addition_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        A_np += B_np
        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

  # ADDITION FLOAT TYPE TESTS
  def test_matrix_addition_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        C_np = A_np + B_np
        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  def test_in_place_matrix_addition_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        A_np += B_np
        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_float)

  # ADDITION INTEGER TYPE TESTS
  def test_matrix_addition_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        C_np = A_np + B_np
        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(C_pysycl[i, j], C_np[i, j])

  def test_in_place_matrix_addition_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        A_np += B_np
        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(A_pysycl[i, j], A_np[i, j])

############################################
############ SUBTRACTION TESTS #############
############################################
class TestMatrix_Subtraction(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: SUBTRACTION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: SUBTRACTION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SUBTRACTION DOUBLE TYPE TESTS
  def test_matrix_subtraction_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        C_np = A_np - B_np
        C_pysycl = A_pysycl - B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  def test_in_place_matrix_subtraction_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        A_np -= B_np
        A_pysycl -= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

  # SUBTRACTION FLOAT TYPE TESTS
  def test_matrix_subtraction_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        C_np = A_np - B_np
        C_pysycl = A_pysycl - B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  def test_in_place_matrix_subtraction_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        A_np -= B_np
        A_pysycl -= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_float)

  # SUBTRACTION INTEGER TYPE TESTS
  def test_matrix_subtraction_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        C_np = A_np - B_np
        C_pysycl = A_pysycl - B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(C_pysycl[i, j], C_np[i, j])

  def test_in_place_matrix_addition_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        A_np -= B_np
        A_pysycl -= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(A_pysycl[i, j], A_np[i, j])

############################################
########## MULTIPLICATION TESTS ############
############################################
class TestMatrix_Multiplication(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: MULTIPLICATION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: MULTIPLICATION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MULTIPLICATION DOUBLE TYPE TESTS
  def test_matrix_element_multiplication_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        C_np = A_np * B_np
        C_pysycl = A_pysycl * B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  def test_in_place_matrix_element_multiplication_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        A_np *= B_np
        A_pysycl *= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

  # MULTIPLICATION FLOAT TYPE TESTS
  def test_matrix_element_multiplication_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        C_np = A_np * B_np
        C_pysycl = A_pysycl * B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  def test_in_place_matrix_element_multiplication_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        A_np *= B_np
        A_pysycl *= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_float)

  # MULTIPLICATION INTEGER TYPE TESTS
  def test_matrix_element_multiplication_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        C_np = A_np * B_np
        C_pysycl = A_pysycl * B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(C_pysycl[i, j], C_np[i, j])

  def test_in_place_matrix_element_multiplication_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        A_np *= B_np
        A_pysycl *= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(A_pysycl[i, j], A_np[i, j])

############################################
############# DIVISION TESTS ###############
############################################
class TestMatrix_Division(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: DIVISION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: DIVISION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # DIVISION DOUBLE TYPE TESTS
  def test_matrix_division_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        C_np = A_np / B_np
        C_pysycl = A_pysycl / B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  def test_in_place_matrix_division_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        B_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.double)

        A_np /= B_np
        A_pysycl /= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

  # DIVISION FLOAT TYPE TESTS
  def test_matrix_division_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        C_np = A_np / B_np
        C_pysycl = A_pysycl / B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  def test_in_place_matrix_division_float(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float32)
        B_np = np.random.rand(M, N).astype(np.float32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.float)

        A_np *= B_np
        A_pysycl *= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_float)

  # DIVISION INTEGER TYPE TESTS
  def test_matrix_division_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        C_np = A_np // B_np
        C_pysycl = A_pysycl / B_pysycl
        C_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(C_pysycl[i, j], C_np[i, j])

  def test_in_place_matrix_division_int(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        B_np = np.random.randint(1, 100, (M, N), dtype=np.int32)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.matrix(B_np, device= self.device, dtype= pysycl.int)

        A_np //= B_np
        A_pysycl /= B_pysycl
        A_pysycl.mem_to_cpu()

        for i in range(M):
          for j in range(N):
            self.assertEqual(A_pysycl[i, j], A_np[i, j])

############################################
########## ROWS AND COLS TESTS #############
############################################
class TestMatrix_Rows_Cols(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: ROWS AND COLS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: ROWS AND COLS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SIZE TYPE TESTS
  def test_vector_size_double(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_pysycl = pysycl.matrix((M, N), device= self.device, dtype= pysycl.int)

        self.assertEqual(A_pysycl.num_rows(), M)
        self.assertEqual(A_pysycl.num_cols(), N)

############################################
######### MAX, MIN, SUM TESTS ##############
############################################
class TestMatrix_Reductions(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: MAX, MIN, SUM (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: MAX, MIN, SUM (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_double = 1e-12
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # MAX, MIN, SUM TESTS
  def test_reductions(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)

        max_pysycl = A_pysycl.max()
        min_pysycl = A_pysycl.min()
        sum_pysycl = A_pysycl.sum()

        max_np = A_np.max()
        min_np = A_np.min()
        sum_np = A_np.sum()

        self.assertAlmostEqual(max_pysycl, max_np, delta= self.tolerance_double)
        self.assertAlmostEqual(min_pysycl, min_np, delta= self.tolerance_double)
        self.assertAlmostEqual(sum_pysycl, sum_np, delta= self.tolerance_double)

############################################
############# TRANSPOSE TESTS ##############
############################################
class TestMatrix_Transpose(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mMATRIX TESTS: TRANSPOSE (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mMATRIX TESTS: TRANSPOSE (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # TRANSPOSE TESTS
  def test_tranpose(self):
    for M in [5, 15, 80]:
      for N in [4, 20, 35]:
        A_np = np.random.rand(M, N).astype(np.float64)
        A_pysycl = pysycl.matrix(A_np, device= self.device, dtype= pysycl.double)

        A_pysycl.transpose()
        A_np = A_np.T

        A_pysycl.mem_to_cpu()

        for i in range(N):
          for j in range(M):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

if __name__ == '__main__':
  unittest.main()