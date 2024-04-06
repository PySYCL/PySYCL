import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(1, '../../build/')
import pysycl

pysycl_device = pysycl.device.get_device(0, 0)
torch_device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(37)

Ns = [2**4, 2**8, 2**12, 2**14, 2**16, 2**18, 2**20, 2**24, 2**25, 2**26]
pysycl_times = []
numpy_times = []
pytorch_times = []

for N in Ns:
  print("N = " + str(N))
  A_np = np.random.rand(N).astype(np.float64)

  A_pysycl = pysycl.vector(A_np, device=pysycl_device, dtype=pysycl.double)
  A_torch  = torch.tensor(A_np, device=torch_device, dtype=torch.float64)

  # pysycl timings
  start_time_pysycl = time.time()
  A_pysycl_fft = pysycl.fft1d(A_pysycl)
  end_time_pysycl = time.time()
  pysycl_times.append(end_time_pysycl - start_time_pysycl)

  # numpy timings
  start_time_np = time.time()
  A_np_fft = np.fft.fft(A_np)
  end_time_np = time.time()
  numpy_times.append(end_time_np - start_time_np)

  # pytorch timings
  start_time_torch = time.time()
  A_torch_fft = torch.fft.fft(A_torch)
  end_time_torch = time.time()
  pytorch_times.append(end_time_torch - start_time_torch)

# plotting
plt.figure(figsize=(10, 6))
plt.plot(Ns, numpy_times,   marker='o', color='purple', label='NumPy')
plt.plot(Ns, pysycl_times,  marker='o', color='blue',   label='PySYCL')
plt.plot(Ns, pytorch_times, marker='o', color='orange', label='PyTorch')

plt.xlabel('N')
plt.ylabel('Time (s)')
plt.title('Performance Comparison')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig("fft_bench.png")
