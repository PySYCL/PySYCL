import sys
sys.path.insert(1, '../../build/')

import pysycl

# Creating a new device instance
my_gpu = pysycl.device(0, 0)

# Check my gpu name
print("My GPU name: " + my_gpu.name())

# Check my gpu vendor
print("My GPU vendor: " + my_gpu.vendor())