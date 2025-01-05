import sys
sys.path.insert(1, '../../build/')

import pysycl

# View all available devices
print("List of available devices:")
print("--------------------------")
pysycl.device.output_device_list()

print(" ")

print("Device indices")
print("--------------------------")
# Get a list of available device indices
my_devices = pysycl.device.get_device_list()
print(my_devices)