# About
PySYCL is an open-source Python interface for SYCL that enables Python applications to leverage SYCL-based functionalities for heterogeneous computing. PySYCL aims to abstract away the complexities of GPU programming and provide to Python users an easy to use numerical library that efficiently targets hardware accelerators. The benefits to PySYCL is both accessibility to the large python community and development towards seamless integration with popular Python libraries, such as numpy, matplotlib, and pytorch.

![pysycl_equals](https://github.com/OsmanAEG/PySYCL/assets/79581083/ba5a297e-0941-4034-bb34-1bf97b4c7e91)

# Official Website
The official PySYCL website serves as the central hub for all things PySYCL. Get the latest PySYCL news, updates, and community resources, documentation, tutorials, and examples.

Access the official website here: https://pysycl.github.io/PySYCL/

# User Guide
The PySYCL User Guide is intended for python users aiming to leverage PySYCL in their projects. It covers basic concepts and practical examples to get started.

Find the user-guide here: https://pysycl.github.io/PySYCL/user-guide.html

# Developer Guide
The PySYCL Developer Guide is intended for developers aiming to contribute to the PySYCL backend. It provided developers with a complete set of documentation for the projects backend.

Find the dev-guide here: https://pysycl.github.io/PySYCL/dev-guide.html

# Installation
Installation guides for various development enviornments will be made available here.

## Installing PySYCL on Ubuntu WSL2 with CUDA
![ubuntu_wsl_cuda](https://github.com/PySYCL/PySYCL/assets/79581083/d7de96b9-8554-4a0e-a485-9443304702a0)

This guide provides instructions for setting up PySYCL on Windows 11 with an Ubuntu WSL2 enviornment with CUDA. 

### Step 1: Installing WSL 2 on Windows 11

1. Open **Windows PowerShell** as an Admin.
2. Enter the command:
   ```bash
   wsl --install
   ```

3. **Reboot** your PC when prompted to complete WSL installation.
4. **Ubuntu setup** will appear. Follow the instructions to complete Ubuntu installation.

### Step 2: Installing CUDA for WSL 2

1. Update your NVIDIA drivers on Windows. Enter the following command to check your CUDA version compatibility.
   ```bash
   nvidia-smi
   ```
2. Visit the [CUDA Downloads WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local). Alternatively, visit [CUDA Archive](https://developer.nvidia.com/cuda-toolkit-archive) for archived CUDA installation instructions.
3. Follow the instructions on the website to download and install the latest version of CUDA. For CUDA 12.4, the installation instructions can be found below:

   ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
   ```

### Step 3: Install Dependencies

Install the necessary dependencies using the following commands:

```bash
sudo apt-get update
sudo apt-get install cmake git python3-pybind11 python3-dev -y
```

### Step 4: Installing SYCL for CUDA

1. Visit [Intel's guide to build SYCL for different GPUs](https://www.intel.com/content/www/us/en/developer/articles/technical/compiling-sycl-with-different-gpus.html) to find information regarding different SYCL installations.
2. Use the following commands to clone the SYCL repository, configure, and build the SYCL toolchain for CUDA:

   ```bash
    git clone https://github.com/intel/llvm.git -b sycl
    cd llvm
    python ./buildbot/configure.py --cuda -t release --cmake-gen "Unix Makefiles"
    cd build
    sudo make deploy-sycl-toolchain -j `nproc`
   ```

   > Note: `nproc` will use all available processors to build. Adjust the `-j` flag according to your system's capabilities.

   > Note: If you want to install sycl locally, edit the CMAKE_INSTALL_PREFIX to be the following:

   ```bash
   /usr/local/
   ```

### Step 5: Building PySYCL from Source

1. To build PySYCL from source, first clone the repo and setup the build directory
   ```bash
    git clone https://github.com/PySYCL/PySYCL.git
    cd PySYCL
    mkdir build
    cd build
    cmake ../src
   ```
2. Edit the CMAKE_CXX_COMPILER to be the location of your SYCL compatible clang++ compiler. If you installed it locally, set the following
   ```bash
    /usr/local/bin/clang++
   ```
3. Build the project
   ```bash
    make
   ```

# Inquiries
For any questions or further information, feel free to reach out to us via the following email addresses:

- **Founding Contributor:** For any questions specific to the founding contributor and organization owner, please contact Osman El-Ghotmi at the following address.

email: osman.elghotmi@gmail.com.

- **General Inquiries:** For general inquiries about the PySYCL project, please email us at the following address.

email: pysycl.official@gmail.com.

We welcome your feedback, questions, and contributions to the PySYCL community!
