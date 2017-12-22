# To start tensorflow:
# Step 1: install the CUDA Toolkit 9.1 from https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
# Step 2: download the cuDNN library from https://developer.nvidia.com/rdp/cudnn-download
# Step 3: extract and put the cuDNN library at C:\\cudnn-9.1-windows10-x64-v7
# Step 4: add C:\\cudnn-9.1-windows10-x64-v7\cuda\bin to the $PATH environment variable
# Step 5: run the commands below
conda create --name tensorflow-gpu python=3.6.1
activate tensorflow-gpu
conda install jupyter
conda install scipy
conda install numpy
conda install tensorflow-gpu
conda install Flask
conda install gevent
conda install keras
conda install nltk
conda install h5py
conda install pillow