export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build
cd build
cmake ..
make
./cu_hello_world