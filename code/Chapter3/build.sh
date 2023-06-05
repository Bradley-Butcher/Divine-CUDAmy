export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build
cd build
cmake ..
make
./add_element
./add_vector
./add_matrix