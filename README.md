# PONNI: Portable Online Neural Network Inferencing
### Efficient in-loop neural network inferencing made easy in C++

Author: Matt Norman, Oak Ridge National Laboratory, https://mrnorman.github.io

PONNI provides a convenient way to build an efficient, portable Neural Network inference model in C++ with minimal syntax and full disclosure of exactly how the model is running on an accelerator device. It is built on the [Yet Another Kernel Launcher (YAKL)](https://github.com/mrnorman/YAKL) portable C++ library and runs out of the box on Nvidia, AMD, and Intel GPUs as well as CPUs with or without OpenMP 3.5 threading.

