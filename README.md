## About The Project

**shuce** is a fun deep learning library built in **Node.js**, inspired by **PyTorch**  
It supports **automatic differentiation** and uses [`GPU.js`](https://github.com/gpujs/gpu.js) to accelerate tensor operations

## Features
- `Tensor` class with `.add`, `.mul`, `.matmul`, `.mean`, and `.backward` methods
- Autodiff (like PyTorch’s `.backward()`)
- GPU acceleration for element-wise and matrix operations
- Dynamic computational graph with operation tracking