#include <torch/extension.h>

// Simple example: Adds two tensors
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) { return a + b; }

// Bind the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_tensors", &add_tensors, "A function that adds two tensors");
}

int main() {
  //
  return 0;
}