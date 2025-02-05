#include "SIMD/kernels.h"
#include <iostream>

int main() {
  std::cout << "Testing Vector Add" << std::endl;
  int v1[] = {1, 2, 3, 4, 5};
  int v2[] = {1, 2, 3, 4, 5};
  int *result = vectorAdd<Computation::Scalar, int>(v1, v2, 5);
  return 0;
}