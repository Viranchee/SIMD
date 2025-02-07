#include "SIMD/kernels.h"
#include "SIMD/scalars.cpp"
#include "SIMD/vectorNeon.cpp"
#include <cstdint>

bool testVectorAdd() {
  int v1[] = {1, 2, 3, 4, 5};
  int v2[] = {1, 2, 3, 4, 5};
  int *result = vectorAdd<Computation::Scalar, int>(v1, v2, 5);
  for (int i = 0; i < 5; i++) {
    if (result[i] != v1[i] + v2[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  auto neon = new Neon();
  //
  return 0;
}