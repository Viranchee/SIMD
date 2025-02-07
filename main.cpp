#include "SIMD/kernels.h"
#include "SIMD/scalars.cpp"
#include "SIMD/tests.h"
#include "SIMD/vectorNeon.cpp"
#include <cstdint>

int main() {
  auto neon = new Neon();
  auto scalar = new Scalar();
  testVectorAdd<int8_t>(neon);
  return 0;
}