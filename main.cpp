#include "SIMD/kernels.h"
#include "SIMD/scalars.cpp"
#include "SIMD/tests.h"
#include "SIMD/vectorNeon.cpp"
#include <cstdint>

int main() {
  auto neon = new Neon();
  auto scalar = new Scalar();
  testVectorAdd(scalar);
  testVectorAdd(neon);
  testVectorReduce(scalar);
  testVectorReduce(neon);
  testPrefixSum(scalar);
  testPrefixSum(neon);
  return 0;
}