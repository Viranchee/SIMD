#include "SIMD/kernels.h"
#include "SIMD/scalars.cpp"
#include "SIMD/tests.h"
#include "SIMD/vectorNeon.cpp"
#include <cstdint>
#include <iostream>

using namespace std;
int main() {
  auto neon = new Neon();
  auto scalar = new Scalar();
  // run prefixSum and print the values
  const int arrSize = 64;
  int8_t *v1 = new int8_t[arrSize];
  for (int i = 0; i < arrSize; ++i) {
    v1[i] = 1;
  }
  auto *result = neon->prefixSum(v1, arrSize);
  for (int i = 0; i < arrSize; ++i) {
    cout << (int)result[i] << " ";
  }

  testVectorAdd(scalar);
  testVectorAdd(neon);
  testVectorReduce(scalar);
  testVectorReduce(neon);
  testPrefixSum(scalar);
  testPrefixSum(neon);
  testVectorMin(scalar);
  testVectorMin(neon);
  testVectorMax(scalar);
  testVectorMax(neon);
  testConv1D(scalar, scalar);
  testConv2D(scalar, neon);
  testGEMM(scalar, neon);
  return 0;
}