#include "kernels.h"
#include <arm_neon.h>
#include <cassert>
#include <cstdint>

// Use ARM vector instructions
using namespace std;

template <>
int8_t *vectorAdd<Computation::NEON, int8_t>(int8_t *v1, int8_t *v2, int size) {
  int8_t *result = new int8_t[size];
  int8x8_t a, b, c;
  for (int i = 0; i < size; i += 8) {
    a = vld1_s8(v1 + i);
    b = vld1_s8(v2 + i);
    c = vadd_s8(a, b);
    vst1_s8(result + i, c);
  }
  return result;
}

template <>
std::vector<int8_t>
vectoredAdd<Computation::NEON, int8_t>(std::vector<int8_t> v1,
                                       std::vector<int8_t> v2) {
  //
  assert(v1.size() == v2.size());
  std::vector<int8_t> result(v1.size());
  int8x8_t a, b, c;
  for (size_t i = 0; i < v1.size(); i += 8) {
    a = vld1_s8(v1.data() + i);
    b = vld1_s8(v2.data() + i);
    c = vadd_s8(a, b);
    vst1_s8(result.data() + i, c);
  }

  return result;
}

template <Computation C, typename T> T prefixSum(T *v, int size);

// Write a Computation::Scalar implementation, T is flexible
template <> int prefixSum<Computation::Scalar, int>(int *v, int size) {
  int sum = 0;
  for (int i = 0; i < size; i++) {
    sum += v[i];
  }
  return sum;
}