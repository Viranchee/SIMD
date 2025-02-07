#ifndef SIMD_VECTORNEON_CPP
#define SIMD_VECTORNEON_CPP
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

class Neon : public SIMD<int8_t> {
public:
  virtual int8_t *prefixSum(int8_t *v, int size) override {
    assert(size % 16 == 0);
    // Have a running sum
    int8_t *result = new int8_t[size];
    for (int i = 0; i < size; i += 16) {
      // load next 16 elements
      int8x16_t a = vld1q_s8(v + i);
      // Log based prefix sum for these 16 elements
      // select all
    }
    return result;
  }

  virtual int8_t *vectorAdd(int8_t *v1, int8_t *v2, int size) override {
    assert(size % 16 == 0);
    int8_t *result = new int8_t[size]; // Allocate memory for result
    for (int i = 0; i < size; i += 16) {
      int8x16_t a = vld1q_s8(v1 + i);
      int8x16_t b = vld1q_s8(v2 + i);
      int8x16_t sum = vaddq_s8(a, b);
      vst1q_s8(result + i, sum); // Store result
    }
    return result;
  }

  // Add all elements in the vector
  virtual int8_t vectorReduce(int8_t *v, int size) override {
    //
    return -1;
  };
  // Vector Max
  virtual int8_t vectorMax(int8_t *v, int size) override {
    //
    return -1;
  };
  // Vector Min
  virtual int8_t vectorMin(int8_t *v, int size) override {
    //
    return -1;
  };
  // Convolution 1D
  virtual int8_t *convolution_1d(int8_t *input, int iSize, int8_t *kernel,
                                 int kSize, int **oSize, int padding,
                                 int stride) override {
    //
    return nullptr;
  };
  // MatMul gemm
  virtual int8_t *matMul(int8_t *A, int M, int8_t *B, int N, int K) override {
    return nullptr;
  };
};
#endif // SIMD_VECTORNEON_CPP