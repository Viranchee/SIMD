#ifndef SIMD_VECTORNEON_CPP
#define SIMD_VECTORNEON_CPP
#include "kernels.h"
#include <arm_neon.h>
#include <cassert>
#include <cstdint>

// Use ARM vector instructions
using namespace std;

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