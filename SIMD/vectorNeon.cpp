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
    int8_t runningSum = 0;
    for (int i = 0; i < size; i += 16) {
      // load next 16 elements
      int8x16_t zeroVec = vdupq_n_s8(0);
      int8x16_t data = vld1q_s8(v + i);
      // Log based prefix sum for these 16 elements
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 1));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 2));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 4));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 8));

      // broadcast the running sum
      data = vaddq_s8(data, vdupq_n_s8(runningSum));
      runningSum = vgetq_lane_s8(data, 15);
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
    assert(size % 16 == 0);
    int8x16_t sumV = vdupq_n_s8(0);
    for (int i = 0; i < size; i += 16) {
      int8x16_t data = vld1q_s8(v + i);
      sumV = vaddq_s8(sumV, data);
    }
    int8_t sum = 0;
    const int16x8_t sumV16 = vpaddlq_s8(sumV);
    const int32x4_t sumV32 = vpaddlq_s16(sumV16);
    const int64x2_t sumV64 = vpaddlq_s32(sumV32);
    sum = vgetq_lane_s64(sumV64, 0) + vgetq_lane_s64(sumV64, 1);
    return sum;
  };
  // Vector Max
  virtual int8_t vectorMax(int8_t *v, int size) override {
    assert(size % 16 == 0);
    int8x16_t maxV = vdupq_n_s8(0);
    for (int i = 0; i < size; i += 16) {
      int8x16_t data = vld1q_s8(v + i);
      maxV = vmaxq_s8(maxV, data);
    }
    const int8x8_t maxV8 = vpmax_s8(vget_low_s8(maxV), vget_high_s8(maxV));
    const int8x8_t maxV4 = vpmax_s8(maxV8, maxV8);
    const int8x8_t maxV2 = vpmax_s8(maxV4, maxV4);
    const int8x8_t maxV1 = vpmax_s8(maxV2, maxV2);
    int8_t max = vget_lane_s8(maxV1, 0);

    return max;
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