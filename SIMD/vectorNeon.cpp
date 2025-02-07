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
    int8_t *result = new int8_t[size];
    int8_t runningSum = 0;
    for (int i = 0; i < size; i += 16) {
      int8x16_t zeroVec = vdupq_n_s8(0);
      int8x16_t data = vld1q_s8(v + i);
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 1));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 2));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 4));
      data = vaddq_s8(data, vextq_s8(zeroVec, data, 8));
      data = vaddq_s8(data, vdupq_n_s8(runningSum));
      runningSum = vgetq_lane_s8(data, 15);
      vst1q_s8(result + i, data);
    }

    return result; // Return the result array
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

  virtual int8_t vectorMin(int8_t *v, int size) override {
    assert(size % 16 == 0);
    int8x16_t minV = vdupq_n_s8(INT8_MAX);
    for (int i = 0; i < size; i += 16) {
      int8x16_t data = vld1q_s8(v + i);
      minV = vminq_s8(minV, data);
    }
    const int8x8_t minV8 = vpmin_s8(vget_low_s8(minV), vget_high_s8(minV));
    const int8x8_t minV4 = vpmin_s8(minV8, minV8);
    const int8x8_t minV2 = vpmin_s8(minV4, minV4);
    const int8x8_t minV1 = vpmin_s8(minV2, minV2);
    int8_t min = vget_lane_s8(minV1, 0);

    return min;
  };

  virtual int8_t *convolution_1d(int8_t *input, int iSize, int8_t *kernel,
                                 int kSize, int **oSize, int padding,
                                 int stride) override {
    assert(iSize % 16 == 0);
    assert(kSize % 16 == 0);

    int outSize = (iSize + 2 * padding - kSize) / stride + 1;
    int8_t *output = new int8_t[outSize];

    for (int i = 0; i < outSize; i++) {
      int16x8_t sum = vdupq_n_s16(0); // Use int16 to avoid overflow

      for (int j = 0; j < kSize; j += 16) {
        int8x16_t inputV = vld1q_s8(input + i * stride + j);
        int8x16_t kernelV = vld1q_s8(kernel + j);
        int16x8_t prod1 = vmull_s8(vget_low_s8(inputV), vget_low_s8(kernelV));
        int16x8_t prod2 = vmull_s8(vget_high_s8(inputV), vget_high_s8(kernelV));
        sum = vaddq_s16(sum, prod1);
        sum = vaddq_s16(sum, prod2);
      }
      int result = vaddvq_s16(sum);
      output[i] = (int8_t)std::max(-128, std::min(127, result));
    }

    *oSize = new int(outSize);
    return output;
  }

  virtual int8_t *convolution_2d(int8_t *input, int iSide, int8_t *kernel,
                                 int kSide, int **oSide, int padding,
                                 int stride) override {

    int oSideValue = (iSide + 2 * padding - kSide) / stride + 1;
    *oSide = new int(oSideValue);

    int8_t *output = new int8_t[oSideValue * oSideValue];

    for (int i = 0; i < oSideValue; i++) {
      for (int j = 0; j < oSideValue; j++) {
        int32x4_t sumVec = vdupq_n_s32(0);

        for (int ki = 0; ki < kSide; ki++) {
          for (int kj = 0; kj < kSide; kj += 16) {
            int row = i * stride + ki - padding;
            int col = j * stride + kj - padding;
            if (row >= 0 && row < iSide && col >= 0 && col < iSide) {

              int8x16_t inputVec = vld1q_s8(input + row * iSide + col);
              int8x16_t kernelVec = vld1q_s8(kernel + ki * kSide + kj);

              sumVec = vmlal_s8(sumVec, vget_low_s8(inputVec),
                                vget_low_s8(kernelVec));
              sumVec = vmlal_s8(sumVec, vget_high_s8(inputVec),
                                vget_high_s8(kernelVec));
            }
          }
        }

        int32_t sumArr[4];
        vst1q_s32(sumArr, sumVec);
        int sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];

        output[i * oSideValue + j] = std::min(std::max(sum, -128), 127);
      }
    }

    return output;
  }

  virtual int8_t *matMul(int8_t *A, int M, int8_t *B, int N, int K) override {
    int8_t *C = new int8_t[M * N];
    for (int i = 0; i < M * N; i++) {
      C[i] = 0;
    }

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        int32x4_t sumVec = vdupq_n_s32(0);
        for (int k = 0; k < K; k += 16) {
          int8x16_t Avec = vld1q_s8(A + i * K + k);
          int8x16_t Bvec = vld1q_s8(B + k * N + j);
          sumVec = vmlal_s8(sumVec, vget_low_s8(Avec), vget_low_s8(Bvec));
          sumVec = vmlal_s8(sumVec, vget_high_s8(Avec), vget_high_s8(Bvec));
        }
        int32_t sumArr[4];
        vst1q_s32(sumArr, sumVec);
        int sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
        C[i * N + j] = std::min(std::max(sum, -128), 127);
      }
    }

    return C;
  }
};
#endif // SIMD_VECTORNEON_CPP