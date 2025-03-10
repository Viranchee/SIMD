#ifndef SIMD_VECTORNEON_CPP
#define SIMD_VECTORNEON_CPP
#include "kernels.h"
#include <arm_neon.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>

// Use ARM vector instructions
using namespace std;

class Neon : public SIMD<int8_t> {
private:
  const int L3 = 64;

public:
  //

  virtual int8_t *prefixSum(int8_t *v, int size) override {
    assert(size % 16 == 0);
    int8_t *result = new int8_t[size];
    int8_t carry = 0;
    for (int i = 0; i < size; i += 16) {
      int8x16_t sum = vld1q_s8(v + i);
      int8x16_t temp = vextq_s8(vdupq_n_s8(0), sum, 15);
      sum = vaddq_s8(sum, temp);
      temp = vextq_s8(vdupq_n_s8(0), sum, 14);
      sum = vaddq_s8(sum, temp);
      temp = vextq_s8(vdupq_n_s8(0), sum, 12);
      sum = vaddq_s8(sum, temp);
      temp = vextq_s8(vdupq_n_s8(0), sum, 8);
      sum = vaddq_s8(sum, temp);
      sum = vaddq_s8(sum, vdupq_n_s8(carry));
      carry = vgetq_lane_s8(sum, 15);
      vst1q_s8(result + i, sum);
    }

    return result;
  }

  virtual int8_t *vectorAdd(int8_t *v1, int8_t *v2, int size) override {
    assert(size % 16 == 0);
    int8_t *result = new int8_t[size];
    for (int i = 0; i < size; i += 16) {
      int8x16_t a = vld1q_s8(v1 + i);
      int8x16_t b = vld1q_s8(v2 + i);
      int8x16_t sum = vaddq_s8(a, b);
      vst1q_s8(result + i, sum);
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
                                 int kSize, int &oSize, int padding,
                                 int stride) override {
    assert(iSize % 16 == 0);
    assert(kSize % 16 == 0);

    const int outSize = (iSize + 2 * padding - kSize) / stride + 1;
    oSize = outSize;
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

    return output;
  }

  virtual int8_t *convolution_2d(int8_t *input, int iSide, int8_t *kernel,
                                 int kSide, int **oSide, int padding,
                                 int stride) override {
    assert(iSide % 4 == 0);
    assert(kSide % 4 == 0);
    int outSize = (iSide + 2 * padding - kSide) / stride + 1;
    *oSide = new int(outSize);
    int8_t *output = new int8_t[outSize * outSize];
    //
    for (int i = 0; i < outSize; i++) {
      for (int j = 0; j < outSize; j++) {
        int32_t sum = 0;
        // kside is 4
        for (int ki = 0; ki < kSide; ki += 4) {
          for (int kj = 0; kj < kSide; kj += 4) {
            int row = i * stride + ki - padding;
            int col = j * stride + kj - padding;
            if (row >= 0 && row < iSide && col >= 0 && col < iSide) {
              int8x16_t inputV = vld1q_s8(&input[row * iSide + col]);
              int8x16_t kernelV = vld1q_s8(&kernel[ki * kSide + kj]);
              int16x8_t prod1 =
                  vmull_s8(vget_low_s8(inputV), vget_low_s8(kernelV));
              int16x8_t prod2 =
                  vmull_s8(vget_high_s8(inputV), vget_high_s8(kernelV));
              int16x8_t sumV = vaddq_s8(prod1, prod2);
              sum += vaddvq_s16(sumV);
            }
          }
        }
        output[i * outSize + j] =
            static_cast<int8_t>(std::max(INT8_MIN, std::min(INT8_MAX, sum)));
      }
    }

    return output;
  }

  virtual int8_t *matMul(int8_t *A, int M, int8_t *B, int N, int K) override {
    assert(M % 16 == 0);
    assert(N % 16 == 0);
    assert(K % 16 == 0);
    int8_t *C = new int8_t[M * N]();
    const int tileSize = 16; // Tile size for blocking

    for (int i = 0; i < M; i += tileSize) {
      for (int j = 0; j < N; j += tileSize) {
        for (int k = 0; k < K; k += tileSize) {
          for (int ii = i; ii < std::min(i + tileSize, M); ++ii) {
            for (int jj = j; jj < std::min(j + tileSize, N); ++jj) {
              int32x4_t sumVec = vdupq_n_s32(0);
              for (int kk = k; kk < std::min(k + tileSize, K); kk += 16) {
                int8x16_t Avec = vld1q_s8(A + ii * K + kk);
                int8x16_t Bvec = vld1q_s8(B + kk * N + jj);
                int16x8_t prod1 =
                    vmull_s8(vget_low_s8(Avec), vget_low_s8(Bvec));
                int16x8_t prod2 =
                    vmull_s8(vget_high_s8(Avec), vget_high_s8(Bvec));
                sumVec = vaddq_s32(sumVec, vpaddlq_s16(prod1));
                sumVec = vaddq_s32(sumVec, vpaddlq_s16(prod2));
              }
              int32_t sumArr[4];
              vst1q_s32(sumArr, sumVec);
              int sum = sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];
              C[ii * N + jj] =
                  static_cast<int8_t>(std::max(-128, std::min(127, sum)));
            }
          }
        }
      }
    }

    return C;
  }

  float32x4_t approxETaylorSeries(float32x4_t x) {
    // Use a polynomial approximation for exp(x), like Estrin's scheme
    // Approximation: exp(x) ≈ 1 + x + (x^2)/2 + (x^3)/6 + (x^4)/24
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x4 = vmulq_f32(x3, x);
    float32x4_t x5 = vmulq_f32(x4, x);

    float32x4_t a = vaddq_f32(one, x);
    float32x4_t b =
        vaddq_f32(vmulq_n_f32(x2, 0.5f), vmulq_n_f32(x3, 1.0f / 6.0f));
    float32x4_t c = vaddq_f32(vmulq_n_f32(x4, 1.0f / 24.0f),
                              vmulq_n_f32(x5, 1.0f / 120.0f));

    return vaddq_f32(a, vaddq_f32(b, c));
  }

  float32x4_t approxEBitManipulation(float32x4_t x) {
    // Clamp x to prevent overflow/underflow
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0f)), vdupq_n_f32(-88.0f));

    // Convert ln(2) to float32x4_t
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f); // log2(e)

    // Compute exponent base-2: 2^(x * log2(e))
    float32x4_t y = vmulq_f32(x, log2e);
    int32x4_t i = vcvtq_s32_f32(vsubq_f32(y, vdupq_n_f32(0.5f))); // Floor(y)
    float32x4_t f = vsubq_f32(y, vcvtq_f32_s32(i)); // Fractional part

    // Polynomial approximation for 2^f in range [-0.5, 0.5]
    float32x4_t poly =
        vaddq_f32(vdupq_n_f32(1.0f),
                  vmulq_f32(f,
                            vaddq_f32(vdupq_n_f32(0.6931472f), // ln(2)
                                      vmulq_f32(f, vdupq_n_f32(0.2402265f)))));

    // Reconstruct 2^(x * log2(e)) using exponent bit-shifting trick
    int32x4_t exponent = vaddq_s32(i, vdupq_n_s32(127)); // Bias of 127
    exponent = vshlq_n_s32(exponent, 23); // Shift to exponent position
    float32x4_t result = vreinterpretq_f32_s32(exponent); // Reinterpret bits
    return vmulq_f32(result, poly);                       // Final approximation
  }
  float32x4_t approxERemez(float32x4_t x) {
    // Clamp x to prevent large overflows
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0f)), vdupq_n_f32(-88.0f));

    const float32x4_t c1 = vdupq_n_f32(0.9999997f);
    const float32x4_t c2 = vdupq_n_f32(0.6931472f);
    const float32x4_t c3 = vdupq_n_f32(0.2402265f);
    const float32x4_t c4 = vdupq_n_f32(0.0555051f);
    const float32x4_t c5 = vdupq_n_f32(0.0096181f);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x4 = vmulq_f32(x3, x);
    float32x4_t x5 = vmulq_f32(x4, x);

    return vaddq_f32(
        vaddq_f32(vaddq_f32(vaddq_f32(c1, vmulq_f32(x, c2)), vmulq_f32(x2, c3)),
                  vmulq_f32(x3, c4)),
        vmulq_f32(x4, c5));
  }

  virtual void softMax(float32_t *input, float32_t *output,
                       int length) override {
    float max_val = -INFINITY;
    int i = 0;

    { // Step 1: Find the maximum value for numerical stability

      float32x4_t max_vec = vdupq_n_f32(-INFINITY);
      for (; i <= length - 4; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        max_vec = vmaxq_f32(max_vec, v);
      }
      max_vec = vpmaxq_f32(max_vec, max_vec);
      max_vec = vpmaxq_f32(max_vec, max_vec);
      max_val = vgetq_lane_f32(max_vec, 0);
      // Handle remaining elements
      for (; i < length; i++) {
        max_val = std::max(max_val, input[i]);
      }
    }

    float sum_exp = 0.0f;
    { // Step 2: Compute exponentials and sum
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      i = 0;
      for (; i <= length - 4; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float32x4_t exp_v = approxEBitManipulation(
            vsubq_f32(v, vdupq_n_f32(max_val))); // exp(x - max)
        vst1q_f32(output + i, exp_v);            // Store exponentials
        sum_vec = vaddq_f32(sum_vec, exp_v);     // Accumulate sum
      }

      // Sum up the elements in the vector register
      sum_vec = vpaddq_f32(sum_vec, sum_vec);
      sum_vec = vpaddq_f32(sum_vec, sum_vec);
      sum_exp = vgetq_lane_f32(sum_vec, 0);
      // Handle remaining elements
      for (; i < length; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
      }
    }

    { // Step 3: Normalize by dividing each element by sum_exp
      float32x4_t sum_inv = vdupq_n_f32(1.0f / sum_exp);
      i = 0;
      for (; i <= length - 4; i += 4) {
        float32x4_t v = vld1q_f32(output + i);
        vst1q_f32(output + i, vmulq_f32(v, sum_inv)); // Normalize
      }

      // Handle remaining elements
      for (; i < length; i++) {
        output[i] /= sum_exp;
      }
    }
  }
};

#endif // SIMD_VECTORNEON_CPP