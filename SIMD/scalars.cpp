#ifndef SCALAR_H
#define SCALAR_H
#include "kernels.h"
#include <arm_neon.h>
#include <cassert>
#include <cmath>
#include <cstdint>

class Scalar : public SIMD<int8_t> {
public:
  virtual int8_t *prefixSum(int8_t *v, int size) override {
    int8_t *result = new int8_t[size];
    result[0] = v[0];
    int8_t acc = v[0];
    for (int i = 1; i < size; i++) {
      result[i] = v[i] + acc;
      acc += v[i];
    }
    return result;
  }
  virtual int8_t *vectorAdd(int8_t *v1, int8_t *v2, int size) override {
    int8_t *result = new int8_t[size];
    for (int i = 0; i < size; i++) {
      result[i] = v1[i] + v2[i];
    }
    return result;
  }
  virtual int8_t vectorReduce(int8_t *v, int size) override {
    int8_t result = 0;
    for (int i = 0; i < size; i++) {
      result += v[i];
    }
    return result;
  }
  virtual int8_t vectorMax(int8_t *v, int size) override {
    int8_t result = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] > result) {
        result = v[i];
      }
    }
    return result;
  }
  virtual int8_t vectorMin(int8_t *v, int size) override {
    int8_t result = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] < result) {
        result = v[i];
      }
    }
    return result;
  }
  virtual int8_t *convolution_1d(int8_t *input, int iSize, int8_t *kernel,
                                 int kSize, int &oSize, int padding,
                                 int stride) override {
    const int outSize = (iSize + 2 * padding - kSize) / stride + 1;
    oSize = outSize;
    int8_t *output = new int8_t[outSize];

    for (int i = 0; i < outSize; i++) {
      int32_t sum = 0;
      for (int j = 0; j < kSize; j++) {
        int inputIdx = i * stride + j - padding;
        if (inputIdx >= 0 && inputIdx < iSize) {
          sum += input[inputIdx] * kernel[j];
        }
      }
      output[i] = std::max(INT8_MIN, std::min(INT8_MAX, sum));
    }
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
        int32_t sum = 0;
        for (int ki = 0; ki < kSide; ki++) {
          for (int kj = 0; kj < kSide; kj++) {
            int row = i * stride + ki - padding;
            int col = j * stride + kj - padding;
            if (row >= 0 && row < iSide && col >= 0 && col < iSide) {
              sum += input[row * iSide + col] * kernel[ki * kSide + kj];
            }
          }
        }
        output[i * oSideValue + j] =
            static_cast<int8_t>(std::max(INT8_MIN, std::min(INT8_MAX, sum)));
      }
    }

    return output;
  }

  virtual int8_t *matMul(int8_t *A, int M, int8_t *B, int N, int K) override {
    int8_t *result = new int8_t[M * N];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        result[i * N + j] = sum;
      }
    }
    return result;
  }

  virtual void softMax(float32_t *input, float32_t *output,
                       int length) override {
    // Do softmax over x axis in this 1D tensor
    float32_t maxVal = -INFINITY;
    float32_t sum = 0;
    for (int i = 0; i < length; i++) {
      maxVal = std::max(maxVal, input[i]);
    }

    for (int i = 0; i < length; i++) {
      output[i] = std::exp(input[i] - maxVal);
      sum += output[i];
    }

    // Normalize
    for (int i = 0; i < length; i++) {
      output[i] /= sum;
    }
    return;
  }
};

#endif