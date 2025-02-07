#ifndef SCALAR_H
#define SCALAR_H
#include "kernels.h"
#include <cassert>
#include <cstdint>

class Scalar : public SIMD<int8_t> {
public:
  virtual int8_t *prefixSum(int8_t *v, int size) override {
    int8_t *result = new int8_t[size];
    result[0] = v[0];
    for (int i = 1; i < size; i++) {
      result[i] = result[i - 1] + v[i];
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
                                 int kSize, int **oSize, int padding,
                                 int stride) override {}
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
  virtual int8_t *convolution_2d(int8_t *input, int iRows, int iCols,
                                 int8_t *kernel, int kRows, int kCols,
                                 int **oRows, int **oCols, int padding,
                                 int stride) override {
    int outRows = (iRows + 2 * padding - kRows) / stride + 1;
    int outCols = (iCols + 2 * padding - kCols) / stride + 1;
    *oRows = new int(outRows);
    *oCols = new int(outCols);
    int8_t *output = new int8_t[outRows * outCols];
    for (int i = 0; i < outRows; i++) {
      for (int j = 0; j < outCols; j++) {
        int sum = 0;
        for (int k = 0; k < kRows; k++) {
          for (int l = 0; l < kCols; l++) {
            int inputRow = i * stride + k;
            int inputCol = j * stride + l;
            if (inputRow < 0 || inputRow >= iRows || inputCol < 0 ||
                inputCol >= iCols) {
              continue;
            }
            sum += input[inputRow * iCols + inputCol] * kernel[k * kCols + l];
          }
        }
        output[i * outCols + j] = sum;
      }
    }
    return output;
  }
};

#endif