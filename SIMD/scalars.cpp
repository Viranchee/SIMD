#include "kernels.h"

template <Computation C, typename T> T *vectorAdd(T *v1, T *v2, int size) {
  T *result = new T[size];
  for (int i = 0; i < size; i++) {
    result[i] = v1[i] + v2[i];
  }
  return result;
}

// specialize

template <>
int *vectorAdd<Computation::Scalar, int>(int *v1, int *v2, int size) {
  return vectorAdd<Computation::Scalar, int>(v1, v2, size);
}

bool testScalarImplementations() {}

class Scalar : public SIMD<int> {
public:
  virtual int *prefixSum(int *v, int size) override {
    int *result = new int[size];
    result[0] = v[0];
    for (int i = 1; i < size; i++) {
      result[i] = result[i - 1] + v[i];
    }
    return result;
  }

  virtual int *vectorAdd(int *v1, int *v2, int size) override {
    int *result = new int[size];
    for (int i = 0; i < size; i++) {
      result[i] = v1[i] + v2[i];
    }
    return result;
  }

  virtual int vectorReduce(int *v, int size) override {
    int sum = 0;
    for (int i = 0; i < size; i++) {
      sum += v[i];
    }
    return sum;
  }

  virtual int vectorMax(int *v, int size) override {
    int max = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] > max) {
        max = v[i];
      }
    }
    return max;
  }

  virtual int vectorMin(int *v, int size) override {
    int min = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] < min) {
        min = v[i];
      }
    }
    return min;
  }

  virtual int *convolution_1d(int *input, int iSize, int *kernel, int kSize,
                              int **oSize, int padding, int stride) override {
    int outSize = (iSize + 2 * padding - kSize) / stride + 1;
    *oSize = new int(outSize);
    int *output = new int[outSize];
    for (int i = 0; i < outSize; i++) {
      output[i] = 0;
      for (int j = 0; j < kSize; j++) {
        output[i] += input[i * stride + j] * kernel[j];
      }
    }
    return output;
  }

  virtual int *matMul(int *A, int M, int *B, int N, int K) override {
    int *C = new int[M * N];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] = 0;
        for (int k = 0; k < K; k++) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
    return C;
  }
};