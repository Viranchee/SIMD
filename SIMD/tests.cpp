#include "tests.h"
#include <arm_neon.h>
#include <cstdint>
#include <iomanip>
#include <iostream>

using namespace std;

auto printArray = [](int8_t *res, int size, string name = "") {
  cout << name << ": " << size << ":" << endl;
  for (int i = 0; i < size; i++) {
    int8_t val = res[i];
    cout << setw(3) << static_cast<int>(val) << " ";
  }
  cout << endl;
};
auto printMatrix = [](int8_t *res, int M, int N, string name = "") {
  cout << name << ": " << M << "x" << N << ":" << endl;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int8_t val = res[i * N + j];
      cout << setw(3) << static_cast<int>(val) << " ";
    }
    cout << endl;
  }

  cout << endl;
};

template <> void testVectorAdd(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int8_t *v2 = new int8_t[16]{5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
  int8_t *result =
      new int8_t[16]{6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12};
  auto res = impl->vectorAdd(v1, v2, 16);
  for (int i = 0; i < 4; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: " << res[i] << " != " << result[i] << endl;
      return;
    }
  }
  cout << "VectorAdd passed" << endl;
  delete[] v1;
  delete[] v2;
}

template <typename T> void testVectorAdd(SIMD<T> *impl) {
  throw runtime_error("Not Implemented");
}

template <> void testVectorReduce(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int8_t res = impl->vectorReduce(v1, 16);
  if (res != 40) {
    cout << "Failed: " << res << " != " << 40 << endl;
    return;
  }
  cout << "VectorReduce passed" << endl;
  delete[] v1;
}

template <> void testPrefixSum(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int8_t *result = new int8_t[16]{1,  3,  6,  10, 11, 13, 16, 20,
                                  21, 23, 26, 30, 31, 33, 36, 40};
  auto res = impl->prefixSum(v1, 16);
  for (int i = 0; i < 16; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: " << res[i] << " != " << result[i] << endl;
      return;
    }
  }
  cout << "PrefixSum passed" << endl;
  delete[] v1;
  delete[] result;
}

template <> void testVectorMin(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int8_t res = impl->vectorMin(v1, 16);
  if (res != 1) {
    cout << "Failed: " << res << " != " << 1 << endl;
    return;
  }
  cout << "VectorMin passed" << endl;
  free(v1);
}

template <> void testVectorMax(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 20};
  int8_t res = impl->vectorMax(v1, 16);
  if (res != 20) {
    cout << "Failed: " << res << " != " << 1 << endl;
    return;
  }
  cout << "VectorMax passed" << endl;
  free(v1);
}

template <> void testConv1D(SIMD<int8_t> *impl, SIMD<int8_t> *impl2) {
  int8_t *v1 = new int8_t[64];
  for (int i = 0; i < 64; i++) {
    v1[i] = 1;
  }
  int8_t *kernel = new int8_t[16];
  for (int i = 0; i < 16; i++) {
    kernel[i] = i % 4; // 0, 1, 2, 3
  }
  int *oSize = new int;
  int *oSize2 = new int;
  auto res = impl->convolution_1d(v1, 64, kernel, 16, *oSize, 0, 1);
  auto result = impl2->convolution_1d(v1, 64, kernel, 16, *oSize2, 0, 1);
  if (*oSize != *oSize2) {
    cout << "Failed: " << oSize << " != " << oSize2 << endl;
    return;
  }
  for (int i = 0; i < *oSize; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: " << res[i] << " != " << result[i] << endl;
      return;
    }
  }
  cout << "Conv1D passed" << endl;
  delete[] v1;
  delete[] kernel;
  delete[] result;
  delete[] res;
  delete oSize;
  delete oSize2;
}

template <> void testConv2D(SIMD<int8_t> *impl, SIMD<int8_t> *impl2) {
  const int iSide = 8;
  const int kSide = 4;
  int8_t *v1 = new int8_t[iSide * iSide];
  for (int i = 0; i < iSide * iSide; i++) {
    v1[i] = 1;
  }
  int8_t *kernel = new int8_t[kSide * kSide];
  for (int i = 0; i < kSide * kSide; i++) {
    kernel[i] = 1;
  }
  int *oSize;
  auto res = impl->convolution_2d(v1, iSide, kernel, kSide, &oSize, 0, 1);

  int *oSize2;
  int8_t *result =
      impl2->convolution_2d(v1, iSide, kernel, kSide, &oSize2, 0, 1);

  if (*oSize != *oSize2) {
    cout << "Failed: " << oSize << " != " << oSize2 << endl;
    return;
  }
  for (int i = 0; i < *oSize; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: idx " << i << " Values: " << res[i]
           << " != " << result[i] << endl;
      return;
    }
  }
  cout << "Conv2D passed" << endl;
  delete[] v1;
  delete[] kernel;
  delete[] result;
  delete[] res;
  delete oSize;
  delete oSize2;
}

template <> void testSoftmax(SIMD<float32_t> *impl, SIMD<float32_t> *impl2) {
  const int length = 16;
  float32_t *input = new float32_t[length];
  for (int i = 0; i < length; i++) {
    input[i] = i;
  }
  float32_t *output = new float32_t[length];
  float32_t *output2 = new float32_t[length];
  impl->softMax(input, output, length);
  impl2->softMax(input, output2, length);
  for (int i = 0; i < length; i++) {
    if (output[i] != output2[i]) {
      cout << "Failed: idx " << i << " Values: " << output[i]
           << " != " << output2[i] << endl;
      return;
    }
  }
  cout << "Softmax passed" << endl;
  delete[] input;
  delete[] output;
  delete[] output2;
}

template <> void testGEMM(SIMD<int8_t> *impl, SIMD<int8_t> *impl2) {
  const int mkn[] = {16, 16, 16};
  int8_t *A = new int8_t[mkn[0] * mkn[1]];
  int8_t *B = new int8_t[mkn[1] * mkn[2]];
  for (int i = 0; i < mkn[0] * mkn[1]; i++) {
    A[i] = i % 4;
    B[i] = i % 4;
  }
  printMatrix(A, mkn[0], mkn[2], "in A");
  printMatrix(B, mkn[0], mkn[2], "in B");
  int8_t *result = impl->matMul(A, mkn[0], B, mkn[2], mkn[1]);
  printMatrix(result, mkn[0], mkn[2], "scalar");
  int8_t *result2 = impl2->matMul(A, mkn[0], B, mkn[2], mkn[1]);
  printMatrix(result2, mkn[0], mkn[2], "vector");
  for (int i = 0; i < mkn[0] * mkn[2]; i++) {
    if (result[i] != result2[i]) {
      cout << "Failed: idx: " << i << " " << result[i] << " != " << result2[i]
           << endl;
      return;
    }
  }
  cout << "GEMM passed" << endl;
  delete[] A;
  delete[] B;
  delete[] result;
  delete[] result2;
}
