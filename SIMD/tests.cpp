#include "tests.h"
#include <cstdint>
#include <iostream>

using namespace std;
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
  free(v1);
  free(v2);
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
  free(v1);
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
  free(v1);
  free(result);
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
    v1[i] = i % 4;
  }
  int8_t *kernel =
      new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int **oSize;
  auto res = impl->convolution_1d(v1, 64, kernel, 16, oSize, 0, 1);
  int **oSize2;
  int8_t *result = impl2->convolution_1d(v1, 64, kernel, 16, oSize2, 0, 1);
  if (*oSize != *oSize2) {
    cout << "Failed: " << oSize << " != " << oSize2 << endl;
    return;
  }
  for (int i = 0; i < 49; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: " << res[i] << " != " << result[i] << endl;
      return;
    }
  }
  cout << "Conv1D passed" << endl;
  free(v1);
  free(kernel);
  free(result);
  free(*oSize);
  free(oSize);
  free(*oSize2);
  free(oSize2);
}
template <> void testConv2D(SIMD<int8_t> *impl, SIMD<int8_t> *impl2) {
  int8_t *v1 = new int8_t[64];
  for (int i = 0; i < 64; i++) {
    v1[i] = i % 4;
  }
  int8_t *kernel =
      new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int **oSize;
  auto res = impl->convolution_2d(v1, 8, kernel, 4, oSize, 0, 1);
  int **oSize2;
  int8_t *result = impl2->convolution_2d(v1, 8, kernel, 4, oSize2, 0, 1);
  if (oSize != oSize2) {
    cout << "Failed: " << oSize << " != " << oSize2 << endl;
    return;
  }
  for (int i = 0; i < 25; i++) {
    if (res[i] != result[i]) {
      cout << "Failed: " << res[i] << " != " << result[i] << endl;
      return;
    }
  }
  cout << "Conv2D passed" << endl;
  free(v1);
  free(kernel);
  free(result);
  free(*oSize);
  free(oSize);
  free(*oSize2);
  free(oSize2);
}
template <> void testGEMM(SIMD<int8_t> *impl, SIMD<int8_t> *impl2) {
  throw runtime_error("Not Implemented");
}
