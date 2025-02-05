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