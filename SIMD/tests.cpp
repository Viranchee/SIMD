#include "tests.h"
#include <cstdint>
#include <iostream>

using namespace std;
template <> void testVectorAdd<int8_t>(SIMD<int8_t> *impl) {
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
}
template <typename T> void testVectorAdd(SIMD<T> *impl) {
  throw runtime_error("Not Implemented");
}
template <> void testVectorReduce<int8_t>(SIMD<int8_t> *impl) {
  int8_t *v1 = new int8_t[16]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  int8_t res = impl->vectorReduce(v1, 16);
  if (res != 40) {
    cout << "Failed: " << res << " != " << 40 << endl;
    return;
  }
  cout << "VectorReduce passed" << endl;
}