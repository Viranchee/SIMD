#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H
#include <cstdint>
#include <vector>

// enum class Computation { Scalar, AVX, NEON };

// template <Computation C, typename T> T *vectorAdd(T *v1, T *v2, int size);

// template <Computation C, typename T>
// std::vector<T> vectoredAdd(std::vector<T> v1, std::vector<T> v2);

// template <Computation C>
// std::vector<float> convolution_1d(std::vector<float> input,
//                                   std::vector<float> kernel, int padding,
//                                   int stride);

// template <Computation C, typename T>
// void convolution_1d(T *input, int inputSize, T *kernel, int kernelSize,
//                     T *output, int **outputSize, int padding, int stride);

template <typename T> class SIMD {
public:
  // Initializer
  SIMD() {}
  // Prefix Sum
  virtual T *prefixSum(T *v, int size) = 0;
  // Vector Add
  virtual T *vectorAdd(T *v1, T *v2, int size) = 0;
  // Add all elements in the vector
  virtual T vectorReduce(T *v, int size) = 0;
  // Vector Max
  virtual T vectorMax(T *v, int size) = 0;
  // Vector Min
  virtual T vectorMin(T *v, int size) = 0;
  // Convolution 1D
  virtual T *convolution_1d(T *input, int iSize, T *kernel, int kSize,
                            int **oSize, int padding, int stride) = 0;
  // MatMul gemm: MxK * KxN = MxN
  virtual T *matMul(T *A, int M, T *B, int N, int K) = 0;

  virtual ~SIMD() {}
};
#endif