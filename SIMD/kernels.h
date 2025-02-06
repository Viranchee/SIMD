#include <vector>

enum class Computation { Scalar, AVX, NEON };

template <Computation C, typename T> T *vectorAdd(T *v1, T *v2, int size);

template <Computation C, typename T>
std::vector<T> vectoredAdd(std::vector<T> v1, std::vector<T> v2);

template <Computation C>
std::vector<float> convolution_1d(std::vector<float> input,
                                  std::vector<float> kernel, int padding,
                                  int stride);

template <Computation C, typename T>
void convolution_1d(T *input, int inputSize, T *kernel, int kernelSize,
                    T *output, int **outputSize, int padding, int stride);

template <typename T> class SIMD {
public:
  // Prefix Sum
  T prefixSum(T *v, int size) = 0;
  // Vector Add
  T *vectorAdd(T *v1, T *v2, int size) = 0;
  // Add all elements in the vector
  T vectorReduce(T *v, int size) = 0;
  // Vector Max
  T vectorMax(T *v, int size) = 0;
  // Vector Min
  T vectorMin(T *v, int size) = 0;
  // Convolution 1D
  T *convolution_1d(T *input, int iSize, T *kernel, int kSize, int **oSize,
                    int padding, int stride) = 0;
  // MatMul gemm
  T *matMul(T *A, int aRows, int aCols, T *B, int bRows, int bCols, T **C,
            int **cRows, int **cCols) = 0;
};