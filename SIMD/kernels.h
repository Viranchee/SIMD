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
