#include <vector>
enum class Computation { Scalar, AVX, NEON };

template <Computation C>
std::vector<float> convolution_1d(std::vector<float> input,
                                  std::vector<float> kernel, int padding,
                                  int stride);
