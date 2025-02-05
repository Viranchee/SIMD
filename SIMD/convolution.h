#include <vector>
enum Computation { Scalar, SIMD_Vector };

template <Computation C>
void convolution_2d(std::vector<float> input, std::vector<float> kernel,
                    std::vector<float> &output, int padding, int stride);
