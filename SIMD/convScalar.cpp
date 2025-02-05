#include "conv.h"

template <>
std::vector<float>
convolution_1d<Computation::Scalar>(std::vector<float> input,
                                    std::vector<float> kernel, int padding,
                                    int stride) {
  // Scalar implementation
  std::vector<float> output;
  output.resize((input.size() + 2 * padding - kernel.size()) / stride + 1);

  for (int i = 0; i < output.size(); i++) {
    float sum = 0;
    for (int j = 0; j < kernel.size(); j++) {
      int input_index = i * stride + j - padding;
      if (input_index >= 0 && input_index < input.size()) {
        sum += input[input_index] * kernel[j];
      }
    }
    output[i] = sum;
  }
  return output;
}
