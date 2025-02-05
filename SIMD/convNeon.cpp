#include "conv.h"
#include <arm_neon.h>

template <>
std::vector<float> convolution_1d<Computation::NEON>(std::vector<float> input,
                                                     std::vector<float> kernel,
                                                     int padding, int stride) {
  std::vector<float> output;
  output.resize((input.size() + 2 * padding - kernel.size()) / stride + 1);

  // ARM NEON SIMD implementation
  for (int i = 0; i < output.size(); i++) {
    float32x4_t sum = vdupq_n_f32(0);
    for (int j = 0; j < kernel.size(); j += 4) {
      int input_index = i * stride + j - padding;
      float32x4_t input_values = vld1q_f32(&input[input_index]);
      float32x4_t kernel_values = vld1q_f32(&kernel[j]);
      sum = vmlaq_f32(sum, input_values, kernel_values);
    }
    output[i] = sum[0] + sum[1] + sum[2] + sum[3];
  }

  return output;
}