#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#endif
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  return 0;
}