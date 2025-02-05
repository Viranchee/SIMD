#include <iostream>

extern "C" int IntegerAddSub_(int a, int b, int c, int d);

int main() {
  std::cout << "Hello, World!" << std::endl;
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  int result = IntegerAddSub_(a, b, c, d);
  std::cout << "Result: " << result << std::endl;

  return 0;
}