#include <cstdio>
#include <iostream>

// Tasks:
// ---------------------------------------------------------------------------------
// - Convert the HelloWorld function to a kernel, and call it from main()
// - In the kernel, fill in the variables that print thread index and block index
// - Try a few launch configurations with more threads / more blocks

// kernel definition
void HelloWorld() {
  printf("Hello world from block %d thread %d.\n", -1, -1);
}

int main() {
  const auto nBlock = 1;
  const auto nThread = 1;

  HelloWorld();

  if (auto errorCode = cudaDeviceSynchronize();
      errorCode != cudaSuccess) {
    std::cerr << "Encountered cuda error '"
      << cudaGetErrorName(errorCode)
      << "' with description: "
      << cudaGetErrorString(errorCode) << "\n";
    return 1;
  }

  return 0;
}

