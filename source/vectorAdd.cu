#include "Timer.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>

__global__
void add(int n,  int * x,  int * y)
{
  // Task 1:
  // ------------------------------------------------------
  // Set index and stride such that we can run an efficient
  // grid-strided loop

  const auto index = 0;
  const auto stride = 1;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

bool checkResult(int * array, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (array[i] != i) return false;
  }
  return true;
}

int main() {
  const auto N = 100'000'000;

  int * x;
  int * y;
  cudaMallocManaged(&x, N * sizeof( int), cudaMemAttachHost);
  cudaMallocManaged(&y, N * sizeof( int), cudaMemAttachHost);

  // Initialise arrays as follows:
  // x = { 0,  1,  2, ... }
  // y = {-0, -1, -2, ... }
  {
    Timer timer{ "init arrays on host" };
    std::iota(x, x + N, 0);
    std::transform(x, x + N, y, [](int i){ return -1 * i; });
  }

  // Add them once. Now y should be equal to 0:
  {
    Timer timer{ "add on host" };
    for (unsigned int i = 0; i < N; ++i) {
      y[i] = x[i] + y[i];
    }
  }

  // Bring arrays to GPU.
  // Note that this step is optional, because they would automatically
  // be copied once the kernel accesses them.
  // This enables us to time copy and compute separately.
  {
    Timer timer{ "copy to device memory" };
    int currentDevice;
    cudaGetDevice(&currentDevice);
    cudaMemPrefetchAsync(x, N*sizeof(int), currentDevice);
    cudaMemPrefetchAsync(y, N*sizeof(int), currentDevice);

    if (const auto errorCode = cudaDeviceSynchronize();
        errorCode != cudaSuccess) {
      std::cerr << "When copying, encountered cuda error " << errorCode << " '"
        << cudaGetErrorName(errorCode)
        << "' with description:"
        << cudaGetErrorString(errorCode) << "\n";
      return 2;
    }
  }


  // Add them on the GPU. Now y should be {0, 1, 2, ...}
  {
    Timer timer{ "add on device" };

    // Task 2:
    // --------------------------------------------------------
    // Find an efficient launch configuration that exhausts the
    // capabilities of the device

    const auto nBlock = 1;
    const auto nThread = 1;

    //add<<< nBlock , nThread >>>(N, x, y);

    if (const auto errorCode = cudaDeviceSynchronize();
        errorCode != cudaSuccess) {
      std::cerr << "Encountered cuda error '"
        << cudaGetErrorName(errorCode)
        << "' with description: "
        << cudaGetErrorString(errorCode) << "\n";
      return 1;
    }
  }

  {
    Timer timer{ "Access y array on host" };

    std::cout << "\ny[0] = " << y[0]
      << "\ny[" << N/2 << "] = " << y[N/2]
      << "\ny[" << N-1 << "] = " << y[N-1] << "\n";
  }
  if (checkResult(y, N))
    std::cout << "Addition seems to be correct.\n";
  else
    std::cout << "Addition seems to have failed.\n";

  cudaFree(x);
  cudaFree(y);

  return 0;
}

