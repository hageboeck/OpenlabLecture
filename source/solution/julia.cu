#include "Timer.h"
#include "julia.h"

#include <iostream>
#include <cstdio>

// This function writes an image to disk
void writePPM(unsigned char const * pixels, size_t nx, size_t ny, const char * filename);
struct Timer;

// Use this to change the floating-point precision in the kernel
using FPType = double;

__global__
void julia(FPType xmin, FPType xmax, size_t nx,
           FPType ymin, FPType ymax, size_t ny,
           size_t maxIter, FPType maxMagnitude, unsigned char * image,
           FPType cReal, FPType cImag)
{
  const FPType dx = (xmax - xmin) / nx;
  const FPType dy = (ymax - ymin) / ny;

  // Task 3: From threadIdx and blockIdx, compute the indices i and j
  // to address the pixels in x and y direction
  // ------------------------------------------------------------------
  const size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t i = threadId / nx;
  const size_t j = threadId % nx;

  if (i >= nx || j >= ny) return;

  // Compute the starting values for z based on the pixel location
  FPType zReal = xmin + i * dx;
  FPType zImag = ymin + j * dy;

  // Task 4: Compute Julia set
  // -----------------------------
  size_t k = 0;
  while (k < maxIter && (zReal*zReal + zImag*zImag) < maxMagnitude*maxMagnitude) {
    // Compute z^2 + c for complex numbers:
    auto const tmpzReal = zReal*zReal - zImag*zImag + cReal;
    zImag = 2. * zReal*zImag + cImag;
    zReal = tmpzReal;
    ++k;
  }

  image[i + nx*j] = k < maxIter ? 1 + (255 * k)/maxIter : 0;
}


int main(int argc, char * argv[]) {
  // Set up:
  constexpr auto plotRange = 1.6;
  const FPType cReal = argc > 1 ? std::stod(argv[1]) : -0.4;
  const FPType cImag = argc > 2 ? std::stod(argv[2]) :  0.6;
  constexpr size_t sizeX = 1024;
  constexpr size_t sizeY = 1024;

  // Task 1: Allocate memory
  // -----------------------
  unsigned char * pixels;
  cudaMallocManaged(&pixels, sizeX * sizeY * sizeof(unsigned char));

  if (const auto errorCode = cudaGetLastError(); errorCode != cudaSuccess) {
    std::cerr << "When allocating memory, encountered cuda error " << errorCode << " '"
              << cudaGetErrorName(errorCode)
              << "' with description:"
              << cudaGetErrorString(errorCode) << "\n";
    return 2;
  }

  /* call julia kernel to draw the Julia set into a buffer */
  {
    Timer kernelTimer{ "Compute Julia set" };

    // Task 2: Launch the kernel
    // -------------------------
    constexpr auto nThread = 1024;
    constexpr auto nBlock = (sizeX*sizeY + nThread - 1) / nThread;

    julia<<<nBlock, nThread>>>(-plotRange, plotRange, sizeX, -plotRange, plotRange, sizeY, 256, 2.f, pixels, cReal, cImag);

    if (const auto errorCode = cudaDeviceSynchronize(); errorCode != cudaSuccess) {
      std::cerr << "When submitting kernel, encountered cuda error '"
                << cudaGetErrorName(errorCode)
                << "' with description:"
                << cudaGetErrorString(errorCode) << "\n";
      return 3;
    }
  }

  writePPM(pixels, sizeX, sizeY, "julia.ppm");

  cudaFree(pixels);

  return 0;
}
