#ifndef JULIA_H
#define JULIA_H

#ifdef USE_BOOST_GIL
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
namespace gil = boost::gil;
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>

std::tuple<int, int, int> colormapPPM(int c)
{
    if (!c)
      return {0, 0, 0};

    int r = 0, g = 0, b = 0;
    const int val = (c%64) * 4;

    if (c < 64) {
      r = 0; g = 0; b = val;
    } else if (c >= 64 && c < 128) {
      r = 0; g = val; b = 255 - val;
    } else if (c >= 128 && c < 192) {
      r = val; g = 255; b = 0;
    } else {
      r = 255; g = 255-val; b = 0;
    }

    return {r, g, b};
}

void writePPM(unsigned char const * pixels, size_t nx, size_t ny, const char * filename)
{
  std::ofstream output{filename, std::ios::trunc};

  if (!output) {
    std::cerr << "Error: cannot open file " << filename << "\n";
    return;
  }

  output << "P3\n" << nx << " " << ny << "\n" << 256 << "\n";


  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      unsigned char col = pixels[i + (ny-j-1)*nx];
      auto [r, g, b] = colormapPPM(col);
      output << r << " " << g << " " << b << "\t";
    }
    output << "\n";
  }
  output << "\n";
    
  std::cout << "Wrote " << filename << "\n";
}

#ifdef USE_BOOST_GIL
void writePNG(unsigned char const * pixels, size_t nx, size_t ny, const char * filename) {
  gil::rgb8_image_t img(nx, ny);
  auto view = gil::view(img);

  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      unsigned char col = pixels[i + (ny-j-1)*nx];
      auto [r, g, b] = colormapPPM(col);
      gil::rgb8_pixel_t pix(r, g, b);
      view(i, j) = pix;
    }
  }

  gil::write_view(filename, gil::const_view(img), gil::png_tag{});
}
#else
void writePNG(unsigned char const * /*pixels*/, size_t /*nx*/, size_t /*ny*/, const char * /*filename*/) {
  std::cerr << "writePNG requires boost::gil and libpng\n";
}
#endif

void juliaCPU(float xmin, float xmax, size_t nx,
           float ymin, float ymax, size_t ny,
           size_t maxIter, float maxMagnitude, unsigned char *image,
           float cReal, float cImag)
{
  const float dx = (xmax - xmin) / nx;
  const float dy = (ymax - ymin) / ny;

  for (unsigned int j = 0; j < ny; ++j) {
    for (unsigned int i = 0; i < nx; ++i) {
      float zReal = xmin + i * dx;
      float zImag = ymin + j * dy;
      size_t k = 0;

      do {
        // Compute z^2 + c for complex numbers:
        auto const tmpzReal = zReal*zReal - zImag*zImag + cReal;
        zImag = 2. * zReal*zImag + cImag;
        zReal = tmpzReal;
      } while (++k < maxIter && (zReal*zReal + zImag*zImag) < maxMagnitude*maxMagnitude);

      image[i + nx*j] = k < maxIter ? 1 + (255 * k)/maxIter : 0;
    }
  }
}

#endif
