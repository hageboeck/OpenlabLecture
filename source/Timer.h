#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * A timer to measure wall clock time from creation until it goes out of scope.
 * Use as
 * ```
 * {
 *   Timer timer{ "description" };
 *   ...
 * }
 * ```
 */
struct Timer {
  static constexpr unsigned int outputWidth = 30;
  std::chrono::time_point<std::chrono::steady_clock> _start;
  std::string _message;

  Timer(std::string message) :
  _start{ std::chrono::steady_clock::now() },
  _message{ message }
  { }

  ~Timer()
  {
    using namespace std::chrono_literals;
    auto end = std::chrono::steady_clock::now();
    std::cout << std::setw(outputWidth) << std::left << _message << " " << (end - _start)/1ms << " ms\n";
  }
};

