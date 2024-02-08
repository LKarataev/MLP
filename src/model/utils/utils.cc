#include "utils.h"

namespace s21 {

namespace utils {

double RandomDouble() {
  static std::mt19937_64 engine(std::random_device{}());
  static std::uniform_real_distribution<double> distribution(-1, 1);
  return distribution(engine);
}

double ActivateFunction(double x) { return 1.0 / (1.0 + exp(-x)); }

double Proportion(double x, double y) { return (x + y > 0) ? x / (x + y) : 0; }

double HarmonicMean(double x, double y) {
  return (x + y > 0) ? 2.0 * x * y / (x + y) : 0;
}

}  // namespace utils

}  // namespace s21
