#ifndef SRC_MODEL_UTILS_UTILS_H_
#define SRC_MODEL_UTILS_UTILS_H_

#include <cmath>
#include <random>

namespace s21 {

namespace utils {

const int kImgSize = 28;
const int kInputSize = 784;
const int kOutputSize = 26;
const int kHiddenLayersSize = 52;
const double kLearningRate = 0.1;

double RandomDouble();
double ActivateFunction(double x);
double Proportion(double x, double y);
double HarmonicMean(double x, double y);

}  // namespace utils

}  // namespace s21

#endif  // SRC_MODEL_UTILS_UTILS_H_
