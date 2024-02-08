#ifndef SRC_MODEL_INTERFACES_NEURALNETWORK_H_
#define SRC_MODEL_INTERFACES_NEURALNETWORK_H_

#include <string>
#include <vector>

namespace s21 {

class NeuralNetwork {
 public:
  virtual ~NeuralNetwork() = default;

  virtual void Train(const std::pair<int, std::vector<double>>& data) = 0;
  virtual int Classify(const std::vector<double>& data) = 0;
  virtual void LoadWeights(const std::string& filename) = 0;
  virtual void SaveWeights(const std::string& filename) = 0;
};
}  // namespace s21

#endif  // SRC_MODEL_INTERFACES_NEURALNETWORK_H_
