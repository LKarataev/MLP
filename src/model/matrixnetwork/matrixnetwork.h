#ifndef SRC_MODEL_MATRIXNETWORK_MATRIXNETWORK_H_
#define SRC_MODEL_MATRIXNETWORK_MATRIXNETWORK_H_

#include <fstream>

#include "../interfaces/neuralnetwork.h"
#include "matrixneuron.h"

namespace s21 {
class MatrixNetwork : public NeuralNetwork {
 public:
  explicit MatrixNetwork(int number_of_hidden_layers);
  ~MatrixNetwork() override;

  void Train(const std::pair<int, std::vector<double>>& data) override;
  int Classify(const std::vector<double>& data) override;
  void LoadWeights(const std::string& filename) override;
  void SaveWeights(const std::string& filename) override;

 private:
  void SetInputData(std::vector<double> data);
  void PassForward();
  void BackPropagate(int label);
  void InitializeLayers(int number_of_hidden_layers);

  std::vector<std::vector<MatrixNeuron>> layers_;
};
}  // namespace s21

#endif  // SRC_MODEL_MATRIXNETWORK_MATRIXNETWORK_H_
