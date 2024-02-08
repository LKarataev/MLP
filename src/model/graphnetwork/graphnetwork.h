#ifndef SRC_MODEL_GRAPHNETWORK_GRAPHNETWORK_H_
#define SRC_MODEL_GRAPHNETWORK_GRAPHNETWORK_H_

#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

#include "../interfaces/neuralnetwork.h"
#include "graphneuron.h"

namespace s21 {

class GraphNetwork : public NeuralNetwork {
 public:
  explicit GraphNetwork(int number_of_hidden_layers);
  ~GraphNetwork() override;

  void Train(const std::pair<int, std::vector<double>> &data) override;
  int Classify(const std::vector<double> &data) override;
  void LoadWeights(const std::string &filename) override;
  void SaveWeights(const std::string &filename) override;

 private:
  void SetInputData(const std::vector<double> &data);
  void PassForward();
  void BackPropagate(int label);
  void InitLayers(int number_of_hidden_layers);
  void InitLayer(std::vector<GraphNeuron> &layer,
                 std::vector<GraphNeuron> &previous_layer, int reserve_size);
  void CheckWeightsFileParams(int input_size, int output_size,
                              int hidden_layers_count, int hidden_layers_size);

  std::vector<GraphNeuron> input_layer_;
  std::vector<std::vector<GraphNeuron>> hidden_layers_;
  std::vector<GraphNeuron> output_layer_;
};

}  // namespace s21

#endif  // SRC_MODEL_GRAPHNETWORK_GRAPHNETWORK_H_
