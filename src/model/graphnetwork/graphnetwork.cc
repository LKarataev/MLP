#include "graphnetwork.h"

namespace s21 {

GraphNetwork::GraphNetwork(int number_of_hidden_layers) {
  InitLayers(number_of_hidden_layers);
}

GraphNetwork::~GraphNetwork() {}

void GraphNetwork::Train(const std::pair<int, std::vector<double>>& data) {
  SetInputData(data.second);
  PassForward();
  BackPropagate(data.first - 1);
}

int GraphNetwork::Classify(const std::vector<double>& data) {
  SetInputData(data);
  PassForward();
  double max = 0;
  int res = 0;
  for (int i = 0; i < utils::kOutputSize; ++i) {
    if (output_layer_[i].GetOutput() > max) {
      max = output_layer_[i].GetOutput();
      res = i;
    }
  }
  return res + 1;
}

void GraphNetwork::LoadWeights(const std::string& filename) {
  std::ifstream infile(filename);
  if (!infile) {
    throw std::invalid_argument("Error opening file for reading.");
  }

  int input_size, output_size, hidden_layers_count, hidden_layers_size;
  infile >> input_size >> output_size >> hidden_layers_count >>
      hidden_layers_size;
  CheckWeightsFileParams(input_size, output_size, hidden_layers_count,
                         hidden_layers_size);

  for (auto& layer : hidden_layers_) {
    for (auto& neuron : layer) {
      for (auto& weight : *neuron.GetWeights()) {
        infile >> weight;
      };
    }
  }
  for (auto& neuron : output_layer_) {
    for (auto& weight : *neuron.GetWeights()) {
      infile >> weight;
    }
  }
  infile.close();
}

void GraphNetwork::CheckWeightsFileParams(int input_size, int output_size,
                                          int hidden_layers_count,
                                          int hidden_layers_size) {
  if (input_size != utils::kInputSize) {
    throw std::out_of_range("Error: different input size in weights file.");
  }
  if (output_size != utils::kOutputSize) {
    throw std::out_of_range("Error: different output size in weights file.");
  }
  if (hidden_layers_count != static_cast<int>(hidden_layers_.size())) {
    throw std::out_of_range(
        "Error: different hidden layers count in weights file.");
  }
  if (hidden_layers_size != utils::kHiddenLayersSize) {
    throw std::out_of_range(
        "Error: different hidden layers size in weights file.");
  }
}

void GraphNetwork::SaveWeights(const std::string& filename) {
  std::ofstream outfile(filename);
  if (!outfile) {
    throw std::invalid_argument("Error opening file for writing.");
  }

  outfile << utils::kInputSize << " " << utils::kOutputSize << " "
          << hidden_layers_.size() << " " << utils::kHiddenLayersSize << "\n";

  for (auto& layer : hidden_layers_) {
    for (auto& neuron : layer) {
      for (const auto& weight : *neuron.GetWeights()) {
        outfile << weight << " ";
      }
      outfile << "\n";
    }
  }
  for (auto& neuron : output_layer_) {
    for (const auto& weight : *neuron.GetWeights()) {
      outfile << weight << " ";
    }
    outfile << "\n";
  }
  outfile.close();
}

void GraphNetwork::PassForward() {
  for (auto& layer : hidden_layers_) {
    for (auto& neuron : layer) {
      neuron.ComputeOutput();
    }
  }
  for (auto& neuron : output_layer_) {
    neuron.ComputeOutput();
  }
}

void GraphNetwork::BackPropagate(int label) {
  for (auto& layer : hidden_layers_)
    for (auto& neuron : layer) {
      neuron.SetError(0);
    }
  for (int i = 0; i < static_cast<int>(output_layer_.size()); ++i) {
    output_layer_[i].UpdateNeuron(true, true, i == label);
  }
  for (int i = static_cast<int>(hidden_layers_.size()) - 1; i >= 0; --i)
    for (auto& neuron : hidden_layers_[i]) {
      neuron.UpdateNeuron(i != 0);
    }
}

void GraphNetwork::InitLayers(int number_of_hidden_layers) {
  input_layer_.reserve(utils::kInputSize);
  for (int i = 0; i < utils::kInputSize; i++) {
    input_layer_.emplace_back(GraphNeuron());
  }

  hidden_layers_.reserve(number_of_hidden_layers);
  hidden_layers_.emplace_back(std::vector<GraphNeuron>());
  InitLayer(hidden_layers_[0], input_layer_, utils::kHiddenLayersSize);
  for (int k = 1; k < number_of_hidden_layers; ++k) {
    hidden_layers_.emplace_back(std::vector<GraphNeuron>());
    InitLayer(hidden_layers_[k], hidden_layers_[k - 1],
              utils::kHiddenLayersSize);
  }

  output_layer_.reserve(utils::kOutputSize);
  InitLayer(output_layer_, hidden_layers_[number_of_hidden_layers - 1],
            utils::kOutputSize);
}

void GraphNetwork::InitLayer(std::vector<GraphNeuron>& layer,
                             std::vector<GraphNeuron>& previous_layer,
                             int reserve_size) {
  layer.reserve(reserve_size);
  for (int i = 0; i < reserve_size; ++i) {
    layer.emplace_back(GraphNeuron());
    for (auto& neuron : previous_layer) {
      layer[i].AddInputNeuron(&neuron);
    }
    layer[i].SetRandomWeights();
  }
}

void GraphNetwork::SetInputData(const std::vector<double>& data) {
  for (int i = 0; i < utils::kInputSize; ++i)
    input_layer_[i].SetOutput(data[i]);
}

}  // namespace s21
