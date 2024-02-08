#include "matrixnetwork.h"

namespace s21 {
MatrixNetwork::MatrixNetwork(int number_of_hidden_layers) {
  InitializeLayers(number_of_hidden_layers);
}

MatrixNetwork::~MatrixNetwork() {}

int MatrixNetwork::Classify(const std::vector<double> &data) {
  SetInputData(data);
  PassForward();

  std::vector<MatrixNeuron> &output_layer = layers_[layers_.size() - 1];
  int index = 0;
  double max = output_layer[0].GetOutput();

  for (size_t i = 0; i < output_layer.size(); ++i) {
    if (output_layer[i].GetOutput() > max) {
      max = output_layer[i].GetOutput();
      index = i;
    }
  }

  return index + 1;
}

void MatrixNetwork::LoadWeights(const std::string &filename) {
  std::ifstream file(filename, std::ios::out | std::ios::binary);

  if (!file) {
    throw std::invalid_argument(
        "MatrixNetwork::LoadWeights::Error opening the file!");
  }

  int ImgSize, InputSize, OutputSize, HiddenLayersSize, HiddenLayersCount;

  file.read(reinterpret_cast<char *>(&ImgSize), sizeof(int));
  file.read(reinterpret_cast<char *>(&InputSize), sizeof(int));
  file.read(reinterpret_cast<char *>(&OutputSize), sizeof(int));
  file.read(reinterpret_cast<char *>(&HiddenLayersSize), sizeof(int));
  file.read(reinterpret_cast<char *>(&HiddenLayersCount), sizeof(int));

  if (ImgSize != utils::kImgSize || InputSize != utils::kInputSize ||
      OutputSize != utils::kOutputSize ||
      HiddenLayersSize != utils::kHiddenLayersSize ||
      HiddenLayersCount != static_cast<int>(layers_.size()) - 2) {
    std::string error =
        "MatrixNetwork::LoadWeights::Invalid network parameters. ImgSize: " +
        std::to_string(ImgSize) + " InputSize: " + std::to_string(InputSize) +
        " OutputSize: " + std::to_string(OutputSize) +
        " HiddenLayersSize: " + std::to_string(HiddenLayersSize) +
        " HiddenLayersCount: " + std::to_string(HiddenLayersCount);
    throw std::invalid_argument(error);
  }

  for (size_t i = 1; i < layers_.size(); ++i) {
    auto &layer = layers_[i];
    for (auto &neuron : layer) {
      for (auto &weight : *neuron.GetWeights()) {
        file.read(reinterpret_cast<char *>(&weight), sizeof(double));
      }
    }
  }

  file.close();
}

void MatrixNetwork::SaveWeights(const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  if (!file) {
    throw std::invalid_argument(
        "MatrixNetwork::SaveWeights::Error opening the file!");
  }

  const int kHiddenLayersCount = layers_.size() - 2;

  file.write(reinterpret_cast<const char *>(&utils::kImgSize), sizeof(int));
  file.write(reinterpret_cast<const char *>(&utils::kInputSize), sizeof(int));
  file.write(reinterpret_cast<const char *>(&utils::kOutputSize), sizeof(int));
  file.write(reinterpret_cast<const char *>(&utils::kHiddenLayersSize),
             sizeof(int));
  file.write(reinterpret_cast<const char *>(&kHiddenLayersCount), sizeof(int));

  for (size_t i = 1; i < layers_.size(); ++i) {
    auto &layer = layers_[i];
    for (auto &neuron : layer) {
      for (auto &weight : *neuron.GetWeights()) {
        file.write(reinterpret_cast<char *>(&weight), sizeof(double));
      }
    }
  }

  file.close();
}

void MatrixNetwork::InitializeLayers(int number_of_hidden_layers) {
  size_t layers_count = number_of_hidden_layers + 2;
  size_t neurons_count;

  layers_.resize(layers_count);
  for (size_t i = 0; i < layers_count; ++i) {
    if (i == 0) {
      neurons_count = utils::kInputSize;
    } else if (i == layers_count - 1) {
      neurons_count = utils::kOutputSize;
    } else {
      neurons_count = utils::kHiddenLayersSize;
    }

    layers_[i].reserve(neurons_count);

    std::vector<MatrixNeuron> *previous_layer =
        i == 0 ? nullptr : &layers_[i - 1];

    for (size_t j = 0; j < neurons_count; ++j) {
      layers_[i].emplace_back(MatrixNeuron(previous_layer));
    }
  }
}

void MatrixNetwork::SetInputData(std::vector<double> data) {
  for (size_t i = 0; i < layers_[0].size(); ++i) {
    layers_[0][i].SetOutput(data[i]);
  }
}

void MatrixNetwork::PassForward() {
  for (auto &layer : layers_) {
    for (auto &neuron : layer) {
      neuron.ComputeOutput();
    }
  }
}

void MatrixNetwork::BackPropagate(int label) {
  for (auto &layer : layers_) {
    for (auto &neuron : layer) {
      neuron.SetError(0);
    }
  }

  std::vector<MatrixNeuron> &output_layer = layers_[layers_.size() - 1];

  for (int i = 0; i < static_cast<int>(output_layer.size()); ++i) {
    output_layer[i].ComputeError(i == label);
  }

  for (int i = layers_.size() - 1; i >= 0; --i) {
    for (auto &neuron : layers_[i]) {
      neuron.UpdateNeuron();
    }
  }
}

void MatrixNetwork::Train(const std::pair<int, std::vector<double>> &data) {
  SetInputData(data.second);
  PassForward();
  BackPropagate(data.first - 1);
}

}  // namespace s21
