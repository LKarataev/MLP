#include "networks_storage.h"

#include "../graphnetwork/graphnetwork.h"
#include "../matrixnetwork/matrixnetwork.h"

namespace s21 {
NetworksStorage::NetworksStorage() {
  networks_.push_back(QMap<int, NeuralNetwork*>{{2, new MatrixNetwork(2)}});
  networks_.push_back(QMap<int, NeuralNetwork*>{{2, new GraphNetwork(2)}});
}

NetworksStorage::~NetworksStorage() {
  for (const auto& map : networks_) {
    for (const auto& network : map) {
      delete network;
    }
  }
}

void NetworksStorage::SetNetworkType(NetworkType type) { network_type_ = type; }

int NetworksStorage::GetHiddenLayersCount() const {
  return hidden_layers_count_;
}

void NetworksStorage::SetHiddenLayersCount(int hidden_layers_count) {
  hidden_layers_count_ = hidden_layers_count;
}

NeuralNetwork* NetworksStorage::RecreateNetwork() {
  QMap<int, NeuralNetwork*>::iterator iter =
      networks_[network_type_].find(hidden_layers_count_);
  delete iter.value();
  if (network_type_ == kMatrix) {
    iter.value() = new MatrixNetwork(hidden_layers_count_);
  } else {
    iter.value() = new GraphNetwork(hidden_layers_count_);
  }

  return iter.value();
}

NeuralNetwork* NetworksStorage::GetNetwork() {
  QMap<int, NeuralNetwork*>::iterator iter =
      networks_[network_type_].find(hidden_layers_count_);
  if (iter == networks_[network_type_].end()) {
    if (network_type_ == kMatrix) {
      iter = networks_[network_type_].insert(
          hidden_layers_count_, new MatrixNetwork(hidden_layers_count_));
    } else {
      iter = networks_[network_type_].insert(
          hidden_layers_count_, new GraphNetwork(hidden_layers_count_));
    }
  }

  return iter.value();
}

}  // namespace s21
