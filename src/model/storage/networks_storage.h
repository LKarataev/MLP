#ifndef SRC_MODEL_STORAGE_NETWORKS_STORAGE_H_
#define SRC_MODEL_STORAGE_NETWORKS_STORAGE_H_

#include <QMap>
#include <QVector>

#include "../interfaces/neuralnetwork.h"

namespace s21 {
enum NetworkType { kMatrix, kGraph };

class NetworksStorage {
 public:
  NetworksStorage();
  ~NetworksStorage();

  void SetNetworkType(NetworkType type);
  int GetHiddenLayersCount() const;
  void SetHiddenLayersCount(int hidden_layers_count);
  NeuralNetwork* RecreateNetwork();
  NeuralNetwork* GetNetwork();

 private:
  NetworkType network_type_ = kMatrix;
  int hidden_layers_count_ = 2;
  QVector<QMap<int, NeuralNetwork*>> networks_;
};
}  // namespace s21

#endif  // SRC_MODEL_STORAGE_NETWORKS_STORAGE_H_
