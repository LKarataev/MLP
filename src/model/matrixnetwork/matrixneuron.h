#ifndef SRC_MODEL_MATRIXNETWORK_MATRIXNEURON_H_
#define SRC_MODEL_MATRIXNETWORK_MATRIXNEURON_H_

#include "../interfaces/neuron.h"
#include "../utils/utils.h"
namespace s21 {

class MatrixNeuron : public Neuron {
 public:
  explicit MatrixNeuron(std::vector<MatrixNeuron>* previous_layer);

  void SetOutput(double value) override;
  void SetError(double value) override;
  void ComputeError(double right_answer) override;
  void AddValueToError(double value) override;
  void ComputeOutput() override;
  double GetOutput() override;
  void UpdateNeuron();
  std::vector<double>* GetWeights();

 private:
  void ComputeWeightsDelta() override;
  void UpdateWeight();
  void UpdatePreviousLayerErrors();
  void SetRandomWeights();

  double output_ = 0.0;
  double error_ = 0.0;
  double weights_delta_ = 0.0;

  std::vector<MatrixNeuron>* previous_layer_ = nullptr;
  std::vector<double> weights_;
};

}  // namespace s21

#endif  // SRC_MODEL_MATRIXNETWORK_MATRIXNEURON_H_
