#ifndef SRC_MODEL_GRAPHNETWORK_GRAPHNEURON_H_
#define SRC_MODEL_GRAPHNETWORK_GRAPHNEURON_H_

#include <vector>

#include "../interfaces/neuron.h"
#include "../utils/utils.h"

namespace s21 {

class GraphNeuron : public Neuron {
 public:
  GraphNeuron();
  explicit GraphNeuron(double value);
  ~GraphNeuron();

  void AddInputNeuron(GraphNeuron* input);
  void SetRandomWeights();
  std::vector<double>* GetWeights();
  void SetOutput(double value) override;
  void SetError(double value) override;
  void AddValueToError(double value) override;
  void ComputeOutput() override;
  double GetOutput() override;
  void UpdateNeuron(bool compute_previous_error = false,
                    bool compute_error = false, bool right_answer = false);

 private:
  void ComputeError(double right_answer) override;
  void ComputeWeightsDelta() override;
  void UpdateWeight();
  void UpdatePreviousLayerErrors();

  double output_ = 0.0;
  double error_ = 0.0;
  double weights_delta_ = 0.0;
  std::vector<GraphNeuron*> inputs_;
  std::vector<double> weights_;
};

}  // namespace s21

#endif  // SRC_MODEL_GRAPHNETWORK_GRAPHNEURON_H_
