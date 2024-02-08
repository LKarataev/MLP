#ifndef SRC_MODEL_INTERFACES_NEURON_H_
#define SRC_MODEL_INTERFACES_NEURON_H_

#include <vector>

namespace s21 {

class Neuron {
 public:
  virtual ~Neuron() = default;

  virtual void SetOutput(double value) = 0;
  virtual void SetError(double value) = 0;
  virtual void AddValueToError(double value) = 0;
  virtual void ComputeOutput() = 0;
  virtual double GetOutput() = 0;

 private:
  virtual void ComputeError(double right_answer) = 0;
  virtual void ComputeWeightsDelta() = 0;
};

}  // namespace s21

#endif  // SRC_MODEL_INTERFACES_NEURON_H_
