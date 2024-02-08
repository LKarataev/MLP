#include "graphneuron.h"

namespace s21 {

GraphNeuron::GraphNeuron() {}

GraphNeuron::~GraphNeuron() {}

std::vector<double>* GraphNeuron::GetWeights() { return &weights_; }

void GraphNeuron::SetError(double value) { error_ = value; }

void GraphNeuron::AddValueToError(double value) { error_ += value; }

void GraphNeuron::AddInputNeuron(GraphNeuron* input) {
  inputs_.push_back(input);
}

void GraphNeuron::ComputeOutput() {
  double sum = 0.0;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    sum += inputs_[i]->GetOutput() * weights_[i];
  }
  output_ = utils::ActivateFunction(sum);
}

void GraphNeuron::SetOutput(double value) { output_ = value; }

double GraphNeuron::GetOutput() { return output_; }

void GraphNeuron::SetRandomWeights() {
  weights_.reserve(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    weights_.emplace_back(utils::RandomDouble());
  }
}

void GraphNeuron::UpdateNeuron(bool compute_previous_error, bool compute_error,
                               bool right_answer) {
  if (compute_error) {
    ComputeError(right_answer);
  }
  ComputeWeightsDelta();
  UpdateWeight();
  if (compute_previous_error) {
    UpdatePreviousLayerErrors();
  }
}

void GraphNeuron::ComputeError(double right_answer) {
  error_ = output_ - right_answer;
}

void GraphNeuron::ComputeWeightsDelta() {
  weights_delta_ = error_ * output_ * (1.0 - output_);
}

void GraphNeuron::UpdateWeight() {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    weights_[i] -=
        inputs_[i]->GetOutput() * weights_delta_ * utils::kLearningRate;
  }
}

void GraphNeuron::UpdatePreviousLayerErrors() {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->AddValueToError(weights_[i] * weights_delta_);
  }
}

}  // namespace s21
