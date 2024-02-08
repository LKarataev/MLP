#include "matrixneuron.h"

namespace s21 {
MatrixNeuron::MatrixNeuron(std::vector<MatrixNeuron>* previous_layer)
    : previous_layer_{previous_layer} {
  SetRandomWeights();
}

std::vector<double>* MatrixNeuron::GetWeights() { return &weights_; }

void MatrixNeuron::SetOutput(double value) { output_ = value; }

void MatrixNeuron::SetError(double value) { error_ = value; }

void MatrixNeuron::AddValueToError(double value) { error_ += value; }

void MatrixNeuron::ComputeOutput() {
  if (previous_layer_ != nullptr) {
    double sum = 0.0;
    for (size_t i = 0; i < previous_layer_->size(); ++i) {
      sum += (*previous_layer_)[i].GetOutput() * weights_[i];
    }
    output_ = utils::ActivateFunction(sum);
  }
}

double MatrixNeuron::GetOutput() { return output_; }

void MatrixNeuron::SetRandomWeights() {
  if (previous_layer_ != nullptr) {
    weights_.reserve(previous_layer_->size());
    for (size_t i = 0; i < previous_layer_->size(); ++i) {
      weights_.emplace_back(utils::RandomDouble());
    }
  }
}

void MatrixNeuron::ComputeError(double right_answer) {
  error_ = output_ - right_answer;
}

void MatrixNeuron::ComputeWeightsDelta() {
  weights_delta_ = error_ * output_ * (1.0 - output_);
}

void MatrixNeuron::UpdateWeight() {
  if (previous_layer_ != nullptr) {
    for (size_t i = 0; i < previous_layer_->size(); ++i) {
      weights_[i] -= (*previous_layer_)[i].GetOutput() * weights_delta_ *
                     utils::kLearningRate;
    }
  }
}

void MatrixNeuron::UpdatePreviousLayerErrors() {
  if (previous_layer_ != nullptr) {
    for (size_t i = 0; i < previous_layer_->size(); ++i) {
      (*previous_layer_)[i].AddValueToError(weights_[i] * weights_delta_);
    }
  }
}

void MatrixNeuron::UpdateNeuron() {
  ComputeWeightsDelta();
  UpdateWeight();
  UpdatePreviousLayerErrors();
}

}  // namespace s21
