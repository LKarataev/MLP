#include "controller.h"

namespace s21 {
Controller::Controller(Perceptron* pctn) : pctn_(pctn) {
  pctn_->moveToThread(&thread);
  connect(this, &Controller::LoadDataRequested, pctn_, &Perceptron::LoadData);
  connect(this, &Controller::LearningRequested, pctn_, &Perceptron::Learn);
  connect(this, &Controller::TestingRequested, pctn_, &Perceptron::Test);
  thread.start();
}

Controller::~Controller() {
  thread.quit();
  thread.wait();
}

void Controller::MatrixNetworkSelected() const {
  pctn_->SwitchNetworkType(kMatrix);
}

void Controller::GraphNetworkSelected() const {
  pctn_->SwitchNetworkType(kGraph);
}

void Controller::HiddenLayersCountChanged(int count) const {
  pctn_->SwitchHiddenLayersCount(count);
}

void Controller::WeightsLoadingRequsted(const QString& filename) const {
  pctn_->LoadWeights(filename);
}

void Controller::WeightsSavingRequsted(const QString& filename) const {
  pctn_->SaveWeights(filename);
}

void Controller::ClassificationRequested(const QImage& image) const {
  pctn_->ClassifyImage(image);
}

void Controller::ResetLearningProgressRequested() const {
  pctn_->ResetLearningProgress();
}

bool Controller::IsDataLoaded() const { return pctn_->IsDataLoaded(); }

int Controller::GetHiddenLayersCount() const {
  return pctn_->GetHiddenLayersCount();
}

const std::vector<double>& Controller::GetLearningResults() const {
  return pctn_->GetLearningResults();
}

const TestingMetrics& Controller::GetTestingResults() const {
  return pctn_->GetTestingResults();
}

const QChar& Controller::GetClassificationResult() const {
  return pctn_->GetClassificationResult();
}

}  // namespace s21
