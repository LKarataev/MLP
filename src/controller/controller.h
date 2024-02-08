#ifndef CONTROLLER_CONTROLLER_H_
#define CONTROLLER_CONTROLLER_H_

#include <QThread>

#include "../model/perceptron.h"

namespace s21 {
class Controller : public QObject {
  Q_OBJECT

 public:
  explicit Controller(Perceptron* pctn);
  ~Controller();

  void MatrixNetworkSelected() const;
  void GraphNetworkSelected() const;
  void HiddenLayersCountChanged(int count) const;
  void WeightsLoadingRequsted(const QString& filename) const;
  void WeightsSavingRequsted(const QString& filename) const;
  void ClassificationRequested(const QImage& image) const;
  void ResetLearningProgressRequested() const;

  bool IsDataLoaded() const;
  int GetHiddenLayersCount() const;

  const std::vector<double>& GetLearningResults() const;
  const TestingMetrics& GetTestingResults() const;
  const QChar& GetClassificationResult() const;

 signals:
  void LearningRequested(int epochs_count, int k_groups_count = 0);
  void TestingRequested(double sample_proportion);
  void LoadDataRequested();

 private:
  Perceptron* pctn_;
  QThread thread;
};
}  // namespace s21

#endif  // CONTROLLER_CONTROLLER_H_
