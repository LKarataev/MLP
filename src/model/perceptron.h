#ifndef SRC_MODEL_PERCEPTRON_H_
#define SRC_MODEL_PERCEPTRON_H_

#include <QCoreApplication>
#include <QDir>
#include <QImage>
#include <QPair>
#include <QString>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "./storage/networks_storage.h"
#include "./utils/utils.h"

namespace s21 {

struct TestingMetrics {
  double accuracy;
  double precision;
  double recall;
  double f_measure;
  double total_time;
};

struct Metrics {
  std::vector<double> accuracy_per_epoch;
  TestingMetrics t_metrics;
};

struct DataSet {
  std::vector<std::pair<int, std::vector<double>>> train;
  std::vector<std::pair<int, std::vector<double>>> test;
};

class Perceptron : public QObject {
  Q_OBJECT

 public:
  Perceptron();

  void SwitchNetworkType(NetworkType type);
  void SwitchHiddenLayersCount(int hidden_layers_count);
  void ClassifyImage(const QImage& image);
  void LoadWeights(const QString& path);
  void SaveWeights(const QString& path);
  void ResetLearningProgress();

  const std::vector<double>& GetLearningResults() const;
  const TestingMetrics& GetTestingResults() const;
  const QChar& GetClassificationResult() const;
  int GetHiddenLayersCount() const;
  bool IsDataLoaded() const;

 public slots:
  void Learn(int epochs_count, int k_groups_count);
  void Test(double sample_proportion);
  void LoadData();

 signals:
  void NotifyDataLoaded();
  void NotifyError(const QString& error);
  void NotifyTestingEnds();
  void NotifyLearningEnds();
  void NotifyClassificationEnds();

 private:
  std::pair<int, std::vector<double>> ParseLine(const std::string& line);
  std::vector<std::pair<int, std::vector<double>>> LoadDataFromFile(
      const std::string& filename);
  void LearnEpochs(int epochs_count);
  void LearnCross(int epochs_count, int k_groups_count);
  std::vector<double> GetDataFromImage(const QImage& image);
  void ResetNetworkResults();

  NetworksStorage storage_;
  NeuralNetwork* network_;
  Metrics metrics_ = {{0, 0}, {0, 0, 0, 0, 0}};
  QChar output_;
  DataSet ds_;
};

}  // namespace s21

#endif  // SRC_MODEL_PERCEPTRON_H_
