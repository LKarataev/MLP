#include "perceptron.h"

namespace s21 {
Perceptron::Perceptron() : network_(storage_.GetNetwork()) {}

// PUBLIC FUNCTIONS
void Perceptron::SwitchNetworkType(NetworkType type) {
  storage_.SetNetworkType(type);
  network_ = storage_.GetNetwork();
  ResetNetworkResults();
}

void Perceptron::SwitchHiddenLayersCount(int hidden_layers_count) {
  storage_.SetHiddenLayersCount(hidden_layers_count);
  network_ = storage_.GetNetwork();
  ResetNetworkResults();
}

void Perceptron::ClassifyImage(const QImage& image) {
  std::vector<double> data(GetDataFromImage(image));
  output_ = QChar(network_->Classify(data) + 64);
  emit NotifyClassificationEnds();
}

void Perceptron::LoadWeights(const QString& path) {
  try {
    network_->LoadWeights(path.toStdString());
  } catch (const std::invalid_argument& e) {
    emit NotifyError(QString::fromStdString(e.what()));
  } catch (const std::out_of_range& e) {
    emit NotifyError(QString::fromStdString(e.what()));
  }
}

void Perceptron::SaveWeights(const QString& path) {
  try {
    network_->SaveWeights(path.toStdString());
  } catch (const std::invalid_argument& e) {
    emit NotifyError(QString::fromStdString(e.what()));
  }
}

void Perceptron::ResetLearningProgress() {
  network_ = storage_.RecreateNetwork();
  ResetNetworkResults();
}

const std::vector<double>& Perceptron::GetLearningResults() const {
  return metrics_.accuracy_per_epoch;
}

const TestingMetrics& Perceptron::GetTestingResults() const {
  return metrics_.t_metrics;
}

const QChar& Perceptron::GetClassificationResult() const { return output_; }

int Perceptron::GetHiddenLayersCount() const {
  return storage_.GetHiddenLayersCount();
}

bool Perceptron::IsDataLoaded() const {
  return ds_.train.size() != 0 && ds_.test.size() != 0;
}

// PUBLIC SLOTS
void Perceptron::Learn(int epochs_count, int k_groups_count) {
  metrics_.accuracy_per_epoch.clear();

  if (!k_groups_count) {
    LearnEpochs(epochs_count);
  } else {
    LearnCross(epochs_count, k_groups_count);
  }

  emit NotifyLearningEnds();
}

void Perceptron::Test(double sample_proportion) {
  if (sample_proportion == 0) {
    metrics_.t_metrics = TestingMetrics{};
    emit NotifyTestingEnds();
    return;
  }

  auto start_time = std::chrono::steady_clock::now();
  int count_right_answers = 0;
  std::vector<int> true_positives(utils::kOutputSize, 0);
  std::vector<int> false_positives(utils::kOutputSize, 0);
  std::vector<int> false_negatives(utils::kOutputSize, 0);
  int test_size = static_cast<int>(sample_proportion * ds_.test.size());

  for (int i = 0; i < test_size; ++i) {
    int true_letter = ds_.test.at(i).first - 1;
    int predicted_letter = network_->Classify(ds_.test.at(i).second) - 1;
    if (true_letter == predicted_letter) {
      ++count_right_answers;
      ++true_positives[true_letter];
    } else {
      ++false_positives[predicted_letter];
      ++false_negatives[true_letter];
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_precision = 0;
  double total_recall = 0;
  double total_f_measure = 0;
  for (int i = 0; i < utils::kOutputSize; ++i) {
    double precision = utils::Proportion(true_positives[i], false_positives[i]);
    double recall = utils::Proportion(true_positives[i], false_negatives[i]);
    total_precision += precision;
    total_recall += recall;
    total_f_measure += utils::HarmonicMean(precision, recall);
  }

  metrics_.t_metrics.accuracy =
      static_cast<double>(count_right_answers) / test_size * 100;
  metrics_.t_metrics.precision = total_precision / utils::kOutputSize;
  metrics_.t_metrics.recall = total_recall / utils::kOutputSize;
  metrics_.t_metrics.f_measure = total_f_measure / utils::kOutputSize;
  metrics_.t_metrics.total_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count() /
      1000.0;
  emit NotifyTestingEnds();
}

void Perceptron::LoadData() {
  QString path = QCoreApplication::applicationDirPath() + "/../../";
#ifdef Q_OS_MACOS
  path += QString("../../../");
#endif
  path += QString("datasets/emnist-letters/");
  path = QDir::toNativeSeparators(path);
  try {
    ds_.train = LoadDataFromFile(
        (path + QString("emnist-letters-train.csv")).toStdString());
    ds_.test = LoadDataFromFile(
        (path + QString("emnist-letters-test.csv")).toStdString());
    emit NotifyDataLoaded();
  } catch (const std::invalid_argument& e) {
    emit NotifyError(QString::fromStdString(e.what()));
  }
}

// PRIVATE
std::pair<int, std::vector<double>> Perceptron::ParseLine(
    const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  std::getline(iss, token, ',');
  int letter = std::stoi(token);
  std::vector<double> data(utils::kInputSize);
  for (int i = 0; i < utils::kInputSize; ++i) {
    std::getline(iss, token, ',');
    data[i] = std::stod(token) / 255.0;
  }
  return {letter, data};
}

std::vector<std::pair<int, std::vector<double>>> Perceptron::LoadDataFromFile(
    const std::string& filename) {
  std::ifstream file(filename);
  std::vector<std::pair<int, std::vector<double>>> result;
  if (!file.is_open()) {
    throw std::invalid_argument("Failed to open file:\n" + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    result.emplace_back(ParseLine(line));
  }

  file.close();
  return result;
}

void Perceptron::LearnEpochs(int epochs_count) {
  for (int i = 0; i < epochs_count; ++i) {
    for (const auto& data : ds_.train) {
      network_->Train(data);
    }

    double count_right_answers =
        std::count_if(ds_.test.begin(), ds_.test.end(),
                      [this](std::pair<int, std::vector<double>> i) {
                        return i.first == network_->Classify(i.second);
                      });

    metrics_.accuracy_per_epoch.push_back(count_right_answers /
                                          ds_.test.size() * 100);
  }
}

void Perceptron::LearnCross(int epochs_count, int k_groups_count) {
  int group_size = ds_.train.size() / k_groups_count;
  for (int j = 0; j < epochs_count; ++j) {
    double average_accuracy = 0;
    for (int i = 0; i < k_groups_count; ++i) {
      auto start = i * group_size;
      auto end = (i + 1) * group_size;

      std::vector<std::pair<int, std::vector<double>>> test_data(
          ds_.train.begin() + start, ds_.train.begin() + end);

      std::vector<std::pair<int, std::vector<double>>> train_data = ds_.train;
      train_data.erase(train_data.begin() + start, train_data.begin() + end);

      for (const auto& data : train_data) {
        network_->Train(data);
      }

      double count_right_answers =
          std::count_if(test_data.begin(), test_data.end(),
                        [this](std::pair<int, std::vector<double>> i) {
                          return i.first == network_->Classify(i.second);
                        });

      average_accuracy += count_right_answers / test_data.size() * 100;
    }

    metrics_.accuracy_per_epoch.push_back(average_accuracy / k_groups_count);
  }
}

std::vector<double> Perceptron::GetDataFromImage(const QImage& image) {
  QImage transformed = image.transformed(QTransform().rotate(90).scale(1, -1));
  transformed = transformed.scaled(utils::kImgSize, utils::kImgSize);
  std::vector<double> result(utils::kInputSize);
  for (int y = 0; y < utils::kImgSize; ++y) {
    for (int x = 0; x < utils::kImgSize; ++x) {
      QColor color = transformed.pixelColor(x, y);
      double gray_value = (color.redF() + color.greenF() + color.blueF()) / 3.0;
      result[y * utils::kImgSize + x] = gray_value;
    }
  }

  return result;
}

void Perceptron::ResetNetworkResults() {
  metrics_.t_metrics = TestingMetrics{};
  metrics_.accuracy_per_epoch.clear();
  output_ = '\0';
}

}  // namespace s21
