#ifndef VIEW_MAINWINDOW_H
#define VIEW_MAINWINDOW_H

#include <QMainWindow>

#include "../controller/controller.h"
#include "qpushbutton.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

namespace s21 {
class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(Controller* ctrl, QWidget* parent = nullptr);
  ~MainWindow();

 public slots:
  void HandleClassificationResult();
  void HandleLearningResult();
  void HandleTestingResult();
  void HandleError(const QString& error);
  void HandleDataLoaded();

 private slots:
  void on_matrixPushBtn_clicked();
  void on_graphPushBtn_clicked();

  void on_layersCountHorSlider_sliderMoved(int position);
  void on_layersCountHorSlider_valueChanged(int value);

  void on_epochsHorSlider_sliderMoved(int position);
  void on_crossValidationHorSlider_valueChanged(int value);
  void on_kGroupsHorSlider_sliderMoved(int position);
  void on_startLearningPushBtn_clicked();

  void on_proportionHorSlider_sliderMoved(int position);
  void on_startTestingPushBtn_clicked();

  void on_saveWeightsPushBtn_clicked();
  void on_loadWeightsPushBtn_clicked();

  void on_openBmpPushBtn_clicked();
  void on_drawModePushBtn_clicked();

  void on_resetPushBtn_clicked();
  void on_classifyPushBtn_clicked();

 private:
  void InitWidgetsAfterNetworkChanging();
  void ToggleButtons(QPushButton* active_btn, QPushButton* inactive_btn);
  void SetTestingResultMetrix(const TestingMetrics& metrics);
  void ShowProgressBar(const QString& text);
  void HideProgressBar();
  void OfferToLoadData();

  Controller* ctrl_;
  Ui::MainWindow* ui_;
};
}  // namespace s21
#endif  // MAINWINDOW_H
