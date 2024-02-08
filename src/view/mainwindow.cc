#include "mainwindow.h"

#include "./ui_mainwindow.h"
#include "QFileDialog"
#include "QGraphicsBlurEffect"
#include "QMessageBox"

namespace s21 {
MainWindow::MainWindow(Controller* ctrl, QWidget* parent)
    : QMainWindow(parent), ctrl_(ctrl), ui_(new Ui::MainWindow) {
  ui_->setupUi(this);
  ui_->matrixPushBtn->setCheckable(true);
  ui_->graphPushBtn->setCheckable(true);
  ui_->matrixPushBtn->setChecked(true);
  ui_->progressBarGrBox->setVisible(false);
  ui_->mainGrBox->setGraphicsEffect(new QGraphicsBlurEffect());
  ui_->mainGrBox->graphicsEffect()->setEnabled(false);
  ui_->plotGrBox->setVisible(false);
  ui_->classifyPushBtn->setEnabled(false);
}

MainWindow::~MainWindow() { delete ui_; }

// PUBLIC SLOTS
void MainWindow::HandleClassificationResult() {
  ui_->outputLabel->setText(ctrl_->GetClassificationResult());
}

void MainWindow::HandleLearningResult() {
  std::vector<double> errors = ctrl_->GetLearningResults();
  ui_->canvas->DrawPlot(errors);
  ui_->plotGrBox->setVisible(true);
  HideProgressBar();
}

void MainWindow::HandleTestingResult() {
  SetTestingResultMetrix(ctrl_->GetTestingResults());
  HideProgressBar();
}

void MainWindow::HandleError(const QString& error) {
  HideProgressBar();
  QMessageBox::critical(this, "Error", error, QMessageBox::Ok);
}

void MainWindow::HandleDataLoaded() { HideProgressBar(); }

// PRIVATE SLOTS
void MainWindow::on_matrixPushBtn_clicked() {
  if (ui_->graphPushBtn->isChecked()) {
    ctrl_->MatrixNetworkSelected();
    ToggleButtons(ui_->matrixPushBtn, ui_->graphPushBtn);
    InitWidgetsAfterNetworkChanging();
  } else {
    ui_->matrixPushBtn->setChecked(true);
  }
}

void MainWindow::on_graphPushBtn_clicked() {
  if (ui_->matrixPushBtn->isChecked()) {
    ctrl_->GraphNetworkSelected();
    ToggleButtons(ui_->graphPushBtn, ui_->matrixPushBtn);
    InitWidgetsAfterNetworkChanging();
  } else {
    ui_->graphPushBtn->setChecked(true);
  }
}

void MainWindow::on_layersCountHorSlider_sliderMoved(int position) {
  ui_->layersValueLabel->setNum(position / 100);
}

void MainWindow::on_layersCountHorSlider_valueChanged(int value) {
  int prev_layers_count = ctrl_->GetHiddenLayersCount();
  int current_layers_count = value / 100;
  if (prev_layers_count != current_layers_count) {
    ui_->layersValueLabel->setNum(current_layers_count);
    ctrl_->HiddenLayersCountChanged(current_layers_count);
    InitWidgetsAfterNetworkChanging();
  }
}

void MainWindow::on_epochsHorSlider_sliderMoved(int position) {
  ui_->epochsValueLabel->setNum(position / 100);
}

void MainWindow::on_crossValidationHorSlider_valueChanged(int value) {
  QStringList style_sheet =
      ui_->crossValidationHorSlider->styleSheet().split('}');
  QStringList style = style_sheet[2].split(';');
  if (value) {
    style[0] =
        "QSlider::handle:horizontal { background: qlineargradient(spread:pad, "
        "x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(41, 157, 105, 241), "
        "stop:0.898305 rgba(0, 0, 0, 255))";
    ui_->kGroupsGrBox->setEnabled(true);
  } else {
    style[0] = "QSlider::handle:horizontal { background: rgb(0, 0, 0)";
    ui_->kGroupsGrBox->setEnabled(false);
  }

  style_sheet[2] = style.join(';');
  ui_->crossValidationHorSlider->setStyleSheet(style_sheet.join('}'));
}

void MainWindow::on_kGroupsHorSlider_sliderMoved(int position) {
  ui_->kGroupsValueLabel->setNum(position / 100);
}

void MainWindow::on_startLearningPushBtn_clicked() {
  if (ctrl_->IsDataLoaded()) {
    ShowProgressBar("Learning...");
    if (ui_->crossValidationHorSlider->value()) {
      ctrl_->LearningRequested(ui_->epochsValueLabel->text().toInt(),
                               ui_->kGroupsValueLabel->text().toInt());
    } else {
      ctrl_->LearningRequested(ui_->epochsValueLabel->text().toInt());
    }
  } else {
    OfferToLoadData();
  }
}

void MainWindow::on_proportionHorSlider_sliderMoved(int position) {
  ui_->proportionValueLabel->setText(QString::number(position / 100.0, 'f', 2));
}

void MainWindow::on_startTestingPushBtn_clicked() {
  if (ctrl_->IsDataLoaded()) {
    ShowProgressBar("Testing...");
    ctrl_->TestingRequested(ui_->proportionValueLabel->text().toDouble());
  } else {
    OfferToLoadData();
  }
}

void MainWindow::on_saveWeightsPushBtn_clicked() {
  QString filename =
      QFileDialog::getSaveFileName(this, "Save weights", ".", "*.wgts");
  if (!filename.isEmpty()) {
    ctrl_->WeightsSavingRequsted(filename);
  }
}

void MainWindow::on_loadWeightsPushBtn_clicked() {
  QString filename =
      QFileDialog::getOpenFileName(this, "Load weights", ".", "*.wgts");
  if (!filename.isEmpty()) {
    ctrl_->WeightsLoadingRequsted(filename);
  }
}

void MainWindow::on_openBmpPushBtn_clicked() {
  QString filename =
      QFileDialog::getOpenFileName(this, "Open image", ".", "*.bmp");
  if (!filename.isEmpty()) {
    ui_->plotGrBox->setVisible(false);
    ui_->canvas->SetImage(QImage(filename, "BMP"));
    ui_->classifyPushBtn->setEnabled(true);
  }
}

void MainWindow::on_drawModePushBtn_clicked() {
  ui_->plotGrBox->setVisible(false);
  ui_->canvas->EnableDrawingMode();
  ui_->classifyPushBtn->setEnabled(true);
}

void MainWindow::on_resetPushBtn_clicked() {
  int choice =
      QMessageBox::warning(this, "Warning",
                           "This action will erase all learning progress of "
                           "the current network.\nAre you sure?",
                           QMessageBox::Ok, QMessageBox::Cancel);
  if (choice == QMessageBox::Ok) {
    ctrl_->ResetLearningProgressRequested();
    QMessageBox::information(this, "Notice",
                             "The learning progress of the current network has "
                             "been successfully erased",
                             QMessageBox::Ok);
    InitWidgetsAfterNetworkChanging();
  }
}

void MainWindow::on_classifyPushBtn_clicked() {
  ctrl_->ClassificationRequested(ui_->canvas->GetImage());
}

// PRIVATE FUNCTIONS
void MainWindow::InitWidgetsAfterNetworkChanging() {
  SetTestingResultMetrix(ctrl_->GetTestingResults());
  ui_->plotGrBox->setVisible(false);
  ui_->canvas->Init();
  ui_->outputLabel->clear();
  ui_->classifyPushBtn->setEnabled(false);
}

void MainWindow::ToggleButtons(QPushButton* active_btn,
                               QPushButton* inactive_btn) {
  QStringList active_style_sheet = active_btn->styleSheet().split('}');
  QStringList inactive_style_sheet = inactive_btn->styleSheet().split('}');
  QString temp = active_style_sheet[0];
  active_style_sheet[0] = inactive_style_sheet[0];
  inactive_style_sheet[0] = temp;
  active_btn->setStyleSheet(active_style_sheet.join('}'));
  inactive_btn->setStyleSheet(inactive_style_sheet.join('}'));
  active_btn->setChecked(true);
  inactive_btn->setChecked(false);
}

void MainWindow::SetTestingResultMetrix(const TestingMetrics& metrics) {
  ui_->avarageAccuracyValueLabel->setText(
      QString::number(metrics.accuracy, 'f', 2) + " %");
  ui_->precisionValueLabel->setText(QString::number(metrics.precision, 'f', 2));
  ui_->recallValueLabel->setText(QString::number(metrics.recall, 'f', 2));
  ui_->fmeasureValueLabel->setText(QString::number(metrics.f_measure, 'f', 2));
  ui_->timeValueLabel->setText(QString::number(metrics.total_time, 'f', 2) +
                               " sec");
}

void MainWindow::ShowProgressBar(const QString& text) {
  ui_->progressBarLabel->setText(text);
  ui_->progressBarGrBox->setVisible(true);
  ui_->mainGrBox->setDisabled(true);
  ui_->mainGrBox->graphicsEffect()->setEnabled(true);
}

void MainWindow::HideProgressBar() {
  ui_->progressBarGrBox->setVisible(false);
  ui_->mainGrBox->setDisabled(false);
  ui_->mainGrBox->graphicsEffect()->setEnabled(false);
}

void MainWindow::OfferToLoadData() {
  int choice = QMessageBox::warning(
      this, "Warning",
      "First you need to load data for learning and testing.\nLoad data?",
      QMessageBox::Yes, QMessageBox::No);
  if (choice == QMessageBox::Yes) {
    ShowProgressBar("Loading data...");
    ctrl_->LoadDataRequested();
  }
}

}  // namespace s21
