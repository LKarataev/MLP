#include <QApplication>

#include "model/perceptron.h"
#include "view/mainwindow.h"

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  s21::Perceptron pctn;
  s21::Controller ctrl(&pctn);
  s21::MainWindow mw(&ctrl);

  app.connect(&pctn, &s21::Perceptron::NotifyDataLoaded, &mw,
              &s21::MainWindow::HandleDataLoaded);
  app.connect(&pctn, &s21::Perceptron::NotifyClassificationEnds, &mw,
              &s21::MainWindow::HandleClassificationResult);
  app.connect(&pctn, &s21::Perceptron::NotifyLearningEnds, &mw,
              &s21::MainWindow::HandleLearningResult);
  app.connect(&pctn, &s21::Perceptron::NotifyTestingEnds, &mw,
              &s21::MainWindow::HandleTestingResult);
  app.connect(&pctn, &s21::Perceptron::NotifyError, &mw,
              &s21::MainWindow::HandleError);

  mw.show();
  return app.exec();
}
