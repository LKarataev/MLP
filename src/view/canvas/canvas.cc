#include "canvas.h"

#include "qcustomplot.h"

namespace s21 {
Canvas::Canvas(QWidget *parent) : QWidget(parent) { Init(); }

// PUBLIC
void Canvas::SetImage(const QImage &image) {
  image_ =
      image.scaled(width(), height()).convertToFormat(QImage::Format_RGB16);
  is_draw_mode_ = false;
  update();
}

const QImage &Canvas::GetImage() { return image_; }

void Canvas::EnableDrawingMode() {
  is_draw_mode_ = true;
  Clear();
  update();
}

void Canvas::Init() {
  image_ = QImage(":/background_m.bmp", "bmp")
               .scaled(parentWidget()->width(), parentWidget()->width()),
  is_draw_mode_ = false;
  update();
}

void Canvas::DrawPlot(const std::vector<double> &errors) {
  QCustomPlot *plot_ = findChild<QCustomPlot *>("plot");
  QVector<double> x1(errors.size() + 1), y1(errors.size() + 1);
  for (int i = 1; i < x1.size(); ++i) {
    x1[i] = i;
    y1[i] = errors[i - 1];
  }

  // remove previous plot
  plot_->clearGraphs();
  plot_->clearPlottables();
  plot_->clearItems();

  // draw bars
  QCPBars *bars1 = new QCPBars(plot_->xAxis, plot_->yAxis);
  bars1->setData(x1, y1);
  bars1->setPen(Qt::NoPen);
  bars1->setBrush(QColor(10, 140, 70, 160));

  // draw labels of bars
  for (int i = 1; i < x1.size(); ++i) {
    QCPItemText *textLabel = new QCPItemText(plot_);
    textLabel->setText(QString::number(y1[i], 'f', 2) + "%");
    textLabel->position->setCoords(i, y1[i] + 4);
    textLabel->setFont(QFont("Sedoe UI", 9, QFont::Bold));
    textLabel->setColor(Qt::white);
  }

  // set ticks
  plot_->xAxis->setBasePen(QPen(Qt::white, 1));
  plot_->yAxis->setBasePen(QPen(Qt::white, 1));
  plot_->xAxis->setTickPen(QPen(Qt::white, 1));
  plot_->yAxis->setTickPen(QPen(Qt::white, 1));
  plot_->xAxis->setTickLabelColor(Qt::white);
  plot_->yAxis->setTickLabelColor(Qt::white);
  plot_->xAxis->setSubTicks(false);
  plot_->yAxis->setSubTicks(false);

  // set axes
  plot_->xAxis->grid()->setZeroLinePen(Qt::NoPen);
  plot_->yAxis->grid()->setZeroLinePen(Qt::NoPen);
  plot_->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
  plot_->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
  plot_->xAxis->setLabel("Epochs");
  plot_->xAxis->setLabelFont(QFont("Sedoe UI", 9, QFont::Bold));
  plot_->yAxis->setLabel("Accuracy, %");
  plot_->yAxis->setLabelFont(QFont("Sedoe UI", 9, QFont::Bold));
  plot_->xAxis->setLabelColor(Qt::white);
  plot_->yAxis->setLabelColor(Qt::white);

  plot_->setBackground(Qt::black);
  plot_->axisRect()->setBackground(Qt::black);

  plot_->rescaleAxes();
  plot_->yAxis->setRange(0, 110);
  plot_->xAxis->setRange(0.5, 5.5);
  plot_->replot();
}

// PROTECTED
void Canvas::mousePressEvent(QMouseEvent *event) {
  if (is_draw_mode_) {
    if ((current_mouse_btn_ = event->button()) == Qt::LeftButton) {
      mouse_pos_ = event->pos();
    } else {
      Clear();
      update();
    }
  }
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  if (is_draw_mode_ && current_mouse_btn_ == Qt::LeftButton) {
    QPoint end_pos = event->pos();
    QPainter painter(&image_);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(
        QPen(Qt::white, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(mouse_pos_, end_pos);
    painter.end();

    update();
    mouse_pos_ = end_pos;
  }
}

void Canvas::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  QRect rect = event->rect();
  painter.drawImage(rect, image_, rect);
}

// PRIVATE
void Canvas::Clear() {
  image_ = QImage(QSize(parentWidget()->width(), parentWidget()->height()),
                  QImage::Format_RGB16);
  image_.fill(qRgb(0, 0, 0));
}

}  // namespace s21
