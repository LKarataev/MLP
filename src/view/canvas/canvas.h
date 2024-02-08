#ifndef VIEW_CANVAS_H
#define VIEW_CANVAS_H

#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QWidget>

namespace s21 {
class Canvas : public QWidget {
  Q_OBJECT

 public:
  explicit Canvas(QWidget* parent = nullptr);

  void SetImage(const QImage& image);
  const QImage& GetImage();
  void EnableDrawingMode();
  void Init();
  void DrawPlot(const std::vector<double>& errors);

 protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

 private:
  void Clear();

  Qt::MouseButton current_mouse_btn_;
  QPoint mouse_pos_;
  bool is_draw_mode_ = false;
  QImage image_;
};
}  // namespace s21

#endif  // VIEW_CANVAS_H
