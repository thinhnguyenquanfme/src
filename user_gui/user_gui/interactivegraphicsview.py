from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand
from PyQt6.QtCore import Qt, QRect, QRectF, QPoint, QSize
from PyQt6.QtGui import QPixmap, QImage

class InteractiveGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # scene + one persistent pixmap item (no reset on new frames)
        scn = QGraphicsScene(self)
        self.setScene(scn)
        self._item = QGraphicsPixmapItem()
        scn.addItem(self._item)

        # keep original image for exact-pixel crop
        self._qimage: QImage | None = None

        # view behavior
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 0

        # ROI rubber band
        self._rb = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rb_origin = QPoint()
        self._dragging = False

        # optional panning state
        self._panning = False
        self._pan_start = QPoint()

    # ---------- public API ----------
    def setPixmap(self, pix: QPixmap, fit_first: bool = False):
        """Update the displayed image. Set fit_first=True only once (first frame or when user requests reset)."""
        self._item.setPixmap(pix)
        self._qimage = pix.toImage()
        if fit_first and not self._item.boundingRect().isEmpty():
            self.fitInView(self._item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clearRoi(self):
        self._rb.hide()

    def resetZoom(self):
        self.resetTransform()
        self._zoom = 0

    def fitToImage(self):
        if not self._item.boundingRect().isEmpty():
            self.fitInView(self._item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def get_roi_rect_pixels(self) -> QRect | None:
        """Return ROI rect in image pixel coords (QRect), or None if invalid."""
        if self._qimage is None or self._rb.width() == 0 or self._rb.height() == 0:
            return None

        # viewport -> scene
        vrect: QRect = self._rb.geometry()
        spoly = self.mapToScene(vrect)
        srect = QRectF(spoly[0], spoly[2]).normalized()

        # scene -> item (image pixel coords)
        irectf = self._item.mapFromScene(srect).boundingRect()
        irect = irectf.toRect()

        # clamp to image bounds
        bounds = QRect(0, 0, self._qimage.width(), self._qimage.height())
        irect = irect.intersected(bounds)
        return irect if not irect.isEmpty() else None

    def get_roi_qimage(self) -> QImage | None:
        r = self.get_roi_rect_pixels()
        if r is None or self._qimage is None:
            return None
        return self._qimage.copy(r)

    def get_roi_qpixmap(self) -> QPixmap | None:
        qi = self.get_roi_qimage()
        return QPixmap.fromImage(qi) if qi is not None else None

    # ---------- interactions ----------
    # wheel zoom
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 1/1.25
        self.scale(factor, factor)
        self._zoom += 1 if factor > 1 else -1

    # left mouse: draw ROI
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._rb_origin = event.pos()
            self._rb.setGeometry(QRect(self._rb_origin, QSize()))
            self._rb.show()
        elif event.button() == Qt.MouseButton.MiddleButton:
            # optional middle-button pan
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            rect = QRect(self._rb_origin, event.pos()).normalized()
            self._rb.setGeometry(rect)
        elif self._panning:
            delta = self.mapToScene(event.pos()) - self.mapToScene(self._pan_start)
            self._pan_start = event.pos()
            self.translate(-delta.x(), -delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            # keep rubber band visible; call clearRoi() if you want to hide it
        elif event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # double-click to fit image quickly
        self.fitToImage()
        self._zoom = 0
        super().mouseDoubleClickEvent(event)
