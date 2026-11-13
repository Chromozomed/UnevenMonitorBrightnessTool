"""
Screen Uniformity - darken parts of the screen to even out an uneven monitor.

Install:   pip install PyQt6 numpy
Run:       python screen_uniformity.py        (use pythonw.exe to run without a console)

How it works:
  * A click-through, always-on-top, transparent window covers each monitor.
  * It paints black with a per-pixel alpha, built from the areas you draw.
  * Each area has a "darkness" (% reduction in light output) and a "feather"
    (soft edge, measured inward from the border, in pixels).
  * Two area types: "flat" (uniform darkening) and "linear gradient" (darkness
    fades from a start % to an end % along a chosen direction).
  * Darkness is applied in linear light, so 20% really means ~20% less light,
    not 20% less pixel value.

Settings are saved automatically to %APPDATA%\\ScreenUniformity\\config.json

Fullscreen notes:
  * Works over games in borderless / windowed-fullscreen mode (and over most
    "fullscreen optimised" games). It cannot draw over true exclusive-fullscreen.
  * The overlay is deliberately 1 px shorter than the monitor. A topmost window
    that exactly covers a monitor is treated by Windows as a fullscreen app, which
    confuses the taskbar's hide-for-fullscreen logic.
"""
import json
import math
import os
import sys

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF, QRectF

GAMMA = 2.2
CONFIG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "ScreenUniformity")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_STRENGTH = 15   # percent
DEFAULT_FEATHER = 60    # logical pixels


OVERLAY_GAP_PX = 1      # overlay is this many px shorter than the monitor (bottom edge)
KEEP_ON_TOP_MS = 2000   # re-assert "topmost" this often while idle; 0 disables

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, wintypes.UINT]
    _user32.SetWindowPos.restype = wintypes.BOOL
    _HWND_TOPMOST = wintypes.HWND(-1)
    _SWP_FLAGS = 0x0001 | 0x0002 | 0x0010 | 0x0200   # NOSIZE | NOMOVE | NOACTIVATE | NOOWNERZORDER

    def force_topmost(widget):
        """Move the window to the top of the topmost band without activating it."""
        _user32.SetWindowPos(wintypes.HWND(int(widget.winId())), _HWND_TOPMOST, 0, 0, 0, 0, _SWP_FLAGS)
else:
    def force_topmost(widget):
        widget.raise_()


def new_region(kind="flat"):
    grad = kind == "gradient"
    return {
        "type": kind,
        "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0,
        "strength": float(DEFAULT_STRENGTH),
        "strength_end": 0.0 if grad else float(DEFAULT_STRENGTH),
        "angle": 0.0,                              # 0 = left->right, 90 = top->bottom
        "feather": 0.0 if grad else float(DEFAULT_FEATHER),
    }


def parse_region(r):
    strength = float(r.get("strength", DEFAULT_STRENGTH))
    kind = r.get("type", "flat")
    return {
        "type": kind if kind in ("flat", "gradient") else "flat",
        "x": float(r["x"]), "y": float(r["y"]),
        "w": float(r["w"]), "h": float(r["h"]),
        "strength": strength,
        "strength_end": float(r.get("strength_end", strength)),
        "angle": float(r.get("angle", 0.0)),
        "feather": float(r.get("feather", DEFAULT_FEATHER)),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Model(QtCore.QObject):
    changed = QtCore.pyqtSignal()             # regions or enabled flag changed
    selection_changed = QtCore.pyqtSignal()
    enabled_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.data = {}       # screen name -> list of region dicts (normalised 0..1 coords)
        self.selected = -1
        self.new_type = "flat"     # type given to newly drawn areas
        self.load()

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save)
        self.changed.connect(lambda: self._save_timer.start(600))

    def regions(self, key):
        return self.data.setdefault(key, [])

    def select(self, index):
        if index != self.selected:
            self.selected = index
            self.selection_changed.emit()

    def set_enabled(self, value):
        if value != self.enabled:
            self.enabled = value
            self.enabled_changed.emit()
            self.changed.emit()

    def load(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.enabled = bool(cfg.get("enabled", True))
            for key, regs in cfg.get("screens", {}).items():
                self.data[key] = [parse_region(r) for r in regs]
        except FileNotFoundError:
            pass
        except Exception as exc:  # corrupt config: start fresh but keep the file
            print("Could not load config:", exc)

    def save(self):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"enabled": self.enabled, "screens": self.data}, f, indent=2)
        except Exception as exc:
            print("Could not save config:", exc)


# --------------------------------------------------------------------------- #
# Rendering the darkening mask
# --------------------------------------------------------------------------- #
_noise_cache = {}


def _noise(h, w):
    """Static dither noise in [-0.5, 0.5) to hide banding."""
    key = (h, w)
    if key not in _noise_cache:
        _noise_cache.clear()
        rng = np.random.default_rng(1)
        _noise_cache[key] = rng.random((h, w), dtype=np.float32) - 0.5
    return _noise_cache[key]


def render_overlay(regions, W, H, scale):
    """
    regions: list of region dicts; W,H: physical pixel size of the screen;
    scale: physical px per logical px.
    Returns (QImage, (x, y, w, h)) in physical pixels, or None if nothing to draw.
    Only the bounding box of the regions is rendered, which keeps dragging fast.
    """
    boxes = []
    for r in regions:
        peak = r["strength"]
        if r.get("type") == "gradient":
            peak = max(peak, r.get("strength_end", peak))
        if peak <= 0:
            continue
        x0 = max(0, int(round(r["x"] * W)))
        y0 = max(0, int(round(r["y"] * H)))
        x1 = min(W, int(round((r["x"] + r["w"]) * W)))
        y1 = min(H, int(round((r["y"] + r["h"]) * H)))
        if x1 - x0 >= 1 and y1 - y0 >= 1:
            boxes.append((x0, y0, x1, y1, r))
    if not boxes:
        return None

    bx0 = min(b[0] for b in boxes)
    by0 = min(b[1] for b in boxes)
    bx1 = max(b[2] for b in boxes)
    by1 = max(b[3] for b in boxes)
    bw, bh = bx1 - bx0, by1 - by0

    # T = remaining linear light (1.0 = untouched)
    T = np.ones((bh, bw), np.float32)
    for x0, y0, x1, y1, r in boxes:
        f = r["feather"] * scale
        w, h = x1 - x0, y1 - y0
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5

        if r.get("type") == "gradient":
            # t runs 0 -> 1 along the chosen direction across the rectangle
            ang = math.radians(r.get("angle", 0.0))
            dx, dy = math.cos(ang), math.sin(ang)
            length = max(abs(w * dx) + abs(h * dy), 1.0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            t = ((xs - cx) * dx)[None, :] + ((ys - cy) * dy)[:, None]
            t = np.clip(t / length + 0.5, 0.0, 1.0)
            s0 = r["strength"] / 100.0
            s1 = r.get("strength_end", r["strength"]) / 100.0
            s = s0 + (s1 - s0) * t
        else:
            s = r["strength"] / 100.0

        if f > 0.5:
            rx = np.clip(np.minimum(xs - x0, x1 - xs) / f, 0.0, 1.0)
            ry = np.clip(np.minimum(ys - y0, y1 - ys) / f, 0.0, 1.0)
            rx = rx * rx * (3 - 2 * rx)   # smoothstep
            ry = ry * ry * (3 - 2 * ry)
            s = s * np.outer(ry, rx)
        T[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] *= (1.0 - s)

    # A black overlay with alpha a gives encoded value v*(1-a); light ~ v^2.2,
    # so for a linear gain T we need a = 1 - T^(1/2.2).
    alpha = (1.0 - np.power(np.clip(T, 0.0, 1.0), 1.0 / GAMMA)) * 255.0
    alpha += _noise(bh, bw)
    a8 = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)

    rgba = np.zeros((bh, bw, 4), np.uint8)   # black, premultiplied => only alpha matters
    rgba[..., 3] = a8
    img = QtGui.QImage(rgba.data, bw, bh, bw * 4,
                       QtGui.QImage.Format.Format_RGBA8888_Premultiplied).copy()
    return img, (bx0, by0, bw, bh)


# --------------------------------------------------------------------------- #
# Overlay window (click-through)
# --------------------------------------------------------------------------- #
class Overlay(QtWidgets.QWidget):
    def __init__(self, screen, model):
        super().__init__()
        self.scr = screen
        self.model = model
        self.key = screen.name()
        self.result = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(self._target_geometry())

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.rebuild)
        model.changed.connect(lambda: self._timer.start(20))
        screen.geometryChanged.connect(self._geometry_changed)

        self.rebuild()
        self.show()
        force_topmost(self)

    def _target_geometry(self):
        # 1 px shorter than the monitor so Windows doesn't classify this as a fullscreen window.
        return self.scr.geometry().adjusted(0, 0, 0, -OVERLAY_GAP_PX)

    def _geometry_changed(self, _):
        target = self._target_geometry()
        if self.geometry() != target:
            self.setGeometry(target)
        self.rebuild()

    def rebuild(self):
        geo = self.scr.geometry()
        dpr = self.scr.devicePixelRatio()
        W, H = int(round(geo.width() * dpr)), int(round(geo.height() * dpr))
        regions = self.model.regions(self.key) if self.model.enabled else []
        res = render_overlay(regions, W, H, dpr)
        if res is None:
            self.result = None
        else:
            img, (x, y, w, h) = res
            self.result = (img, QRectF(x / dpr, y / dpr, w / dpr, h / dpr))
        self.update()

    def paintEvent(self, _):
        if self.result:
            p = QtGui.QPainter(self)
            img, target = self.result
            p.drawImage(target, img)


# --------------------------------------------------------------------------- #
# Editor window (draw / move / resize areas)
# --------------------------------------------------------------------------- #
class Editor(QtWidgets.QWidget):
    HANDLE = 6

    def __init__(self, screen, model):
        super().__init__()
        self.scr = screen
        self.model = model
        self.key = screen.name()
        self.drag = None

        # NoActivate so clicking here never lifts this window above the settings panel.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(screen.geometry())
        model.selection_changed.connect(self.update)
        model.changed.connect(self.update)

    def regs(self):
        return self.model.regions(self.key)

    def rect_of(self, r):
        W, H = self.width(), self.height()
        return QRectF(r["x"] * W, r["y"] * H, r["w"] * W, r["h"] * H)

    def set_rect(self, index, rect):
        W, H = self.width(), self.height()
        rect = rect.intersected(QRectF(0, 0, W, H))
        r = self.regs()[index]
        r["x"], r["y"] = rect.x() / W, rect.y() / H
        r["w"], r["h"] = rect.width() / W, rect.height() / H

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = e.position()
        regs = self.regs()
        sel = self.model.selected

        if 0 <= sel < len(regs):
            rect = self.rect_of(regs[sel])
            corners = {"tl": rect.topLeft(), "tr": rect.topRight(),
                       "bl": rect.bottomLeft(), "br": rect.bottomRight()}
            for name, pt in corners.items():
                if abs(p.x() - pt.x()) <= self.HANDLE + 4 and abs(p.y() - pt.y()) <= self.HANDLE + 4:
                    self.drag = ("resize", name, QRectF(rect))
                    return

        for i in reversed(range(len(regs))):
            rect = self.rect_of(regs[i])
            if rect.contains(p):
                self.model.select(i)
                self.drag = ("move", QPointF(p), QRectF(rect), i)
                return

        regs.append(new_region(self.model.new_type))
        self.model.select(len(regs) - 1)
        self.drag = ("new", QPointF(p), len(regs) - 1)

    def mouseMoveEvent(self, e):
        if not self.drag:
            return
        p = e.position()
        W, H = self.width(), self.height()
        kind = self.drag[0]
        if kind == "new":
            _, start, idx = self.drag
            rect = QRectF(start, p).normalized()
        elif kind == "move":
            _, start, orig, idx = self.drag
            rect = QRectF(orig)
            rect.translate(p - start)
            rect.moveLeft(min(max(rect.left(), 0), W - rect.width()))
            rect.moveTop(min(max(rect.top(), 0), H - rect.height()))
        else:  # resize
            _, name, orig = self.drag
            idx = self.model.selected
            opposite = {"tl": orig.bottomRight(), "tr": orig.bottomLeft(),
                        "bl": orig.topRight(), "br": orig.topLeft()}[name]
            rect = QRectF(opposite, p).normalized()
        if 0 <= idx < len(self.regs()):
            self.set_rect(idx, rect)
            self.model.changed.emit()

    def mouseReleaseEvent(self, e):
        if not self.drag:
            return
        if self.drag[0] == "new":
            idx = self.drag[2]
            regs = self.regs()
            if 0 <= idx < len(regs):
                rect = self.rect_of(regs[idx])
                if rect.width() < 8 or rect.height() < 8:
                    del regs[idx]
                    self.model.selected = -1
                    self.model.selection_changed.emit()
        self.drag = None
        self.model.changed.emit()

    @staticmethod
    def draw_arrow(p, rect, angle, color):
        """Arrow from the start (darkest, 'Start darkness') to the end of the gradient."""
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)
        length = abs(rect.width() * dx) + abs(rect.height() * dy)
        c = rect.center()
        half = length * 0.35
        tail = QPointF(c.x() - dx * half, c.y() - dy * half)
        head = QPointF(c.x() + dx * half, c.y() + dy * half)
        p.save()
        pen = QtGui.QPen(color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(tail, head)
        for side in (1, -1):
            wing = a + math.pi + side * math.radians(25)
            p.drawLine(head, QPointF(head.x() + 14 * math.cos(wing), head.y() + 14 * math.sin(wing)))
        p.restore()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 40))   # nearly invisible, but catches the mouse

        for i, r in enumerate(self.regs()):
            rect = self.rect_of(r)
            selected = i == self.model.selected
            grad = r.get("type") == "gradient"
            base = QtGui.QColor(190, 120, 255) if grad else QtGui.QColor(0, 200, 255)
            pen = QtGui.QPen(QtGui.QColor(255, 170, 0) if selected else base)
            pen.setWidth(2 if selected else 1)
            pen.setStyle(Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)
            if grad:
                label = f"{i + 1}: {r['strength']:.0f}% -> {r.get('strength_end', 0):.0f}%"
            else:
                label = f"{i + 1}: {r['strength']:.0f}%"
            p.drawText(rect.topLeft() + QPointF(6, 16), label)
            if grad:
                self.draw_arrow(p, rect, r.get("angle", 0.0), pen.color())
            if selected:
                p.setBrush(QtGui.QColor(255, 255, 255))
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
                h = self.HANDLE
                for pt in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
                    p.drawRect(QRectF(pt.x() - h / 2, pt.y() - h / 2, h, h))

        text = "Drag to draw a new area  |  drag inside to move  |  drag corners to resize"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text) + 24
        box = QRectF((self.width() - tw) / 2, 12, tw, fm.height() + 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 200))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


# --------------------------------------------------------------------------- #
# Settings window
# --------------------------------------------------------------------------- #
class GrayField(QtWidgets.QWidget):
    """Opaque flat gray field, so uneven areas are easy to see while drawing."""

    def __init__(self, screen, level):
        super().__init__()
        self.level = level
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(screen.geometry())

    def set_level(self, level):
        self.level = level
        self.update()

    def paintEvent(self, _):
        QtGui.QPainter(self).fillRect(self.rect(), QtGui.QColor.fromRgbF(self.level, self.level, self.level))


class Settings(QtWidgets.QWidget):
    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        self.model = ctl.model
        self.setWindowTitle("Screen Uniformity")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(340)

        lay = QtWidgets.QVBoxLayout(self)

        self.chk_enable = QtWidgets.QCheckBox("Enable correction")
        self.chk_enable.setChecked(self.model.enabled)
        self.chk_enable.toggled.connect(self.model.set_enabled)
        lay.addWidget(self.chk_enable)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Monitor:"))
        self.combo = QtWidgets.QComboBox()
        for i, s in enumerate(QtWidgets.QApplication.screens()):
            self.combo.addItem(f"{i + 1}: {s.size().width()}x{s.size().height()}  ({s.name()})")
        self.combo.currentIndexChanged.connect(self.on_monitor_changed)
        row.addWidget(self.combo, 1)
        lay.addLayout(row)

        self.btn_edit = QtWidgets.QPushButton("Draw / edit areas on this monitor")
        self.btn_edit.setCheckable(True)
        self.btn_edit.toggled.connect(self.on_edit_toggled)
        lay.addWidget(self.btn_edit)

        self.chk_test = QtWidgets.QCheckBox("Show flat gray test field (helps spot uneven areas)")
        self.chk_test.toggled.connect(self.on_test_toggled)
        lay.addWidget(self.chk_test)
        self.s_gray, self.l_gray, self.w_gray, _ = self._slider(lay, "Gray level", 100, "%")
        self.s_gray.setValue(50)
        self.l_gray.setText("50%")
        self.s_gray.valueChanged.connect(self.on_gray)

        lay.addWidget(QtWidgets.QLabel("Areas on this monitor:"))
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(110)
        self.list.currentRowChanged.connect(self.model.select)
        lay.addWidget(self.list)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Type:"))
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItem("Flat (uniform darkening)")
        self.combo_type.addItem("Linear gradient")
        self.combo_type.currentIndexChanged.connect(self.on_type_changed)
        row.addWidget(self.combo_type, 1)
        lay.addLayout(row)

        self.s_strength, self.l_strength, _, self.name_strength = self._slider(lay, "Darkness", 100, "%")
        self.s_end, self.l_end, self.w_end, _ = self._slider(lay, "End darkness", 100, "%")
        self.s_angle, self.l_angle, self.w_angle, _ = self._slider(lay, "Direction", 359, "\u00b0")
        self.angle_hint = QtWidgets.QLabel(
            "Darkness fades from Start to End along the arrow shown on screen. "
            "0\u00b0 = left to right, 90\u00b0 = top to bottom, 180\u00b0 = right to left, "
            "270\u00b0 = bottom to top.")
        self.angle_hint.setWordWrap(True)
        self.angle_hint.setStyleSheet("color: gray;")
        lay.addWidget(self.angle_hint)
        self.s_feather, self.l_feather, _, _ = self._slider(lay, "Soft edge", 400, " px")
        self.s_strength.valueChanged.connect(self.on_strength)
        self.s_end.valueChanged.connect(self.on_end)
        self.s_angle.valueChanged.connect(self.on_angle)
        self.s_feather.valueChanged.connect(self.on_feather)

        row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_clear = QtWidgets.QPushButton("Clear all")
        self.btn_clear.clicked.connect(self.clear_all)
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_clear)
        lay.addLayout(row)

        hint = QtWidgets.QLabel(
            "Tip: darkness is the % of light removed. Closing this window keeps the "
            "correction running from the tray icon."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self.model.changed.connect(lambda: self._refresh_timer.start(60))
        self.model.selection_changed.connect(self.refresh)
        self.model.enabled_changed.connect(self.sync_enabled)
        self.refresh()

    # -- helpers ----------------------------------------------------------- #
    def _slider(self, layout, name, maximum, suffix):
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        name_lab = QtWidgets.QLabel(name)
        name_lab.setMinimumWidth(90)
        s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, maximum)
        lab = QtWidgets.QLabel("0" + suffix)
        lab.setMinimumWidth(60)
        lab.setProperty("suffix", suffix)
        row.addWidget(name_lab)
        row.addWidget(s, 1)
        row.addWidget(lab)
        layout.addWidget(box)
        return s, lab, box, name_lab

    def screen(self):
        screens = QtWidgets.QApplication.screens()
        return screens[max(0, min(self.combo.currentIndex(), len(screens) - 1))]

    def regions(self):
        return self.model.regions(self.screen().name())

    def selected_region(self):
        regs = self.regions()
        i = self.model.selected
        return regs[i] if 0 <= i < len(regs) else None

    @staticmethod
    def _text(i, r):
        if r.get("type") == "gradient":
            return (f"Area {i + 1}:  gradient {r['strength']:.0f}% \u2192 {r.get('strength_end', 0):.0f}%,"
                    f"  {r.get('angle', 0):.0f}\u00b0")
        return f"Area {i + 1}:  flat {r['strength']:.0f}%,  soft edge {r['feather']:.0f} px"

    # -- slots ------------------------------------------------------------- #
    def refresh(self):
        regs = self.regions()
        self.list.blockSignals(True)
        self.list.clear()
        for i, r in enumerate(regs):
            self.list.addItem(self._text(i, r))
        if 0 <= self.model.selected < len(regs):
            self.list.setCurrentRow(self.model.selected)
        self.list.blockSignals(False)

        r = self.selected_region()
        grad = r is not None and r.get("type") == "gradient"
        for sl, lab, value in (
                (self.s_strength, self.l_strength, r["strength"] if r else 0),
                (self.s_end, self.l_end, r.get("strength_end", 0) if r else 0),
                (self.s_angle, self.l_angle, r.get("angle", 0) if r else 0),
                (self.s_feather, self.l_feather, r["feather"] if r else 0)):
            sl.blockSignals(True)
            sl.setEnabled(r is not None)
            if r is not None:
                sl.setValue(int(round(value)))
                lab.setText(f"{int(round(value))}{lab.property('suffix')}")
            sl.blockSignals(False)
        self.w_end.setVisible(grad)
        self.w_angle.setVisible(grad)
        self.angle_hint.setVisible(grad)
        self.name_strength.setText("Start darkness" if grad else "Darkness")
        kind = r.get("type", "flat") if r else self.model.new_type
        self.combo_type.blockSignals(True)
        self.combo_type.setCurrentIndex(1 if kind == "gradient" else 0)
        self.combo_type.blockSignals(False)
        self.btn_delete.setEnabled(r is not None)
        self.btn_clear.setEnabled(bool(regs))

    def sync_enabled(self):
        self.chk_enable.blockSignals(True)
        self.chk_enable.setChecked(self.model.enabled)
        self.chk_enable.blockSignals(False)

    def on_strength(self, v):
        r = self.selected_region()
        if r:
            r["strength"] = float(v)
            self.l_strength.setText(f"{v}{self.l_strength.property('suffix')}")
            self.model.changed.emit()

    def on_feather(self, v):
        r = self.selected_region()
        if r:
            r["feather"] = float(v)
            self.l_feather.setText(f"{v}{self.l_feather.property('suffix')}")
            self.model.changed.emit()

    def on_end(self, v):
        r = self.selected_region()
        if r:
            r["strength_end"] = float(v)
            self.l_end.setText(f"{v}{self.l_end.property('suffix')}")
            self.model.changed.emit()

    def on_angle(self, v):
        r = self.selected_region()
        if r:
            r["angle"] = float(v)
            self.l_angle.setText(f"{v}{self.l_angle.property('suffix')}")
            self.model.changed.emit()

    def on_type_changed(self, idx):
        kind = "gradient" if idx == 1 else "flat"
        self.model.new_type = kind          # also the type of the next area you draw
        r = self.selected_region()
        if r is not None and r.get("type") != kind:
            r["type"] = kind
            if kind == "gradient" and abs(r.get("strength_end", r["strength"]) - r["strength"]) < 0.5:
                r["strength_end"] = 0.0     # sensible starting point: fade out to nothing
            self.model.changed.emit()
        self.refresh()

    def on_test_toggled(self, on):
        self.ctl.set_test(on, self.screen(), self.s_gray.value())

    def on_gray(self, v):
        self.l_gray.setText(f"{v}%")
        self.ctl.set_test_level(v)

    def on_monitor_changed(self, _):
        self.model.selected = -1
        if self.btn_edit.isChecked():
            self.ctl.set_editing(True, self.screen())
        if self.chk_test.isChecked():
            self.ctl.set_test(True, self.screen(), self.s_gray.value())
        self.refresh()

    def on_edit_toggled(self, on):
        self.btn_edit.setText("Finish editing" if on else "Draw / edit areas on this monitor")
        self.ctl.set_editing(on, self.screen())
        self.raise_()

    def delete_selected(self):
        regs = self.regions()
        i = self.model.selected
        if 0 <= i < len(regs):
            del regs[i]
            self.model.selected = -1
            self.model.selection_changed.emit()
            self.model.changed.emit()

    def clear_all(self):
        if QtWidgets.QMessageBox.question(
                self, "Clear all", "Remove all areas on this monitor?"
        ) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.regions().clear()
            self.model.selected = -1
            self.model.selection_changed.emit()
            self.model.changed.emit()

    def stop_editing(self):
        if self.btn_edit.isChecked():
            self.btn_edit.setChecked(False)

    def stop_test(self):
        if self.chk_test.isChecked():
            self.chk_test.setChecked(False)

    def closeEvent(self, e):
        self.stop_editing()
        self.stop_test()
        if self.ctl.tray is not None:
            e.ignore()
            self.hide()
        else:
            QtWidgets.QApplication.quit()


# --------------------------------------------------------------------------- #
# Controller / tray
# --------------------------------------------------------------------------- #
class Controller(QtCore.QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.model = Model()
        self.overlays = []
        self.editor = None
        self.gray = None
        self.tray = None
        self.build_overlays()
        app.screenAdded.connect(lambda _: self.build_overlays())
        app.screenRemoved.connect(lambda _: self.build_overlays())

        self.settings = Settings(self)
        self.setup_tray()
        self.settings.show()

        self._top_timer = QtCore.QTimer(self)
        self._top_timer.timeout.connect(self.keep_on_top)
        if KEEP_ON_TOP_MS > 0:
            self._top_timer.start(KEEP_ON_TOP_MS)

    def keep_on_top(self):
        """Games sometimes become topmost after we do; quietly climb back above them.
        Skipped while the settings/editor/test field are in use, since those must stay above."""
        if self.editor is not None or self.gray is not None or self.settings.isVisible():
            return
        for o in self.overlays:
            force_topmost(o)

    def build_overlays(self):
        for o in self.overlays:
            o.close()
            o.deleteLater()
        self.overlays = [Overlay(s, self.model) for s in QtWidgets.QApplication.screens()]

    def set_editing(self, on, screen):
        if self.editor is not None:
            self.editor.close()
            self.editor.deleteLater()
            self.editor = None
        if on:
            self.editor = Editor(screen, self.model)
            self.editor.show()
        self.restack()

    def set_test(self, on, screen, level):
        if self.gray is not None:
            self.gray.close()
            self.gray.deleteLater()
            self.gray = None
        if on:
            self.gray = GrayField(screen, level / 100.0)
            self.gray.show()
        self.restack()

    def set_test_level(self, level):
        if self.gray is not None:
            self.gray.set_level(level / 100.0)

    def restack(self):
        """Z-order, bottom to top: gray field, overlays, editor, settings."""
        if self.gray is not None:
            self.gray.raise_()
        for o in self.overlays:
            o.raise_()
        if self.editor is not None:
            self.editor.raise_()
        self.settings.raise_()

    def setup_tray(self):
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            return
        pm = QtGui.QPixmap(64, 64)
        pm.fill(Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        grad = QtGui.QLinearGradient(0, 0, 64, 0)
        grad.setColorAt(0, QtGui.QColor(240, 240, 240))
        grad.setColorAt(1, QtGui.QColor(70, 70, 70))
        p.setBrush(grad)
        p.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 3))
        p.drawRoundedRect(4, 10, 56, 44, 6, 6)
        p.end()

        self.tray = QtWidgets.QSystemTrayIcon(QtGui.QIcon(pm), self)
        self.tray.setToolTip("Screen Uniformity")
        menu = QtWidgets.QMenu()
        act_settings = menu.addAction("Settings...")
        act_settings.triggered.connect(self.show_settings)
        self.act_enable = menu.addAction("Enable correction")
        self.act_enable.setCheckable(True)
        self.act_enable.setChecked(self.model.enabled)
        self.act_enable.toggled.connect(self.model.set_enabled)
        self.model.enabled_changed.connect(self._sync_tray)
        menu.addSeparator()
        menu.addAction("Quit").triggered.connect(self.quit)
        self.tray.setContextMenu(menu)
        self._menu = menu
        self.tray.activated.connect(
            lambda reason: self.show_settings()
            if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.show()

    def _sync_tray(self):
        self.act_enable.blockSignals(True)
        self.act_enable.setChecked(self.model.enabled)
        self.act_enable.blockSignals(False)

    def show_settings(self):
        self.settings.show()
        self.settings.raise_()
        self.settings.activateWindow()

    def quit(self):
        self.settings.stop_editing()
        self.settings.stop_test()
        self.model.save()
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    ctl = Controller(app)  # noqa: F841  (keep a reference alive)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
