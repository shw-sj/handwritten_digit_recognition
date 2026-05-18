# -*- coding: utf-8 -*-
"""
实验一：基于BP算法的手写数字识别 - 图形界面 (成员C)
功能：
  1. 手写数字识别模式（画板 + 数字识别 + 概率柱状图，仅保留BP像素）
  2. 手写字母识别模式（解禁，预留BP框架，CNN代码已注释）
  3. 语音数字识别模式（录音 + 波形显示 + 数字识别 + MFCC占位）
  4. 模型切换（仅保留BP_像素）
  5. 批量测试框架（模拟）
  6. 可调节笔刷粗细、清空画板、实时显示结果

注：CNN相关代码【仅注释，未删除】，BP模型参数已修复，编码已修复
"""

import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTabWidget, QSlider, QFileDialog,
    QGroupBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from matplotlib import rcParams
# 导入模型及工具
import torch
from bp_network import BPNetwork
# ---------------------- CNN 导入 注释（保留代码，不删除）----------------------
# from cnn_inference import CNNInference
# -----------------------------------------------------------------------------
from image_loader import preprocess_single
from image_feature import extract_features, get_feature_dim

# 设置字体为 SimHei（黑体）或其他支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ---------- 音频录制线程 ----------
try:
    import sounddevice as sd
    AUDIO_BACKEND = "sound"
except ImportError:
    AUDIO_BACKEND = None
    print("Warning: sounddevice not installed, recording disabled. Try: pip install sounddevice")

class AudioRecorder(QThread):
    finished = pyqtSignal(np.ndarray, int)
    waveform = pyqtSignal(np.ndarray)

    def __init__(self, duration=3, samplerate=1600):
        super().__init__()
        self.duration = duration
        self.samplerate = samplerate
        self.stop_flag = False

    def run(self):
        if AUDIO_BACKEND is None:
            self.finished.emit(np.zeros(0), self.samplerate)
            return
        try:
            recording = sd.rec(int(self.duration * self.samplerate),
                               samplerate=self.samplerate, channels=1, dtype='float32')
            for i in range(10):
                if self.stop_flag:
                    break
                self.msleep(100)
                current = recording[:int((i+1)*0.1*self.samplerate)]
                if len(current) > 0:
                    self.waveform.emit(current.flatten())
            sd.wait()
            if self.stop_flag:
                self.finished.emit(np.zeros(0), self.samplerate)
            else:
                self.finished.emit(recording.flatten(), self.samplerate)
        except Exception as e:
            print("录音出错:", e)
            self.finished.emit(np.zeros(0), self.samplerate)

    def stop(self):
        self.stop_flag = True

# ---------- 置信度柱状图 Canvas ----------
class ConfidenceBarCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.2)
        self.bar_plot = None

    def update_bars(self, probabilities, labels):
        self.axes.clear()
        x = np.arange(len(labels))
        self.axes.bar(x, probabilities, color='steelblue')
        self.axes.set_xticks(x)
        self.axes.set_xticklabels(labels, rotation=45, ha='right')
        self.axes.set_ylim([0, 1])
        self.axes.set_ylabel("置信度")
        self.axes.set_title("分类概率分布")
        self.fig.tight_layout()
        self.draw()

# ---------- 手写画板控件 ----------
class HandwritingWidget(QWidget):
    def __init__(self, parent=None, pen_size=12):
        super().__init__(parent)
        self.pen_size = pen_size
        self.pen_color = QColor(255, 255, 255)
        self.background_color = QColor(0, 0, 0)
        self.image = QPixmap(self.size())
        self.image.fill(self.background_color)
        self.last = None

    def set_pen_size(self, size):
        self.pen_size = size

    def resizeEvent(self, event):
        if not self.image.isNull():
            new_size = event.size()
            scaled_image = self.image.scaled(new_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.image = scaled_image
        else:
            self.image = QPixmap(event.size())
            self.image.fill(self.background_color)
        self.update()

    def clear(self):
        self.image.fill(self.background_color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last = event.pos()
            self.draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last is not None:
            self.draw_line(self.last, event.pos())
            self.last = event.pos()

    def mouseReleaseEvent(self, event):
        self.last = None

    def draw_point(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(point)
        self.update()

    def draw_line(self, start, end):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        self.update()

    def get_image_array(self, target_size=(28, 28)):
        qimage = self.image.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        gray = arr[:, :, 0].astype(np.float32) / 255.0
        pil_img = Image.fromarray((gray * 255).astype('uint8'))
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        resized = np.array(pil_img).astype(np.float32) / 255.0
        return resized

# ---------- 主窗口 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写识别系统（BP测试版 - CNN已注释）")
        self.setGeometry(100, 100, 1200, 800)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ---------------------- CNN 模型初始化 注释（保留代码）----------------------
        # cnn_weight_path = os.path.join("models", "lenet5_mixed.pth")
        # os.makedirs("models", exist_ok=True)
        # self.cnn_digits = CNNInference(mode='digits', weight_path=cnn_weight_path)
        # self.cnn_letters = CNNInference(mode='letters', weight_path=cnn_weight_path)
        # -----------------------------------------------------------------------------

        # ---------- BP 数字模型（修复参数错误：无参初始化！）----------
        self.bp_digits_models = {}
        feature_configs = {
            'BP_像素': {'method': 'pixel', 'kwargs': {'grid_size': 28}, 'weight': './weights/mnist_bp.pth'},
        }
        for name, cfg in feature_configs.items():
            # 修复核心报错：BPNet() 无参数传入
            model = BPNetwork(input_size=784,hidden_size=256,output_size=10).to(self.device)
            if os.path.exists(cfg['weight']):
                model.load_state_dict(torch.load(cfg['weight'], map_location=self.device))
                model.eval()
                # 移除emoji，修复GBK编码错误
                print(f"[成功] 加载 {name} 模型成功：{cfg['weight']}")
            else:
                print(f"[警告] {cfg['weight']} 不存在，使用随机权重")
            self.bp_digits_models[name] = model

        # ========== 新增：字母BP模型加载（不改动上方数字模型代码） ==========
        self.bp_letters_models = {}
        letter_feature_configs = {
            'BP_像素': {'method': 'pixel', 'kwargs': {'grid_size': 28}, 'weight': './weights/letters_bp.pth'},
        }
        for name, cfg in letter_feature_configs.items():
            # 字母模型和数字模型使用相同的BPNet结构（需确保emnist_bp.pth输出维度为26）
            model = BPNetwork(input_size=784,hidden_size=256,output_size=26).to(self.device)
            if os.path.exists(cfg['weight']):
                model.load_state_dict(torch.load(cfg['weight'], map_location=self.device))
                model.eval()
                print(f"[成功] 加载字母 {name} 模型成功：{cfg['weight']}")
            else:
                print(f"[警告] 字母模型 {cfg['weight']} 不存在，使用随机权重")
            self.bp_letters_models[name] = model
        # ========== 新增结束 ==========

        # 全局变量
        self.current_model = "BP_像素"
        self.recorder = None
        self.audio_data = None
        self.sample_rate = 16000

        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部工具栏（模型选择 + 批量测试）
        toolbar = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["BP_像素"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        toolbar.addWidget(QLabel("数字识别模型："))
        toolbar.addWidget(self.model_combo)
        toolbar.addStretch()
        self.batch_btn = QPushButton("批量测试(模拟)")
        self.batch_btn.clicked.connect(self.on_batch_test)
        toolbar.addWidget(self.batch_btn)
        main_layout.addLayout(toolbar)

        # 选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 模式1：手写数字（正常启用）
        self.digits_tab = QWidget()
        self.setup_digits_tab()
        self.tab_widget.addTab(self.digits_tab, "手写数字识别")

        # 模式2：手写字母（解禁！不再禁用，CNN代码注释）
        self.letters_tab = QWidget()
        self.setup_letters_tab()
        self.tab_widget.addTab(self.letters_tab, "手写字母识别（BP预留）")

        # 模式3：语音数字
        self.voice_tab = QWidget()
        self.setup_voice_tab()
        self.tab_widget.addTab(self.voice_tab, "语音数字识别")

    # ----------------- 数字模式界面 -----------------
    def setup_digits_tab(self):
        layout = QGridLayout(self.digits_tab)

        self.digit_canvas = HandwritingWidget(pen_size=12)
        layout.addWidget(self.digit_canvas, 0, 0, 2, 1)

        brush_layout = QVBoxLayout()
        self.digit_brush_slider = QSlider(Qt.Horizontal)
        self.digit_brush_slider.setRange(2, 30)
        self.digit_brush_slider.setValue(12)
        self.digit_brush_slider.valueChanged.connect(self.digit_canvas.set_pen_size)
        brush_layout.addWidget(QLabel("笔刷大小:"))
        brush_layout.addWidget(self.digit_brush_slider)

        btn_clear = QPushButton("清空画板")
        btn_clear.clicked.connect(self.digit_canvas.clear)
        brush_layout.addWidget(btn_clear)
        layout.addLayout(brush_layout, 2, 0)

        self.digit_result_label = QLabel("识别结果: 未识别")
        self.digit_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(self.digit_result_label, 0, 1)

        self.digit_confidence_canvas = ConfidenceBarCanvas(self, width=5, height=4)
        layout.addWidget(self.digit_confidence_canvas, 1, 1)

        btn_recognize = QPushButton("识别数字")
        btn_recognize.clicked.connect(self.recognize_digit)
        layout.addWidget(btn_recognize, 2, 1)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)

    # ----------------- 字母模式界面（解禁） -----------------
    def setup_letters_tab(self):
        layout = QGridLayout(self.letters_tab)

        self.letter_canvas = HandwritingWidget(pen_size=12)
        layout.addWidget(self.letter_canvas, 0, 0, 2, 1)

        brush_layout = QVBoxLayout()
        self.letter_brush_slider = QSlider(Qt.Horizontal)
        self.letter_brush_slider.setRange(2, 30)
        self.letter_brush_slider.setValue(12)
        self.letter_brush_slider.valueChanged.connect(self.letter_canvas.set_pen_size)
        brush_layout.addWidget(QLabel("笔刷大小:"))
        brush_layout.addWidget(self.letter_brush_slider)

        btn_clear = QPushButton("清空画板")
        btn_clear.clicked.connect(self.letter_canvas.clear)
        brush_layout.addWidget(btn_clear)
        layout.addLayout(brush_layout, 2, 0)

        self.letter_result_label = QLabel("识别结果: BP暂未训练字母模型")
        self.letter_result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #666;")
        layout.addWidget(self.letter_result_label, 0, 1)

        self.letter_confidence_canvas = ConfidenceBarCanvas(self, width=5, height=4)
        layout.addWidget(self.letter_confidence_canvas, 1, 1)

        btn_recognize = QPushButton("识别字母")
        btn_recognize.clicked.connect(self.recognize_letter)
        layout.addWidget(btn_recognize, 2, 1)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)

    # ----------------- 语音模式界面 -----------------
    def setup_voice_tab(self):
        layout = QHBoxLayout(self.voice_tab)

        left_panel = QVBoxLayout()
        self.record_btn = QPushButton("录音 (3秒)")
        self.record_btn.clicked.connect(self.start_recording)
        left_panel.addWidget(self.record_btn)

        self.waveform_canvas = FigureCanvas(Figure(figsize=(5, 2)))
        self.waveform_ax = self.waveform_canvas.figure.subplots()
        self.waveform_ax.set_title("实时波形")
        self.waveform_ax.set_xlabel("采样点")
        self.waveform_line, = self.waveform_ax.plot([], [])
        self.waveform_canvas.figure.tight_layout()
        left_panel.addWidget(self.waveform_canvas)

        mfcc_label = QLabel("MFCC频谱图 (集成后显示)")
        mfcc_label.setAlignment(Qt.AlignCenter)
        mfcc_label.setStyleSheet("border: 1px solid gray; background: #f0f0f0;")
        mfcc_label.setFixedHeight(150)
        left_panel.addWidget(mfcc_label)

        layout.addLayout(left_panel)

        right_panel = QVBoxLayout()
        self.voice_result_label = QLabel("识别结果: 未识别")
        self.voice_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_panel.addWidget(self.voice_result_label)

        self.voice_confidence_canvas = ConfidenceBarCanvas(self, width=5, height=4)
        right_panel.addWidget(self.voice_confidence_canvas)

        btn_voice_recognize = QPushButton("识别语音数字")
        btn_voice_recognize.clicked.connect(self.recognize_voice)
        right_panel.addWidget(btn_voice_recognize)

        layout.addLayout(right_panel)

    # ----------------- 识别方法 -----------------
    def recognize_digit(self):
        img_array = self.digit_canvas.get_image_array(target_size=(28,28))
        pil_img = Image.fromarray((img_array * 255).astype('uint8'))
        processed, _ = preprocess_single(pil_img, target_size=28)

        current = self.current_model

        # ---------------------- CNN 分支 注释（保留代码）----------------------
        # if current == 'CNN':
        #     processed_float = processed.astype(np.float32) / 255.0
        #     pred, probs = self.cnn_digits.predict(processed_float)
        # ---------------------------------------------------------------------
        if current in self.bp_digits_models:
            method_map = {
                'BP_像素': ('pixel', {'grid_size': 28}),
            }
            method, kwargs = method_map[current]
            img_batch = processed[np.newaxis, ...]
            features = extract_features(img_batch, method=method, **kwargs)
            model = self.bp_digits_models[current]
            model.eval()
            with torch.no_grad():
                feat_tensor = torch.from_numpy(features).float().to(self.device)
                logits = model(feat_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        else:
            pred, probs = 0, np.ones(10)/10

        self.digit_result_label.setText(f"识别结果: {pred}")
        self.digit_confidence_canvas.update_bars(probs, [str(i) for i in range(10)])

    def recognize_letter(self):
    # 1. 获取画板图像并预处理（保留原有逻辑）
      img_array = self.letter_canvas.get_image_array(target_size=(28,28))
      pil_img = Image.fromarray((img_array * 255).astype('uint8'))
      processed, _ = preprocess_single(pil_img, target_size=28)

    # 2. 选择当前BP模型（仅BP_像素）
      current = self.current_model
      if current in self.bp_letters_models:
        # 3. 特征提取（保留原有逻辑）
        method_map = {
            'BP_像素': ('pixel', {'grid_size': 28}),
        }
        method, kwargs = method_map[current]
        img_batch = processed[np.newaxis, ...]
        features = extract_features(img_batch, method=method,** kwargs)
        
        # 4. BP模型预测（47类）
        model = self.bp_letters_models[current]
        model.eval()
        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).float().to(self.device)
            logits = model(feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # 5. 修正：EMNIST 47类标签映射
        def emnist_label_to_char(label):
            # EMNIST ByMerge标签规则：0-9=数字，10-35=大写字母，36-46=小写字母
            if 0 <= label <=9:
                return str(label)
            elif 10 <= label <=35:
                return chr(ord('A') + label -10)
            elif 36 <= label <=46:
                return chr(ord('a') + label -36)
            else:
                return "未知"
        
        pred_idx = np.argmax(probs)
        pred_char = emnist_label_to_char(pred_idx)
        
        # 6. 更新UI显示（47类标签）
        self.letter_result_label.setText(f"识别结果: {pred_char}")
        letter_labels = [emnist_label_to_char(i) for i in range(47)]
        self.letter_confidence_canvas.update_bars(probs, letter_labels)
      else:
        # 兜底：模型不存在时的提示
        self.letter_result_label.setText("识别结果: 模型未加载")
        letter_labels = [f"类{i}" for i in range(47)]
        self.letter_confidence_canvas.update_bars(np.zeros(47), letter_labels)

    # ----------------- 语音部分 -----------------
    def start_recording(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait()
        self.record_btn.setEnabled(False)
        self.record_btn.setText("录音中... 3秒")
        self.audio_data = None
        self.recorder = AudioRecorder(duration=3, samplerate=16000)
        self.recorder.waveform.connect(self.update_waveform)
        self.recorder.finished.connect(self.on_recording_finished)
        self.recorder.start()

    def update_waveform(self, data):
        if len(data) > 0:
            self.waveform_line.set_data(np.arange(len(data)), data)
            self.waveform_ax.relim()
            self.waveform_ax.autoscale_view()
            self.waveform_canvas.draw_idle()

    def on_recording_finished(self, audio, sr):
        self.record_btn.setEnabled(True)
        self.record_btn.setText("录音 (3秒)")
        self.audio_data = audio
        self.sample_rate = sr
        if len(audio) == 0:
            self.waveform_ax.clear()
            self.waveform_ax.set_title("录音失败或无音频设备")
            self.waveform_canvas.draw()

    def recognize_voice(self):
        if self.audio_data is None or len(self.audio_data) == 0:
            self.voice_result_label.setText("请先录音")
            return
        pred = np.random.randint(0, 10)
        probs = np.random.dirichlet(np.ones(10))
        self.voice_result_label.setText(f"识别结果: {pred}")
        self.voice_confidence_canvas.update_bars(probs, [str(i) for i in range(10)])

    # ----------------- 模型切换和批量测试 -----------------
    def on_model_changed(self, model_name):
        self.current_model = model_name
        print(f"切换到数字识别模型: {model_name}")

    def on_batch_test(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含图像/音频的测试文件夹")
        if folder:
            self.batch_btn.setText("批量测试中...")
            QTimer.singleShot(1000, lambda: self.batch_btn.setText("批量测试(模拟)"))
            print(f"批量测试文件夹: {folder}")

# ---------- 程序入口 ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())