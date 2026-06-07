# -*- coding: utf-8 -*-
"""
实验一：手写数字与字母识别 - 图形界面
功能：
  1. 手写数字识别（CNN / BP 双模型，画板 + 概率柱状图）
  2. 手写字母识别（CNN / BP 双模型，画板 + 概率柱状图）
  3. 语音数字识别（录音 + 波形/MFCC + BP语音数字模型）
  4. 可调节笔刷粗细、清空画板、实时显示结果
"""

import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTabWidget, QSlider, QFileDialog,
    QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
# 导入模型及工具
import torch
from bp_network import BPNetwork
from cnn_model_mnist import DeepCNN
from image_process import preprocess_single, preprocess_letter
from voice_process import audio_to_mfcc_matrix, audio_to_feature_tensor, VOICE_INPUT_SIZE

try:
    import librosa
    VOICE_BACKEND = True
except ImportError:
    VOICE_BACKEND = False
    print("Warning: librosa not installed, voice recognition disabled. Try: pip install librosa")

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

# 置信度柱状图 Canvas
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
        self.setWindowTitle("手写识别系统（CNN + BP）")
        self.setGeometry(100, 100, 1200, 800)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN 数字模型 
        self.cnn_model = None
        cnn_weight_path = os.path.join("weights", "cnn_mnist.pth")
        if os.path.exists(cnn_weight_path):
            self.cnn_model = DeepCNN(num_classes=10).to(self.device)
            self.cnn_model.load_state_dict(torch.load(cnn_weight_path, map_location=self.device))
            self.cnn_model.eval()
            print(f"[成功] 加载 CNN 数字模型：{cnn_weight_path}")
        else:
            print(f"[警告] CNN 数字权重 {cnn_weight_path} 不存在，请先运行 cnn_model_mnist.py 训练")
        # CNN 字母模型 
        self.cnn_letters_model = None
        cnn_letters_path = os.path.join("weights", "cnn_letters.pth")
        if os.path.exists(cnn_letters_path):
            self.cnn_letters_model = DeepCNN(num_classes=26).to(self.device)
            self.cnn_letters_model.load_state_dict(torch.load(cnn_letters_path, map_location=self.device))
            self.cnn_letters_model.eval()
            print(f"[成功] 加载 CNN 字母模型：{cnn_letters_path}")
        else:
            print(f"[警告] CNN 字母权重 {cnn_letters_path} 不存在，请先运行 cnn_model_letters.py 训练")

        # ---------- BP 数字模型 ----------
        self.bp_digits_model = None
        bp_digits_path = os.path.join("weights", "mnist_bp.pth")
        if os.path.exists(bp_digits_path):
            self.bp_digits_model = BPNetwork(input_size=784,output_size=10).to(self.device)
            self.bp_digits_model.load_state_dict(torch.load(bp_digits_path, map_location=self.device))
            self.bp_digits_model.eval()
            print(f"[成功] 加载 BP 数字模型：{bp_digits_path}")
        else:
            print(f"[警告] BP 数字权重 {bp_digits_path} 不存在，请先运行 train_mnist.py 训练")

        # ---------- BP 字母模型 ----------
        self.bp_letters_model = None
        bp_letters_path = os.path.join("weights", "letters_bp.pth")
        if os.path.exists(bp_letters_path):
            self.bp_letters_model = BPNetwork(input_size=784, output_size=26, task="letters").to(self.device)
            self.bp_letters_model.load_state_dict(torch.load(bp_letters_path, map_location=self.device))
            self.bp_letters_model.eval()
            print(f"[成功] 加载 BP 字母模型：{bp_letters_path}")
        else:
            print(f"[警告] BP 字母权重 {bp_letters_path} 不存在，请先运行 train_letters.py 训练")

        # ---------- BP 语音数字模型 ----------
        self.bp_voice_digits_model = None
        bp_voice_path = os.path.join("weights", "voice_digits_bp.pth")
        if os.path.exists(bp_voice_path):
            self.bp_voice_digits_model = BPNetwork(
                input_size=VOICE_INPUT_SIZE, output_size=10
            ).to(self.device)
            self.bp_voice_digits_model.load_state_dict(
                torch.load(bp_voice_path, map_location=self.device)
            )
            self.bp_voice_digits_model.eval()
            print(f"[成功] 加载 BP 语音数字模型：{bp_voice_path}")
        else:
            print(
                f"[警告] BP 语音数字权重 {bp_voice_path} 不存在，"
                "请先运行 train_voice_digits.py 训练"
            )

        # 全局变量 — 默认值稍后在工具栏初始化后设定
        self.current_model = "BP_digits"
        self.current_letter_model = "CNN" if self.cnn_letters_model is not None else "BP_letters"
        self.current_voice_model = "bp_voice_digits"
        self.recorder = None
        self.audio_data = None
        self.sample_rate = 16000

        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部工具栏（批量测试）
        toolbar = QHBoxLayout()
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

        # 模式2：手写字母
        self.letters_tab = QWidget()
        self.setup_letters_tab()
        self.tab_widget.addTab(self.letters_tab, "手写字母识别")

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

        # 结果标签 + 模型选择器
        result_area = QVBoxLayout()
        self.digit_result_label = QLabel("识别结果: 未识别")
        self.digit_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        result_area.addWidget(self.digit_result_label)

        digit_models = []
        if self.cnn_model is not None:
            digit_models.append("CNN")
        digit_models.append("BP")
        self.model_combo = QComboBox()
        self.model_combo.addItems(digit_models)
        self.current_model = digit_models[0]
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("数字模型:"))
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        result_area.addLayout(model_row)
        layout.addLayout(result_area, 0, 1)

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

    # ----------------- 字母模式界面 -----------------
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

        # 结果标签 + 模型选择器 (同一区域，与数字 tab 对齐)
        result_area = QVBoxLayout()
        self.letter_result_label = QLabel("识别结果: 未识别")
        self.letter_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        result_area.addWidget(self.letter_result_label)

        letter_models = []
        if self.cnn_letters_model is not None:
            letter_models.append("CNN")
        letter_models.append("BP")
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("字母模型:"))
        self.letter_model_combo = QComboBox()
        self.letter_model_combo.addItems(letter_models)
        self.letter_model_combo.currentTextChanged.connect(self.on_letter_model_changed)
        model_row.addWidget(self.letter_model_combo)
        model_row.addStretch()
        result_area.addLayout(model_row)
        layout.addLayout(result_area, 0, 1)

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

        self.mfcc_canvas = FigureCanvas(Figure(figsize=(5, 2)))
        self.mfcc_ax = self.mfcc_canvas.figure.subplots()
        self.mfcc_ax.set_title("MFCC 特征")
        self.mfcc_ax.set_xlabel("帧")
        self.mfcc_ax.set_ylabel("系数")
        self.mfcc_canvas.figure.tight_layout()
        left_panel.addWidget(self.mfcc_canvas)

        btn_load_audio = QPushButton("加载音频文件")
        btn_load_audio.clicked.connect(self.load_voice_file)
        left_panel.addWidget(btn_load_audio)

        layout.addLayout(left_panel)

        right_panel = QVBoxLayout()
        self.voice_result_label = QLabel("识别结果: 未识别")
        self.voice_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_panel.addWidget(self.voice_result_label)

        voice_model_row = QHBoxLayout()
        voice_model_row.addWidget(QLabel("语音模型:"))
        self.voice_model_combo = QComboBox()
        self.voice_model_combo.addItem("BP语音数字", "bp_voice_digits")
        self.voice_model_combo.currentIndexChanged.connect(self.on_voice_model_changed)
        voice_model_row.addWidget(self.voice_model_combo)
        voice_model_row.addStretch()
        right_panel.addLayout(voice_model_row)

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
        processed = preprocess_single(pil_img)

        current = self.current_model

        if current == 'CNN' and self.cnn_model is not None:
            # CNN 预处理：与训练时一致 Normalize((0.1307,), (0.3081,))
            img_float = processed.astype(np.float32) / 255.0
            img_norm = (img_float - 0.1307) / 0.3081
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.cnn_model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        elif current == 'CNN' and self.cnn_model is None:
            self.digit_result_label.setText("识别结果: CNN模型未加载")
            return
        elif current == 'BP' and self.bp_digits_model is not None:
            img_float = processed.astype(np.float32) / 255.0
            img_flat = img_float.flatten()  # ✓ 展平成 784 维
            processed_tensor = torch.from_numpy(img_flat).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                logits = self.bp_digits_model(processed_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        elif current == 'BP' and self.bp_digits_model is None:
            self.digit_result_label.setText("识别结果: BP模型未加载")
            return
        else:
            pred, probs = 0, np.ones(10)/10

        self.digit_result_label.setText(f"识别结果: {pred}")
        self.digit_confidence_canvas.update_bars(probs, [str(i) for i in range(10)])

    def recognize_letter(self):
        # 从画布获取原始分辨率图像（保留灰度，不做二值化）
        qimage = self.letter_canvas.image.toImage()
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        gray = arr[:, :, 0]  # uint8, 0-255, 黑底白字

        processed = preprocess_letter(gray)

        current = self.current_letter_model

        if current == 'CNN' and self.cnn_letters_model is not None:
            # CNN 训练时使用 Normalize((0.1307,), (0.3081,))
            img_norm = (processed - 0.1307) / 0.3081
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.cnn_letters_model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_char = chr(ord('A') + pred_idx)
            letter_labels = [chr(ord('A') + i) for i in range(26)]
        elif current == 'CNN' and self.cnn_letters_model is None:
            self.letter_result_label.setText("识别结果: CNN字母模型未加载")
            return
        elif current == 'BP' and self.bp_letters_model is not None:
            # BP 训练时使用 Normalize((0.5,), (0.5,)) → 映射到 [-1, 1]
            img_norm = (processed - 0.5) / 0.5
            img_flat = img_norm.flatten()
            processed_tensor = torch.from_numpy(img_flat).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                logits = self.bp_letters_model(processed_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_char = chr(ord('A') + pred_idx)
            letter_labels = [chr(ord('A') + i) for i in range(26)]
        elif current == 'BP' and self.bp_letters_model is None:
            self.letter_result_label.setText("识别结果: BP字母模型未加载")
            return
        else:
            self.letter_result_label.setText("识别结果: 模型未加载")
            letter_labels = [chr(ord('A') + i) for i in range(26)]
            self.letter_confidence_canvas.update_bars(np.zeros(26), letter_labels)
            return

        self.letter_result_label.setText(f"识别结果: {pred_char}")
        self.letter_confidence_canvas.update_bars(probs, letter_labels)

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

    def load_voice_file(self):
        if not VOICE_BACKEND:
            self.voice_result_label.setText("识别结果: 请安装 librosa")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频 (*.wav *.mp3 *.flac);;所有文件 (*)"
        )
        if not path:
            return
        try:
            audio, sr = librosa.load(path, sr=None, mono=True)
            self.audio_data = audio.astype(np.float32)
            self.sample_rate = sr
            self.waveform_ax.clear()
            self.waveform_ax.plot(np.arange(len(audio)), audio)
            self.waveform_ax.set_title("已加载音频波形")
            self.waveform_ax.set_xlabel("采样点")
            self.waveform_canvas.draw_idle()
            self.voice_result_label.setText(f"已加载: {os.path.basename(path)}")
        except Exception as e:
            self.voice_result_label.setText(f"加载失败: {e}")

    def update_mfcc_plot(self, mfcc):
        self.mfcc_ax.clear()
        self.mfcc_ax.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
        self.mfcc_ax.set_title("MFCC 特征")
        self.mfcc_ax.set_xlabel("帧")
        self.mfcc_ax.set_ylabel("系数")
        self.mfcc_canvas.figure.tight_layout()
        self.mfcc_canvas.draw_idle()

    def recognize_voice(self):
        if self.audio_data is None or len(self.audio_data) == 0:
            self.voice_result_label.setText("请先录音或加载音频")
            return
        if not VOICE_BACKEND:
            self.voice_result_label.setText("识别结果: 请安装 librosa")
            return

        try:
            mfcc = audio_to_mfcc_matrix(self.audio_data, self.sample_rate)
            self.update_mfcc_plot(mfcc)
        except Exception as e:
            self.voice_result_label.setText(f"特征提取失败: {e}")
            return

        current = self.current_voice_model
        digit_labels = [str(i) for i in range(10)]

        if current == "bp_voice_digits" and self.bp_voice_digits_model is not None:
            try:
                features = audio_to_feature_tensor(
                    self.audio_data, self.sample_rate, self.device
                )
                with torch.no_grad():
                    logits = self.bp_voice_digits_model(features)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
            except Exception as e:
                self.voice_result_label.setText(f"识别失败: {e}")
                return
        elif current == "bp_voice_digits":
            self.voice_result_label.setText("识别结果: BP语音模型未加载")
            return
        else:
            self.voice_result_label.setText("识别结果: 未知模型")
            return

        self.voice_result_label.setText(f"识别结果: {pred}")
        self.voice_confidence_canvas.update_bars(probs, digit_labels)

    # ----------------- 模型切换和批量测试 -----------------
    def on_model_changed(self, model_name):
        self.current_model = model_name
        print(f"切换到数字识别模型: {model_name}")

    def on_letter_model_changed(self, model_name):
        self.current_letter_model = model_name
        print(f"切换到字母识别模型: {model_name}")

    def on_voice_model_changed(self, _index):
        self.current_voice_model = self.voice_model_combo.currentData()
        print(f"切换到语音识别模型: {self.current_voice_model}")

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