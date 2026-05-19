import pyaudio
import wave
import torch
import librosa
import numpy as np

from bp_network import BPNetwork


# ======================
# 参数
# ======================

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

RECORD_SECONDS = 1

WAVE_OUTPUT_FILENAME = "temp.wav"


# ======================
# 设备
# ======================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================
# 加载模型
# ======================

input_size = 40 * 32

model = BPNetwork(
    input_size=input_size,
    output_size=10,
).to(device)

model.load_state_dict(
    torch.load(
        "./weights/voice_digits_bp.pth",
        map_location=device
    )
)

model.eval()


# ======================
# 录音函数
# ======================

def record_audio():

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("开始录音...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

        data = stream.read(CHUNK)

        frames.append(data)

    print("录音结束")

    stream.stop_stream()
    stream.close()

    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    wf.writeframes(b''.join(frames))

    wf.close()


# ======================
# 提取 MFCC
# ======================
from voice_process import extract_feature



# ======================
# 预测
# ======================

def predict():

    feature = extract_feature(WAVE_OUTPUT_FILENAME)

    with torch.no_grad():

        output = model(feature)

        _, predicted = torch.max(output, 1)

    return predicted.item()


# ======================
# 主循环
# ======================

while True:

    input("按回车开始录音...")

    record_audio()

    result = predict()

    print(f"预测结果: {result}")

    print()