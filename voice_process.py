import numpy as np
import torch
import librosa

VOICE_SAMPLE_RATE = 16000
VOICE_TARGET_SECONDS = 2
VOICE_N_MFCC = 40
VOICE_N_FRAMES = 32
VOICE_INPUT_SIZE = VOICE_N_MFCC * VOICE_N_FRAMES


def _preprocess_audio(audio, sr=VOICE_SAMPLE_RATE):
    """与 voice_dataset / 训练一致的 MFCC 预处理，返回 (40, 32)。"""
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    target_length = VOICE_SAMPLE_RATE * VOICE_TARGET_SECONDS
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=VOICE_N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = mfcc[:, :VOICE_N_FRAMES]
    if mfcc.shape[1] < VOICE_N_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, VOICE_N_FRAMES - mfcc.shape[1])),
            mode="constant",
        )
    return mfcc


def audio_to_mfcc_matrix(audio, sr=VOICE_SAMPLE_RATE):
    """供 GUI 显示 MFCC 热力图。"""
    if sr != VOICE_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VOICE_SAMPLE_RATE)
        sr = VOICE_SAMPLE_RATE
    return _preprocess_audio(np.asarray(audio, dtype=np.float32), sr)


def audio_to_feature_tensor(audio, sr=VOICE_SAMPLE_RATE, device=None):
    """将录音数组转为模型输入张量，shape (1, 1280)。"""
    mfcc = audio_to_mfcc_matrix(audio, sr)
    tensor = torch.tensor(mfcc.flatten(), dtype=torch.float32).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def extract_feature(file_path, device=None):
    """从 wav 文件提取特征（test.py / 命令行用）。"""
    audio, sr = librosa.load(file_path, sr=VOICE_SAMPLE_RATE)
    return audio_to_feature_tensor(audio, sr, device)
