import os
import librosa
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class VoiceDigitDataset(Dataset):

    def __init__(self, data_dir):

        self.data = []

        for file_name in os.listdir(data_dir):

            if file_name.endswith(".wav"):

                path = os.path.join(data_dir, file_name)

                # 标签
                label = int(file_name[0])

                self.data.append((path, label))

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        path, label = self.data[idx]

        audio, sr = librosa.load(
            path,
            sr=16000
        )

        # 音量归一化
        audio = librosa.util.normalize(audio)

        # 去静音
        audio, _ = librosa.effects.trim(
            audio,
            top_db=20
        )

        # 固定长度
        target_length = 16000 * 2

        if len(audio) < target_length:

            pad_length = target_length - len(audio)

            audio = np.pad(
                audio,
                (0, pad_length),
                mode='constant'
            )

        else:

            audio = audio[:target_length]

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        # 标准化
        mfcc = (
                       mfcc - np.mean(mfcc)
               ) / (
                       np.std(mfcc) + 1e-8
               )

        # 固定帧数
        mfcc = mfcc[:, :32]

        if mfcc.shape[1] < 32:
            pad_width = 32 - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )

        mfcc = mfcc.flatten()

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


def get_voice_loader(data_dir, batch_size=32):

    dataset = VoiceDigitDataset(data_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader