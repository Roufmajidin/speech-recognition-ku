# import librosa
# import matplotlib.pyplot as plt

# # Load audio file
# audio_file = "bener.m4a"

# # Load dan normalisasi audio
# data, sr = librosa.load(audio_file, sr=None)
# data_normalized = librosa.util.normalize(data)

# # Ekstraksi MFCC
# mfcc = librosa.feature.mfcc(y=data_normalized, sr=sr, n_mfcc=13)

# # Konversi indeks frame menjadi waktu
# t = librosa.frames_to_time(range(mfcc.shape[1]), sr=sr)

# # Plot MFCC untuk audio
# plt.figure(figsize=(12, 6))
# for i in range(mfcc.shape[0]):
#     plt.plot(t, mfcc[i], label=f'MFCC {i+1}')
# plt.xlabel('Time (s)')
# plt.ylabel('MFCC Coefficients')
# plt.legend()
# plt.grid(True)
# plt.show()


import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load audio files
audio_file1 = "uploads/audio1.m4a"
audio_file2 = "uploads/bismis.wav"

# Load dan normalisasi audio pertama
data1, sr1 = librosa.load(audio_file1, sr=44000)
data1_normalized = librosa.util.normalize(data1)

# Load dan normalisasi audio kedua
data2, sr2 = librosa.load(audio_file2, sr=16000)
data2_normalized = librosa.util.normalize(data2)

# Ekstraksi MFCC untuk audio pertama
mfcc1 = librosa.feature.mfcc(y=data1_normalized, sr=sr1, n_mfcc=5)

# Ekstraksi MFCC untuk audio kedua
mfcc2 = librosa.feature.mfcc(y=data2_normalized, sr=sr2, n_mfcc=5)

# Konversi indeks frame menjadi waktu
t1 = librosa.frames_to_time(range(mfcc1.shape[1]), sr=sr1)
t2 = librosa.frames_to_time(range(mfcc2.shape[1]), sr=sr2)

# Plot MFCC untuk kedua audio
plt.figure(figsize=(12, 8))
for i in range(mfcc1.shape[0]):
    plt.plot(t1, mfcc1[i], label=f'MFCC {i+1} - Audio 1')
for i in range(mfcc2.shape[0]):
    plt.plot(t2, mfcc2[i], label=f'MFCC {i+1} - Audio 2', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.legend()
plt.grid(True)
plt.show()
