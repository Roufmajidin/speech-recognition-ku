# # framing dan windwinf
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import librosa
# # import librosa.display
# # import sklearn.preprocessing

# # # Load audio file
# # audio_data = 'middun.wav'
# # x, sr = librosa.load(audio_data, sr=None)

# # # Define frame parameters
# # frame_size = 0.025  # Frame size in seconds (e.g., 25 ms)
# # frame_stride = 0.01  # Stride between frames in seconds (e.g., 10 ms)
# # frame_length, frame_step = int(frame_size * sr), int(frame_stride * sr)

# # # Compute frames
# # def compute_frames(signal, frame_length, frame_step):
# #     num_samples = len(signal)
# #     frames = []
# #     for i in range(0, num_samples - frame_length + 1, frame_step):
# #         frame = signal[i:i+frame_length]
# #         frames.append(frame)
# #     return np.array(frames)

# # frames = compute_frames(x, frame_length, frame_step)

# # # Compute frame times
# # frame_times = np.arange(len(frames)) * frame_step / sr

# # # Plot the audio waveform with frame boundaries
# # plt.figure(figsize=(12, 4))
# # librosa.display.waveshow(x, sr=sr, alpha=0.4, color='blue')
# # for i in range(len(frames)):
# #     time_start = i * frame_step / sr  # Waktu mulai frame relatif terhadap sinyal audio
# #     plt.axvline(x=time_start, color='red', linestyle='--', alpha=0.5)
# # plt.title('Audio Waveform with Frame Boundaries')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')

# # # Set the frame axis
# # plt.gca().secondary_xaxis('top', functions=(lambda x: x * sr / frame_step, lambda x: x))
# # plt.xlabel('Frame')

# # plt.show()


# # # FTT
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import librosa
# # import librosa.display

# # # Load audio file
# # audio_data = 'middun.wav'
# # x, sr = librosa.load(audio_data, sr=None)

# # # Define frame parameters
# # frame_size = 0.025  # Frame size in seconds (e.g., 25 ms)
# # frame_stride = 0.01  # Stride between frames in seconds (e.g., 10 ms)
# # frame_length, frame_step = int(frame_size * sr), int(frame_stride * sr)

# # # Compute frames
# # def frames(signal, frame_length, frame_step):
# #     num_samples = len(signal)
# #     frames = []
# #     for i in range(0, num_samples - frame_length + 1, frame_step):
# #         frame = signal[i:i+frame_length]
# #         frames.append(frame)
# #     return np.array(frames)

# # # Get frames
# # audio_frames = frames(x, frame_length, frame_step)

# # # Compute FFT for each frame
# # def fft_transform(frames):
# #     return np.fft.rfft(frames)

# # audio_fft = np.abs(fft_transform(audio_frames))

# # # Plot the FFT spectrum
# # plt.figure(figsize=(12, 6))
# # plt.plot(audio_fft.T, color='blue', alpha=0.7)
# # plt.title('FFT Spectrum dari Audio Frames')
# # plt.xlabel('Frequency Bin')
# # plt.ylabel('Amplitude')
# # plt.show()


# # mel 
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import librosa
# # import librosa.display

# # # Load audio file
# # audio_data = 'middun.wav'
# # x, sr = librosa.load(audio_data, sr=None)

# # # Define frame parameters
# # frame_size = 0.025  # Frame size in seconds (e.g., 25 ms)
# # frame_stride = 0.01  # Stride between frames in seconds (e.g., 10 ms)
# # frame_length, frame_step = int(frame_size * sr), int(frame_stride * sr)

# # # Compute frames
# # def frames(signal, frame_length, frame_step):
# #     num_samples = len(signal)
# #     frames = []
# #     for i in range(0, num_samples - frame_length + 1, frame_step):
# #         frame = signal[i:i+frame_length]
# #         frames.append(frame)
# #     return np.array(frames)

# # # Get frames
# # audio_frames = frames(x, frame_length, frame_step)

# # # Compute FFT for each frame
# # def fft_transform(frames):
# #     return np.fft.rfft(frames)

# # audio_fft = fft_transform(audio_frames)

# # # Compute Mel-frequency wrapping
# # num_filters = 40  # Jumlah filter-bank Mel
# # mel_filters = librosa.filters.mel(sr=sr, n_fft=frame_length, n_mels=num_filters)
# # mel_spectrum = np.dot(mel_filters, np.abs(audio_fft.T)**2)

# # # Log-mel spectrum
# # log_mel_spectrum = np.log(mel_spectrum + 1e-10)

# # # Print the shape of log-mel spectrum
# # print("Log-Mel Spectrum shape:", log_mel_spectrum.shape)

# # # Plot the log-mel spectrum
# # plt.figure(figsize=(12, 6))
# # plt.imshow(log_mel_spectrum, aspect='auto', origin='lower', cmap='viridis')
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Log-Mel Spectrum')
# # plt.xlabel('Frame')
# # plt.ylabel('Mel Filter')
# # plt.show()


# # dct

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import librosa
# # import librosa.display

# # # Load audio file
# # audio_data = 'middun.wav'
# # x, sr = librosa.load(audio_data, sr=None)

# # # Define frame parameters
# # frame_size = 0.025  # Frame size in seconds (e.g., 25 ms)
# # frame_stride = 0.01  # Stride between frames in seconds (e.g., 10 ms)
# # frame_length, frame_step = int(frame_size * sr), int(frame_stride * sr)

# # # Compute frames
# # def frames(signal, frame_length, frame_step):
# #     num_samples = len(signal)
# #     frames = []
# #     for i in range(0, num_samples - frame_length + 1, frame_step):
# #         frame = signal[i:i+frame_length]
# #         frames.append(frame)
# #     return np.array(frames)

# # # Get frames
# # audio_frames = frames(x, frame_length, frame_step)

# # # Compute FFT for each frame
# # def fft_transform(frames):
# #     return np.fft.rfft(frames)

# # audio_fft = fft_transform(audio_frames)

# # # Compute Mel-frequency wrapping
# # num_filters = 40  # Jumlah filter-bank Mel
# # mel_filters = librosa.filters.mel(sr=sr, n_fft=frame_length, n_mels=num_filters)
# # mel_spectrum = np.dot(mel_filters, np.abs(audio_fft.T)**2)

# # # Log-mel spectrum
# # log_mel_spectrum = np.log(mel_spectrum + 1e-10)

# # # Ekstraksi MFCC
# # num_ceps = 13  # Jumlah koefisien cepstral MFCC
# # mfcc = librosa.feature.mfcc(S=log_mel_spectrum, n_mfcc=num_ceps)

# # # Cepstral Liftering
# # num_lift = 22  # Jumlah koefisien liftering
# # cepstral_lifter = 1 + (num_lift / 2) * np.sin(np.pi * np.arange(1, num_lift + 1) / num_lift)

# # # Ubah dimensi cepstral_lifter agar sesuai dengan mfcc
# # cepstral_lifter = cepstral_lifter.reshape(1, -1)

# # # Terapkan cepstral liftering pada setiap frame mfcc secara terpisah
# # mfcc_lifted = mfcc * cepstral_lifter[:, :mfcc.shape[1]]  # Memotong cepstral lifter untuk sesuai dengan jumlah koefisien MFCC

# # # Plot the lifted MFCC coefficients
# # plt.figure(figsize=(12, 6))
# # librosa.display.specshow(mfcc_lifted, sr=sr, hop_length=frame_step, x_axis='time')
# # plt.colorbar()
# # plt.title('Lifted MFCC')
# # plt.xlabel('Time (s)')
# # plt.ylabel('MFCC Coefficients')
# # plt.show()


# # dct
# import numpy as np
# import librosa
# from scipy.fftpack import dct

# # Load audio file
# audio_data, sampling_rate = librosa.load('middun.wav')

# # Define frame parameters
# frame_size = 0.025  # Frame size in seconds
# frame_stride = 0.01  # Stride between frames in seconds
# frame_length, frame_step = int(frame_size * sampling_rate), int(frame_stride * sampling_rate)

# # Compute frames
# def frames(signal, frame_length, frame_step):
#     num_samples = len(signal)
#     frames = []
#     for i in range(0, num_samples - frame_length + 1, frame_step):
#         frame = signal[i:i+frame_length]
#         frames.append(frame)
#     return np.array(frames)

# # Get frames
# audio_frames = frames(audio_data, frame_length, frame_step)

# # Compute FFT for each frame
# def fft_transform(frames):
#     return np.fft.rfft(frames)

# audio_fft = fft_transform(audio_frames)

# # Compute Mel-frequency wrapping
# num_filters = 40  # Number of Mel filter-bank
# mel_filters = librosa.filters.mel(sr=sampling_rate, n_fft=frame_length, n_mels=num_filters)
# mel_spectrum = np.dot(mel_filters, np.abs(audio_fft.T)**2)

# # Log-mel spectrum
# log_mel_spectrum = np.log(mel_spectrum + 1e-10)

# # Discrete Cosine Transform (DCT)
# num_ceps = 13  # Number of MFCC coefficients
# mfcc = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]  # DCT-II
# print("Dimensi audio_frames:", audio_frames.shape)
# print("Nilai mel_spectrum:", mel_spectrum)
# print("Dimensi mfcc:", mfcc.shape)

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from pydub import AudioSegment
# Path ke file audio
audio_path = 'uploads/bismillah.mp3'
output_directory = 'output_directory'
# Nama file output
output_file_name = 'output_audio.wav'

# Memastikan direktori output ada
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# asli
# Membaca file audio dan metadata menggunakan soundfile
with sf.SoundFile(audio_path) as f:
    sample_rate = f.samplerate
    bit_depth = f.subtype_info
    audio_data = f.read()
# Menampilkan informasi bit depth
print(f"Bit depth asli: {bit_depth}")
print(f"Frekuensi pengambilan sampel: {sample_rate} Hz")
print(f"Jumlah sampel: {len(audio_data)}")
print(f"Durasi sinyal: {len(audio_data) / sample_rate} detik")


# Memuat file audio

#TODO 1 Sampling keadaan semula
audio_data, sampling_rate = librosa.load(audio_path, sr=None)

# Hitung durasi audio
duration = len(audio_data) / sampling_rate

# Definisikan waktu sampel
num_samples = len(audio_data)
t = np.linspace(0, duration, num_samples)
# Durasi sinyal audio asli
duration_original = len(audio_data) / sampling_rate


# Plot sinyal audio
#TODO 1.1 Sampling keadaan frekuensi kosong


def auto_crop_audio(audio_signal, sampling_rate, n_fft=2048, hop_length=512):
    # Hitung STFT dari sinyal audio
    stft = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
    # Hitung energi dari STFT
    energy = np.sum(np.abs(stft)**2, axis=0)

    # Tentukan ambang batas energi
    threshold = np.mean(energy)

    # Identifikasi frame pertama yang melebihi ambang batas energi
    start_frame = np.argmax(energy > threshold)

    # Tentukan indeks sampel awal dari frame ini
    start_sample = start_frame * hop_length

    # Potong sinyal audio dari indeks awal yang ditentukan
    audio_signal_cropped = audio_signal[start_sample:]

    return audio_signal_cropped

# Tentukan ambang batas RMS untuk deteksi bagian diam
threshold = 0.02
cropped_audio_signal = auto_crop_audio(audio_data, sampling_rate)


# Plot sinyal audio sebelum dan sesudah pemotongan bagian yang diam
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(audio_data, sr=sampling_rate, color='blue')
plt.title('Sinyal Audio Asli')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')

# plt.subplot(2, 1, 2)
# librosa.display.waveshow(cropped_audio_signal, sr=sampling_rate, color='blue')
# plt.title('Sinyal Audio Setelah Pemotongan Bagian yang Diam')
# plt.xlabel('Waktu (s)')
# plt.ylabel('Amplitudo')

# # Menambahkan keterangan
# plt.text(0.5, 0.9, f'Durasi asli: {duration_original:.2f} detik', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
# plt.text(0.5, 0.8, f'Durasi setelah auto cropping: {duration_cropped:.2f} detik', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
# plt.text(0.5, 0.7, 'Deteksi jeda frekuensi rendah', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)


# plt.tight_layout()
# plt.show()
# Menampilkan informasi sinyal digital
print(f"Sinyal digital: {audio_data[:10]} ...")  # Menampilkan 10 sampel pertama
print(f"Jumlah sampel: {len(audio_data)}")
print(f"Durasi sinyal: {len(audio_data) / sample_rate} detik")
print(f"Frekuensi pengambilan sampel: {sample_rate} Hz")


# TODO 2 Quantizing ke 32 bith
# Menentukan bit depth untuk kuantisasi
bit_depth = 32 
quantization_levels = 2 ** bit_depth

normalized_audio = (audio_data - audio_data.min()) / (audio_data.max() - audio_data.min())

quantized_audio = np.round(normalized_audio * (quantization_levels - 1)) / (quantization_levels - 1)

# Mengembalikan sinyal ke rentang asli

# Menampilkan informasi sinyal digital yang telah dikuantisasi
# print(f"Sinyal digital yang dikuantisasi: {quantized_audio[:10]} ...")  # Menampilkan 10 sampel pertama
# print(f"Jumlah sampel: {len(quantized_audio)}")
# print(f"Durasi sinyal: {len(quantized_audio) / sample_rate} detik")
# print(f"Kedalaman Bit setelah dikuantisasi: {bit_depth} bit")
# print(f"Frekuensi pengambilan sampel: {sampling_rate} Hz")
# print(f"Jumlah sampel: {len(quantized_audio)}")
# print(f"Durasi sinyal: {len(quantized_audio) / sampling_rate} detik")

# Plot sinyal digital

#TODO 3 Encoding
output_audio_path = os.path.join(output_directory, output_file_name)

# Memuat file audio menggunakan pydub
audio = AudioSegment.from_file(audio_path)
binary_representation = [format(int(sample * (2**bit_depth - 1)), '032b') for sample in quantized_audio]

# Cetak representasi biner
print("Representasi Biner dari Sinyal Audio yang Telah Dikuantisasi:")
for i in range(10):  # Mencetak 10 sampel pertama sebagai contoh
    print(f"Sampel {i+1}: {binary_representation[i]}")

# Mengonversi dan menyimpan file audio ke format WAV
audio.export(output_audio_path, format="wav")

# plt.figure(figsize=(14, 5))
# plt.plot(audio_data)
# plt.title("Sinyal Digital")
# plt.xlabel("Sampel")
# plt.ylabel("Amplitudo")
# plt.show()
