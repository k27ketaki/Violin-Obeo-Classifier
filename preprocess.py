import os
import librosa
import librosa.display as disp
import numpy as np
import matplotlib.pyplot as plt

def process_filenames(directory):
    file_names = os.listdir(directory)
    file_data = np.array([])
    for file in file_names:
        if file.startswith(".") or os.path.isdir(file):
            print("Ignored:", file)
            file_names.remove(file)


    for file in file_names:
        data = {}
        data["path"] = os.path.join(directory, file)
        file = file.split('_')
        data["instrument"] = file[0]
        data["note"] = file[1]
        file_data = np.append(file_data, data)

    return file_data


def create_feature_vector(y, num_chunks):
    feature_vector = np.zeros(num_chunks)
    chunk_size = int(y.shape[0] / num_chunks)

    for n in range(num_chunks):
        chunk = y[n*chunk_size : (n+1)*chunk_size]
        feature_vector[n] = np.mean(chunk)

    return feature_vector


def process_training_data(directory, num_chunks):
    file_data = process_filenames(directory)
    features = np.zeros((num_chunks, len(file_data)))
    instruments = np.array([])
    flag = 1
    for i, data in enumerate(file_data):
        file_name = data["path"]
        # Load audio file as a floating point time series, Sampling rate=22050 Hz, converted to mono
        y_time, sr = librosa.load(file_name, sr=22050)
        num_samples = y_time.shape[0]
        
        # normalize by dividing the time domain signal with it's maximum value
        y_time = np.divide(y_time,np.amax(y_time))
        
        # Data is windowed to minimize spectral leakage which happens when you try to Fourier-transform non-cyclical data
        # convert to time frequency representation by short time fourier transform
        y_stft = librosa.core.stft(y_time, n_fft=1024, hop_length=512, win_length=1024, window='hann')
        y_stft = abs(y_stft)
        
        #feature_vector = create_feature_vector(y_stft, num_chunks)

        #features[:, i] = feature_vector
        S= librosa.feature.melspectrogram(y=y_time, S=y_stft, n_fft=1024, hop_length=512, power=2.0)
        if flag == 1:
        	plt.figure(figsize=(10,4))
        	disp.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel',fmax=8000,x_axis='time')
        	plt.colorbar(format='%+2.0f dB')
        	plt.title('Mel spectrogram')
        	plt.tight_layout()
        	flag = 2
        
        instruments = np.append(instruments, data["instrument"])

    keys = np.unique(instruments)
    values = np.arange(len(keys))
    instrument_label_map = dict(zip(keys, values))

    labels = np.zeros((1, len(file_data)))
    for i, insts in enumerate(instruments):
        labels[0, i] = instrument_label_map[insts]

    return features, labels
