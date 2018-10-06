import os
import librosa
import numpy as np


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


def frequency_difference(note):
    notes_freqs = {
        'A4' : 440,
        'As4' : 466.154,
        'B4' : 493.883,
        'C4' : 261.626,
        'Cs4' : 277.183,
        'D4' : 293.665,
        'Ds4' : 311.127,
        'E4' : 329.628,
        'F4' : 349.228,
        'Fs4' : 369.994,
        'G4' : 391.995,
        'Gs4' : 415.305
    }

    return notes_freqs['A4'] - notes_freqs[note]


def shift_frequency(y, frequency, fs, num_samples):
    shift_factor = num_samples / fs
    frequency_shift = round(shift_factor * frequency)

    shifted_fft = np.zeros(y.shape)

    for n in range(0, y.shape[0] - abs(frequency_shift) - 1):
        if n + frequency_shift >= 0:
            shifted_fft[n + frequency_shift] = y[n]

    return shifted_fft


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
    cnt = 1
    for i, data in enumerate(file_data):
        file_name = data["path"]
        # Load audio file as a floating point time series, Sampling rate=22050 Hz, converted to mono
        y, sr = librosa.load(file_name, sr=22050)
        num_samples = y.shape[0]
        
        # normalize by dividing the time domain signal with it's maximum value
        y = np.divide(y,np.amax(y))
        
        # convert to time frequency representation by short time fourier transform
        y_fft = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024, window='hann')
        y_fft = abs(y_fft)
        y_fft = y_fft[:round(num_samples / 2)]

        feature_vector = create_feature_vector(y_fft, num_chunks)

        features[:, i] = feature_vector
        instruments = np.append(instruments, data["instrument"])

    keys = np.unique(instruments)
    values = np.arange(len(keys))
    instrument_label_map = dict(zip(keys, values))

    labels = np.zeros((1, len(file_data)))
    for i, insts in enumerate(instruments):
        labels[0, i] = instrument_label_map[insts]

    return features, labels
