import os
import sys
from typing import Tuple
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.common.utilities import *

def extract_data(data, labels, class_labels, val_size=0.1, test_size=0.2, random=42):
    # train:val:test = 7:1:2
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=random,
        stratify=labels)

    val_size = val_size / (1 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=random,
        stratify=y_train)

    return np.array(x_train), np.array(x_val), np.array(x_test), np.array(y_train), np.array(y_val), np.array(y_test), len(class_labels)

def bind_npys(input_path=ENTERFACE_VISUAL_NPY_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH , class_labels=ENTERFACE_LABELS):
    
    file_name = get_labels_name(class_labels)
    x_path = output_path + file_name + 'x.npy'
    y_path = output_path + file_name + 'y.npy'
    if os.path.isfile(x_path) == True and os.path.isfile(y_path) == True:
        print(x_path + ' is captured')
        print(y_path + ' is captured')
        return 0
    
    remove_file(x_path)
    remove_file(y_path)

    data = []
    labels = []
    names = []
    for i, directory in enumerate(class_labels):
        folder = input_path + directory + '/'
        print('started reading folder %s' % directory)
        for filename in os.listdir(folder):
            if '.npy' in filename:
                filepath = folder + filename
                _data = np.load(filepath, allow_pickle=True)
                data.append(_data)
                labels.append(i)
                names.append(filename)
        print('ended reading folder %s' % directory)
    np.save(x_path, data)
    np.save(y_path, labels)
    print('ended capturing')

# Visual & Audio

def extract_visual_audio_data(visual_data_path: str = ENTERFACE_VISUAL_NPY_FSIM_DATA_PATH, audio_data_path: str = ENTERFACE_AUDIO_DATA_PATH, 
                              class_labels: Tuple = ENTERFACE_LABELS, random: int = 42):
    data, labels = get_visual_audio_data(visual_data_path, audio_data_path, class_labels=class_labels)
    return extract_data(data, labels, class_labels, random=random)

def get_visual_audio_data(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = ENTERFACE_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    for i, directory in enumerate(class_labels):
        visual_folder = visual_data_path + directory + '/'
        audio_folder = audio_data_path + directory + '/'
        print('started reading folder %s' % directory)
        for visual_filename in os.listdir(visual_folder):
            filename = visual_filename.split('.')[0]
            if '.npy' in visual_filename:
                visual_filepath = visual_folder + visual_filename
                audio_filepath = audio_folder + filename + '.wav'
                if os.path.isfile(audio_filepath) == True:
                    visual_data = np.load(visual_filepath, allow_pickle=True)
                    audio_data = audio_filepath
                    data.append([visual_data, audio_data])
                    labels.append(i)
        print('ended reading folder %s' % directory)
    
    return np.array(data), np.array(labels)

def split_visual_audio_data(data: np.ndarray):
    visual_data = []
    audio_data = []
    for d in data:
        visual_data.append(d[0])
        audio_data.append(d[1])   

    return np.array(visual_data), np.array(audio_data)

# Audio

def extract_audio_data(data_path: str = ENTERFACE_AUDIO_DATA_PATH, class_labels: Tuple = ENTERFACE_LABELS, random: int = 42):
    data, labels = get_audio_data(data_path, class_labels=class_labels)
    return extract_data(data, labels, class_labels, random=random)

def get_audio_data(data_path: str, class_labels: Tuple = ENTERFACE_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    names = []
    for i, directory in enumerate(class_labels):
        folder = data_path + directory + '/'
        print('started reading folder %s' % directory)
        for filename in os.listdir(folder):
            if '.wav' in filename:
                filepath = folder + filename
                data.append(filepath)
                labels.append(i)
        print('ended reading folder %s' % directory)
    
    return np.array(data), np.array(labels)

def get_signal_of_audio_data(data, sr=None):
    _sr = 0
    _data = []
    for d in data:
        signal, _sr = librosa.load(d, sr=sr)
        _data.append(signal)
    return np.array(_data), _sr

def get_feature_vector_of_audio_data(signals, sr, mel_filters: int = 224, mel_len: int = 224):
    _data = []
    for s in signals:
        feature_vector = get_feature_vector_from_logmelspec(signal=s, sr=sr, mel_filters=mel_filters, mel_len=mel_len)
        #feature_vector = feature_vector.reshape(feature_vector.shape[0],feature_vector.shape[1],1)
        #_data.append(feature_vector)
        feature_cube = get_feature_vector_from_logmelspec_delta(feature_vector)
        _data.append(feature_cube)
    return np.array(_data)

def get_feature_vector_from_logmelspec(signal, sr, mel_filters: int = 224, mel_len: int = 224):
    s_len = len(signal)
    
    n_fft = 2048
    hop_length = s_len // (mel_len - 1)
    #hop_length = 512
    mean_signal_length = (mel_len - 1) * hop_length

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    logmelspec = librosa.feature.melspectrogram(signal, sr, n_fft=n_fft, hop_length=hop_length, n_mels=mel_filters)

    return librosa.power_to_db(logmelspec)

def get_feature_vector_from_logmelspec_delta(data):
    data_delta = librosa.feature.delta(data)
    data_delta_delta = librosa.feature.delta(data_delta)

    data = data.reshape(data.shape[0],data.shape[1],1)
    data_delta = data_delta.reshape(data_delta.shape[0],data_delta.shape[1],1)
    data_delta_delta = data_delta_delta.reshape(data_delta_delta.shape[0],data_delta_delta.shape[1],1)
    
    return np.concatenate([data,data_delta,data_delta_delta], axis=2)

# Visual

def extract_visual_data(data_path: str = ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels: Tuple = ENTERFACE_LABELS, random: int = 42):
    data, labels = get_visual_data(data_path, class_labels=class_labels)
    return extract_data(data, labels, class_labels, random=random)

def get_visual_data(data_path: str, class_labels: Tuple = ENTERFACE_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    names = []

    file_name = get_labels_name(class_labels)
    x_path = data_path + file_name + 'x.npy'
    y_path = data_path + file_name + 'y.npy'
    if os.path.isfile(x_path) != True or os.path.isfile(y_path) != True:
        bind_npys(input_path=data_path, output_path=data_path, class_labels=class_labels)

    data = np.load(x_path, allow_pickle=True)
    labels = np.load(y_path, allow_pickle=True)

    return np.array(data), np.array(labels)

def extract_predicted_datas(data, model, timesteps = 16):    
    if timesteps == 0:
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                if timesteps < len(data[i]):
                    timesteps = len(data[i])
    
    print('started predicting data')
    predicted_data = extract_predicted_data(data, model, timesteps)
    predicted_data = predicted_data.reshape((predicted_data.shape[0],predicted_data.shape[1],1))
    print('ended predicting data')

    return predicted_data

def extract_predicted_data(data, model, timesteps):
    predicted_data = []
    for i in range(0, len(data)):
        results = []
        for j in range(0, timesteps):
            if j < len(data[i]):
                prediction = np.argmax(model.predict(np.array([data[i][j]])))
            else:
                prediction = -1
            results.append(prediction)
        predicted_data.append(tuple(results))

    return np.array(predicted_data)

def extract_predicted_datas_2(data, models):  
    print('started predicting data')
    predicted_data = extract_predicted_data_2(data, models)
    predicted_data = predicted_data.reshape((predicted_data.shape[0],predicted_data.shape[1],1))
    print('ended predicting data')

    return predicted_data

def extract_predicted_data_2(data, models):
    predicted_data = []
    for i in range(0, len(data)):
        results = []
        for j in range(0, len(data[i])):
            prediction = np.argmax(models[j].predict(np.array([data[i][j]])))
            results.append(prediction)
        predicted_data.append(tuple(results))

    return np.array(predicted_data)

def limit_visual_data(data, limit_farme = 16):
    limited_data = []
    for i in range(0, len(data)):
        results = []
        if len(data[i]) > limit_farme:
            start = int(len(data[i]) / 2) - int(limit_farme / 2)
            end = int(len(data[i]) / 2) + int(limit_farme / 2)
            for j in range(start, end):
                results.append(data[i][j])
        elif len(data[i]) < limit_farme:
            start_repeat = int(limit_farme / 2) - int(len(data[i]) / 2)
            end_repeat = int(len(data[i]) / 2) + int(limit_farme / 2) - len(data[i])
            for j in range(0, start_repeat):
                results.append(data[i][0])
            for j in range(0, len(data[i])):
                results.append(data[i][j])
            for j in range(0, end_repeat):
                results.append(data[i][len(data[i])-1])
        else:
            results = data[i]
        limited_data.append(results)
    
    limited_data = np.array(limited_data).reshape((len(limited_data), limit_farme, len(limited_data[0][0]), len(limited_data[0][0][0]), len(limited_data[0][0][0][0])))
    return limited_data

def transform_visual_data_2d(data, labels):
    data_2d = []
    labels_2d = []
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            data_2d.append(data[i][j])
            labels_2d.append(labels[i])

    return np.array(data_2d), np.array(labels_2d)

def transform_visual_data_2d_group(data, labels):
    data_2d_group = []
    labels_2d_group = []
    for i in range(0, len(data[0])):
        data_2d = []
        labels_2d = []
        for j in range(0, len(data)):
            data_2d.append(data[j][i])
            labels_2d.append(labels[j])
        data_2d_group.append(data_2d)
        labels_2d_group.append(labels_2d)

    return np.array(data_2d_group), np.array(labels_2d_group)

def gaussian_noise(image, mean=0, sigma=3):
    img = np.copy(image)
    noise = np.random.normal(mean, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype('uint8')

def gaussian_noise_2(data, labels):
    _data = []
    _labels = []
    for k in range(0, 4):
        for i in range(0, len(data)):
            gaussian_outputs = []
            for j in range(0, len(data[i])):
                gaussian_outputs.append(gaussian_noise(data[i][j]))
            _data.append(gaussian_outputs)
            _labels.append(labels[i])
    for i in range(0, len(data)):
        _data.append(data[i])
        _labels.append(labels[i])
        
    return np.array(_data), np.array(_labels)

def get_visual_audio_data_fold(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = RAVDESS_LABELS, fold=0) -> Tuple[np.ndarray, np.ndarray]:
    seed_torch(seed=2020)
    if 'enterface' in visual_data_path:
        return get_visual_audio_enterface_data_fold(visual_data_path, audio_data_path, class_labels, fold)
    
    if 'BAUM1s' in visual_data_path or 'BAUM1a' in visual_data_path:
        return get_visual_audio_baum1s_data_fold(visual_data_path, audio_data_path, class_labels, fold)
    
    if 'savee' in visual_data_path:
        return get_visual_audio_savee_data_fold(visual_data_path, audio_data_path, class_labels, fold)

    return get_visual_audio_ravdess_data_fold(visual_data_path, audio_data_path, class_labels, fold)    

def get_visual_audio_ravdess_data_fold(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = RAVDESS_LABELS, fold=0) -> Tuple[np.ndarray, np.ndarray]:
    actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, directory in enumerate(class_labels):
        visual_folder = visual_data_path + directory + '/'
        audio_folder = audio_data_path + directory + '/'
        print('started reading folder %s' % directory)
        for visual_filename in os.listdir(visual_folder):
            filename = visual_filename.split('.')[0]
            if '.npy' in visual_filename:
                visual_filepath = visual_folder + visual_filename
                audio_filepath = audio_folder + filename + '.wav'
                if os.path.isfile(audio_filepath) == True:
                    visual_data = np.load(visual_filepath, allow_pickle=True)
                    audio_data = audio_filepath
                    for j in range(len(actors_per_fold)):
                        if j == fold:
                            for actor in actors_per_fold[j]:
                                if int(filename.split('-')[-1]) == actor:
                                    x_test.append([visual_data, audio_data])
                                    y_test.append(i)
                        else:
                            for actor in actors_per_fold[j]:
                                if int(filename.split('-')[-1]) == actor:
                                    x_train.append([visual_data, audio_data])
                                    y_train.append(i)

        print('ended reading folder %s' % directory)
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(class_labels)

def get_visual_audio_enterface_data_fold(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = ENTERFACE_LABELS, fold=0) -> Tuple[np.ndarray, np.ndarray]:
    actors_per_fold = {
        0: [1, 4, 10, 29, 15, 20, 25, 35, 41],
        1: [2, 5, 11, 31, 16, 21, 27, 36, 42],
        2: [3, 7, 12, 33, 17, 22, 30, 37, 43],
        3: [8, 26, 13, 44, 18, 23, 32, 38],
        4: [9, 28, 14, 19, 24, 34, 39, 40],
    }
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, directory in enumerate(class_labels):
        visual_folder = visual_data_path + directory + '/'
        audio_folder = audio_data_path + directory + '/'
        print('started reading folder %s' % directory)
        for visual_filename in os.listdir(visual_folder):
            filename = visual_filename.split('.')[0]
            if '.npy' in visual_filename:
                visual_filepath = visual_folder + visual_filename
                audio_filepath = audio_folder + filename + '.wav'
                if os.path.isfile(audio_filepath) == True:
                    visual_data = np.load(visual_filepath, allow_pickle=True)
                    audio_data = audio_filepath
                    for j in range(len(actors_per_fold)):
                        if j == fold:
                            for actor in actors_per_fold[j]:
                                if filename.split('_')[0] == 's' + str(actor):
                                    x_test.append([visual_data, audio_data])
                                    y_test.append(i)
                        else:
                            for actor in actors_per_fold[j]:
                                if filename.split('_')[0] == 's' + str(actor):
                                    x_train.append([visual_data, audio_data])
                                    y_train.append(i)

        print('ended reading folder %s' % directory)
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(class_labels)

def get_visual_audio_baum1s_data_fold(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = BAUM1S_LABELS, fold=0) -> Tuple[np.ndarray, np.ndarray]:
    actors_per_fold = {
        0: [1, 25, 6, 30, 14, 19, 24],
        1: [2, 26, 7, 31, 15, 20],
        2: [3, 27, 8, 11, 16, 21],
        3: [4, 28, 9, 12, 17, 22],
        4: [5, 29, 10, 13, 18, 23],
    }
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, directory in enumerate(class_labels):
        visual_folder = visual_data_path + directory + '/'
        audio_folder = audio_data_path + directory + '/'
        print('started reading folder %s' % directory)
        for visual_filename in os.listdir(visual_folder):
            filename = visual_filename.split('.')[0]
            if '.npy' in visual_filename:
                visual_filepath = visual_folder + visual_filename
                audio_filepath = audio_folder + filename + '.wav'
                if os.path.isfile(audio_filepath) == True:
                    visual_data = np.load(visual_filepath, allow_pickle=True)
                    audio_data = audio_filepath
                    for j in range(len(actors_per_fold)):
                        if j == fold:
                            for actor in actors_per_fold[j]:
                                if filename.split('_')[0] == 'S' + str(actor).zfill(3):
                                    x_test.append([visual_data, audio_data])
                                    y_test.append(i)
                        else:
                            for actor in actors_per_fold[j]:
                                if filename.split('_')[0] == 'S' + str(actor).zfill(3):
                                    x_train.append([visual_data, audio_data])
                                    y_train.append(i)

        print('ended reading folder %s' % directory)
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(class_labels)

def get_visual_audio_savee_data_fold(visual_data_path: str, audio_data_path: str, 
                          class_labels: Tuple = SAVEE_LABELS, fold=0) -> Tuple[np.ndarray, np.ndarray]:
    actors_per_fold = {
        0: 'DC',
        1: 'JE',
        2: 'JK',
        3: 'KL',
        4: 'KL',
    }
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, directory in enumerate(class_labels):
        visual_folder = visual_data_path + directory + '/'
        audio_folder = audio_data_path + directory + '/'
        print('started reading folder %s' % directory)
        for visual_filename in os.listdir(visual_folder):
            filename = visual_filename.split('.')[0]
            if '.npy' in visual_filename:
                visual_filepath = visual_folder + visual_filename
                audio_filepath = audio_folder + filename + '.wav'
                if os.path.isfile(audio_filepath) == True:
                    visual_data = np.load(visual_filepath, allow_pickle=True)
                    audio_data = audio_filepath
                    for j in range(len(actors_per_fold)):
                        if j == fold:                            
                            if filename.split('_')[0] == actors_per_fold[j]:
                                x_test.append([visual_data, audio_data])
                                y_test.append(i)
                        else:
                            if filename.split('_')[0] == actors_per_fold[j]:
                                x_train.append([visual_data, audio_data])
                                y_train.append(i)

        print('ended reading folder %s' % directory)
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(class_labels)

def seed_torch(seed=2020):
    """
    Fix the seeds for the random generators of torch and other libraries
    :param seed: Seed to pass to the random seed generators
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def show_datast_count(dataset_name='RAVDESS_AU'):
    data_labels = get_data_labels(dataset_name)
    visual_data_path = get_visual_data_path(dataset_name)
    audio_data_path = get_audio_data_path(dataset_name)
    data, labels = get_visual_audio_data(visual_data_path, audio_data_path, class_labels=data_labels)
    
    return len(labels), data_labels

if __name__ == '__main__':
    counts, labels = [], []
    datasets = ['RAVDESS_AU', 'ENTERFACE_AU', 'SAVEE_AU', 'BAUM1S_AU', 'BAUM1A_AU']

    for i in range(len(datasets)):
        count, label = show_datast_count(datasets[i])
        counts.append(count)
        labels.append(label)

    for i in range(len(datasets)):
        print(datasets[i] + ': ' + str(counts[i]))
        print(labels[i])

    print('END')