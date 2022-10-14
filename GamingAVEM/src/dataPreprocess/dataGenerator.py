import math
import numpy as np

def imageBatchGenerator(imageDatas, imageLabels, batch_size):
    while True:
        batches = len(imageDatas) // batch_size
        if len(imageDatas) % batch_size > 0:
            batches += 1

        for b in range(batches):
            start = b * batch_size
            end = (b+1) * batch_size

            images = imageDatas[start:end]
            labels = imageLabels[start:end]

            yield images, labels

# 影像資料增強(高斯雜訊)
def visual_gaussian_noise_data_generator(data, labels, count=5):
    _data = []
    _labels = []
    for k in range(count):
        for i in range(0, len(data)):
            gaussian_outputs = []
            for j in range(0, len(data[i])):
                gaussian_outputs.append(visual_gaussian_noise(data[i][j]))
            _data.append(gaussian_outputs)
            _labels.append(labels[i])
    for i in range(0, len(data)):
        _data.append(data[i])
        _labels.append(labels[i])
    return np.array(_data), np.array(_labels)

# 聲音資料增強(高斯雜訊)
def audio_gaussian_noise_data_generator(data, labels, count=5):
    _data = []
    _labels = []
    for k in range(count):
        for i in range(0, len(data)):                
            _data.append(audio_gaussian_noise(data[i]))
            _labels.append(labels[i])
    for i in range(0, len(data)):
        _data.append(data[i])
        _labels.append(labels[i])
    return np.array(_data), np.array(_labels)

def visual_gaussian_noise(image, mean=0, sigma=3):
    img = np.copy(image)
    noise = np.random.normal(mean, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype('uint8')

#SNR in dB
#given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
def audio_gaussian_noise(signal, SNR=10) :
    sig = np.copy(signal)
    #RMS value of signal
    RMS_s = math.sqrt(np.mean(sig**2))
    #RMS values of noise
    RMS_n = math.sqrt(RMS_s**2/(pow(10, SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, sig.shape)
    return sig + noise