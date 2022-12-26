import os
import sys

sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import wave
from src.common.utilities import *



# 建立繪製聲波的函式
def visualize(path, img_path):
    raw = wave.open(path)          # 開啟聲音
    signal = raw.readframes(-1)    # 讀取全部聲音採樣
    signal = np.frombuffer(signal, dtype ="int16")  # 將聲音採樣轉換成 int16 的格式所組成的 np 陣列
    f_rate = raw.getframerate()    # 取得 framerate
    time = np.linspace(0, len(signal)/f_rate, num = len(signal))  # 根據聲音採樣產生成對應的時間

    fig, ax = plt.subplots()  # 建立單一圖表
    ax.plot(time, signal)          # 畫線，橫軸時間，縱軸陣列值

    plt.title("Sound Wave")        # 圖表標題
    plt.xlabel("Time")             # 橫軸標題
    remove_file(img_path)
    plt.savefig(img_path)
    plt.close()

    print(img_path)


path = get_audio_data_path('GAVE')
for lab in os.listdir(path):
    _lab = lab + '/'
    for file_name in os.listdir(path + _lab):
        name = file_name.split('.')[0]
        _path = path + _lab + file_name
        img_path = './results/wav/' + _lab + name + '.png'
        make_dirs('./results/wav/' + _lab)
        visualize(_path, img_path)