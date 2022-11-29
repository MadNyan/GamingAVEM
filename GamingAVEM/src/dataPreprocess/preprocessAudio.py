import os
import csv
import sys
import librosa
import numpy as np
from src.common.utilities import *

FFMPEG = 'C:/ffmpeg-4.2.1-win-64/ffmpeg.exe'

def capture_audio_files(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_AUDIO_DATA_PATH, 
                       class_labels=ENTERFACE_LABELS, sampling=None):
    for lab in os.listdir(input_path):
        _lab = lab + '/'
        for file_name in os.listdir(input_path + _lab):

            if 'enterface' in input_path and 's6' in file_name:
                continue;

            name = file_name.split('.')[0]
            _input_path = input_path + _lab + file_name
            _output_path = output_path + _lab

            make_dirs(_output_path)

            output_path_file = _output_path + name + '.wav'

            if os.path.isfile(output_path_file) == True:
                print(name + ' is captured')
                continue

            print('started capturing ' + name)

            capture_audio_file(_input_path, output_path_file, sampling)

            print('ended capturing ' + name)


def capture_audio_file(input_path, output_path, sampling=None):
    try:
        if sampling==None:
            cmd = FFMPEG + ' -i "' + input_path + '" -ac 1 "' + output_path + '"'
        else:
            cmd = FFMPEG + ' -i "' + input_path + '" -ar ' + str(sampling) + ' -ac 1 "' + output_path + '"'
        print("ffmpeg cmd: ", cmd)
        os.system(cmd)
    except:
        print('capture_audio ' + output_path + ' Error')

if __name__ == '__main__':
    ORDERS = ['ENTERFACE','BAUM1S','BAUM1A','SAVEE','RAVDESS','GAVE']
    for order in ORDERS:
        input_path = get_data_path(order)
        output_path = get_audio_data_path(order)
        labels = get_data_labels(order)
        capture_audio_files(input_path=input_path, output_path=output_path, class_labels=labels, sampling=16000)