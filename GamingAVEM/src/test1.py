import os, sys
sys.path.append('./')

import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.common.utilities import *

def to_time(time_str):
    times = time_str.split(':')
    return times[0] + ':' + times[1] + ':' + times[2] + '.' + times[3]

if __name__ == '__main__':
    csv_path = 'D:/Adair/GAMING DATA/GAMING_EMOTION.csv'
    input_path = 'D:/Adair/GAMING DATA/output/'
    #output_path = 'D:/Adair/GAMING DATA/GAVE/'
    output_path = 'D:/Adair/emotion_data/originalData/GAVE/'
    #output_game_path = 'D:/Adair/GAMING DATA/GAVE/inGame/'
    output_game_path = 'D:/Adair/emotion_data/originalData/GAVE/inGame/'

    if (os.path.exists(csv_path)):
        make_dirs(output_path)
        make_dirs(output_game_path)
        csv = read_result(csv_path)
        for row in csv.split('\n'):
            data = row.split(',')

            if data[3] == 'output':
                continue

            output_file_path = output_path + data[3] + '.mp4'
            output_game_file_path = output_game_path + data[3] + '_game.mp4'
            start_time = to_time(data[1])
            end_time = to_time(data[2])
            print('loading... ' + input_path + data[0])

            if os.path.isfile(output_file_path) != True:
                with VideoFileClip(input_path + data[0] + '.mkv') as video:
                    video_clip = video
                    video_clip = video.subclip(start_time, end_time)
                    video_clip.write_videofile(output_file_path, fps=30, codec='libx264')
                    print(output_file_path)

            if os.path.isfile(output_game_file_path) != True:
                with VideoFileClip(input_path + data[0] + '.mp4') as video:
                    video_clip = video
                    video_clip = video.subclip(start_time, end_time)
                    video_clip.write_videofile(output_game_file_path, fps=30, codec='libx264')
                    print(output_game_file_path)
        