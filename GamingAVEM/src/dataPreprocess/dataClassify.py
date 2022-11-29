import os 
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.common.utilities import * 

FFMPEG = 'C:/ffmpeg-4.2.1-win-64/ffmpeg.exe'
 
def classify_EMO_DB(input_path = 'D:/Adair/emotion_data/originalData/EMO_DB/wav/'): 
    for file_name in os.listdir(input_path): 
 
        lab = EMO_DB_LABEL(file_name[5]) 
 
        input_file = os.path.abspath(input_path + file_name) 
 
        output_path = EMO_DATA_PATH + lab + '/' 
        make_dirs(output_path) 
        output_file = os.path.abspath(output_path + file_name) 
         
        print(input_file + '\n') 
        print(output_file + '\n') 
         
        copyfile(input_file, output_file) 
 
def EMO_DB_LABEL(x): 
    return{ 
        'W':'an', 
        'L':'bo', 
        'E':'di', 
        'A':'fe', 
        'F':'ha', 
        'T':'sa', 
        'N':'ne', 
    }.get(x, 'not') 
 
def classify_savee(input_path = 'D:/Adair/emotion_data/originalData/savee/AudioVisualClip/'): 
    dirs = ('DC', 'JE', 'JK', 'KL') 
 
    for dir in dirs: 
        _input_path = input_path + dir + '/' 
        for file_name in os.listdir(_input_path): 
 
            if file_name[0] != 's' : 
                lab = SAVEE_DB_LABEL(file_name[0]) 
            else: 
                lab = file_name[0:2] 
 
            input_file = os.path.abspath(_input_path + file_name) 
 
            output_path = SAVEE_DATA_PATH + lab + '/' 
            make_dirs(output_path) 
            output_file = os.path.abspath(output_path + dir + '_' + file_name) 
         
            print(input_file + '\n') 
            print(output_file + '\n')
            
            copyfile(input_file, output_file) 
 
def SAVEE_DB_LABEL(x): 
    return{ 
        'a':'an', 
        'd':'di', 
        'f':'fe', 
        'h':'ha', 
        'n':'ne', 
    }.get(x, 'not') 
 
def classify_BAUM1S(input_path = 'D:/Adair/emotion_data/originalData/BAUM1/'): 
    csv_path = input_path + 'Annotations_BAUM1s.csv' 
    f = open(csv_path) 
    csv = f.read() 
    #print(csv) 
    f.close() 
 
    rows = csv.split('\n') 
    for row in rows: 
        lab = '' 
        dir_name = '' 
        file_name = '' 
        datas = row.split(',') 
        
        
        lab = BAUM1S_DB_LABEL(datas[5])

        if lab == 'not':
            continue
 
        if len(datas[1]) == 1: 
            dir_name = 's00' + str(datas[1]) + '/' 
        elif len(datas[1]) == 2: 
            dir_name = 's0' + str(datas[1]) + '/' 
 
        file_name = str(datas[3]) + '.mp4' 
 
        input_file = os.path.abspath(input_path + 'BAUM1s_MP4 - All/' + dir_name + file_name) 
        output_path = BAUM1S_DATA_PATH + lab + '/' 
        make_dirs(output_path) 
        output_file = os.path.abspath(output_path + file_name) 
 
        print(input_file + '\n') 
        print(output_file + '\n') 
        
        copyfile(input_file, output_file)

def BAUM1S_DB_LABEL(x): 
    return{ 
        '1':'an',
        '2':'bor',
        '3':'bot',
        '4':'conc',
        '5':'cont',
        '6':'di',
        '7':'fe',
        '8':'ha',
        '9':'ne',
        '10':'sa',
        '11':'su',
        '12':'th',
        '13':'un',
    }.get(x, 'not') 
 
def classify_BAUM1A(input_path = 'D:/Adair/emotion_data/originalData/BAUM1/'): 
    csv_path = input_path + 'Annotations_BAUM1a.csv' 
    f = open(csv_path) 
    csv = f.read() 
    #print(csv) 
    f.close() 
 
    rows = csv.split('\n') 
    for row in rows: 
        lab = '' 
        dir_name = '' 
        file_name = '' 
        datas = row.split(',') 
        
        
        lab = BAUM1A_DB_LABEL(datas[5])

        if lab == 'not':
            continue
 
        if len(datas[1]) == 1: 
            dir_name = 's00' + str(datas[1]) + '/' 
        elif len(datas[1]) == 2: 
            dir_name = 's0' + str(datas[1]) + '/' 
 
        file_name = str(datas[3]) + '.mp4' 
 
        input_file = os.path.abspath(input_path + 'BAUM1a_MP4 - All/' + dir_name + file_name) 
        output_path = BAUM1A_DATA_PATH + lab + '/' 
        make_dirs(output_path) 
        output_file = os.path.abspath(output_path + file_name) 
 
        print(input_file + '\n') 
        print(output_file + '\n') 
        
        if os.path.isfile(input_file) == True:
            copyfile(input_file, output_file)

def BAUM1A_DB_LABEL(x): 
    return{ 
        '1':'an',
        '2':'bor',
        '3':'di',
        '4':'fe',
        '5':'ha',
        '6':'int',
        '7':'sa',
        '8':'su',
        '9':'un',
    }.get(x, 'not') 

def classify_enterface(input_path = 'D:/Adair/emotion_data/originalData/enterface/'): 
    for sub_name in os.listdir(input_path): 
        sub = sub_name.split(' ')[1] 
        for lab_name in os.listdir(input_path + sub_name + '/'):
            lab = ENTERFACE_DB_LABEL(lab_name)
            for sen_name in os.listdir(input_path + sub_name + '/' + lab_name + '/'): 
                if '.avi' in sen_name: 
                    input_file = input_path + sub_name + '/' + lab_name + '/' + sen_name 
                    output_path = ENTERFACE_DATA_PATH + lab + '/' 
                    make_dirs(output_path) 
                    output_file_name = file_name.split('_') 
                    output_file = os.path.abspath(output_path + sen_name) 
                             
                    print(input_file + '\n') 
                    print(output_file + '\n') 
                    
                    if os.path.isfile(output_file) != True:
                        copyfile(input_file, output_file) 
                    continue 
                if '.db' in sen_name: 
                    continue 
                for file_name in os.listdir(input_path + sub_name + '/' + lab_name + '/' + sen_name + '/'): 
                    if '.avi' in file_name: 
                        input_file = input_path + sub_name + '/' + lab_name + '/' + sen_name + '/' + file_name 
                        output_path = ENTERFACE_DATA_PATH + lab + '/' 
                        make_dirs(output_path) 
                        output_file_name = file_name.split('_') 
                        if len(output_file_name) < 4: 
                            output_file = os.path.abspath(output_path + 's' + sub + '_' + output_file_name[1] + '_' + output_file_name[2]) 
                        else: 
                            output_file = os.path.abspath(output_path + 's' + sub + '_' + output_file_name[2] + '_' + output_file_name[3]) 
                             
                        print(input_file + '\n') 
                        print(output_file + '\n') 
                        
                        copyfile(input_file, output_file)

def ENTERFACE_DB_LABEL(x): 
    return{ 
        'anger':'an',
        'disgust':'di',
        'fear':'fe',
        'happiness':'ha',
        'sadness':'sa',
        'surprise':'su',
    }.get(x, 'not') 

def classify_RAVDESS(input_path = 'D:/Adair/emotion_data/originalData/RAVDESS/Video_Speech_Actor_Data/'): 
    for dir_name in os.listdir(input_path): 
        for file_name in os.listdir(input_path + dir_name + '/'):
            if ('.mp4' in file_name) and (file_name.split('-')[0] == '01'):
                lab_name = file_name.split('-')[2]

                lab = RAVDESS_DB_LABEL(lab_name)
                output_path = RAVDESS_DATA_PATH + lab + '/'
                make_dirs(output_path)

                input_file = input_path + dir_name + '/' + file_name
                output_file = os.path.abspath(output_path + file_name) 

                print(input_file + '\n') 
                print(output_file + '\n')
                
                copyfile(input_file, output_file)

def RAVDESS_DB_LABEL(x): 
    return{ 
        '01':'ne',
        '02':'ca',
        '03':'ha',
        '04':'sa',
        '05':'an',
        '06':'fe',
        '07':'di',
        '08':'su',
    }.get(x, 'not')

def classify_GAVE(input_path = 'D:/Adair/emotion_data/originalData/GAVE'): 
    for file_name in os.listdir(input_path):
        if '.mp4' in file_name:
            lab_name = file_name.split('_')[2]

            lab = GAVE_DB_LABEL(lab_name)
            output_path = GAVE_DATA_PATH + lab + '/'
            make_dirs(output_path)

            input_file = input_path + '/' + file_name
            output_file = os.path.abspath(output_path + file_name) 

            print(input_file + '\n') 
            print(output_file + '\n')
            
            copyfile(input_file, output_file)

def GAVE_DB_LABEL1(x): 
    return{ 
        '01':'an',
        '02':'ha',
        '03':'sa',
        '04':'su',
        '05':'bor',
        '06':'ca',
        '07':'conte',
        '08':'fru',
    }.get(x, 'not') 

def GAVE_DB_LABEL(x): 
    return{ 
        '01':'an',
        '02':'de',
        '03':'sa',
        '04':'su',
        '05':'co',
        '06':'fl',
        '07':'ex',
        '08':'fr',
    }.get(x, 'not') 

def copyfile(input_path, output_path, start_time=0, end_time=10):
    output_path = output_path.replace('.avi', '.mp4').replace('.AVI', '.mp4')
    if os.path.isfile(output_path) != True:
        with VideoFileClip(input_path) as video:
            video_clip = video
            if video.duration > end_time:
                video_clip = video.subclip(start_time, end_time)
            video_clip.write_videofile(output_path, fps=30, codec="libx264")

if __name__ == '__main__':
    #classify_EMO_DB()
    
    #copyfile('D:/Adair/emotion_data/OriginalData/savee/AudioVisualClip/DC/a1.avi', 'D:/Adair/emotion_data/preprocessedData/audioVisual/DC_a1.avi')

    classify_savee()
    classify_BAUM1S()
    classify_BAUM1A()
    classify_enterface()
    classify_RAVDESS()
    classify_GAVE()

    #input_path = 'D:/Adair/emotion_data/OriginalData/RAVDESS/Audio_Speech_Actors_01-24/'
    #for dir_name in os.listdir(input_path): 
    #    for file_name in os.listdir(input_path + dir_name + '/'):
    #        if '.wav' in file_name:
    #            lab_name = file_name.split('-')[2]

    #            lab = RAVDESS_DB_LABEL(lab_name)
    #            output_path = RAVDESS_AUDIO_DATA_PATH + lab + '/'
    #            make_dirs(output_path)

    #            input_file = input_path + dir_name + '/' + file_name
    #            output_file = os.path.abspath(output_path + file_name) 

    #            print(input_file + '\n') 
    #            print(output_file + '\n')

    #            if os.path.isfile(output_file) != True:
    #                copyfile(input_file, output_file)