import os
import time
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# anger, happiness, sadness, surprise, disgust, fear
# 憤怒、快樂、悲傷、驚訝、厭惡、恐懼
ENTERFACE_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe')
#ENTERFACE_LABELS = ('an', 'ha', 'sa', 'su')
# anger, happiness, sadness, surprise, disgust, fear, neutral, boredom, bothered, concentrating, contempt, thinking, unsure
# 憤怒、快樂、悲傷、驚訝、厭惡、恐懼、中立、無聊、煩惱、集中、蔑視、思考、不確定
BAUM1S_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne', 'bor', 'bot', 'conc', 'cont', 'th', 'un')
BAUM1S_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne', 'bor', 'bot', 'conc', 'cont', 'th')
BAUM1S_LABELS = ('an', 'ha', 'sa', 'su', 'bor', 'bot')
#BAUM1S_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne')
# anger, happiness, sadness, disgust, fear, boredom, unsure, Interest
# 憤怒、快樂、悲傷、厭惡、恐懼、無聊、不確定、感興趣
BAUM1A_LABELS = ('an', 'ha', 'sa', 'di', 'fe', 'bor', 'un', 'int')
BAUM1A_LABELS = ('an', 'ha', 'sa', 'di', 'fe', 'bor', 'int')
# anger, happiness, sadness, surprise, disgust, fear, neutral
# 憤怒、快樂、悲傷、驚訝、厭惡、恐懼、中立
SAVEE_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne')
# anger, happiness, sadness, surprise, disgust, fear, neutral, calm
# 憤怒、快樂、悲傷、驚訝、厭惡、恐懼、中立、平靜
RAVDESS_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne', 'ca')
#RAVDESS_LABELS = ('an', 'ha', 'sa', 'su', 'ca')
#RAVDESS_LABELS = ('an', 'ha', 'sa', 'su', 'di', 'fe', 'ne')
# anger, happiness, sadness, surprise, boredom, calm(serious), contented, frustration
# 憤怒、快樂、悲傷、驚訝、無聊、平靜(嚴肅)、滿足、沮喪
GAVE_LABELS = ('an', 'ha', 'sa', 'su', 'bor', 'ca', 'conte', 'fr')
# anger, delightful, sad, surprised, confused, flow, excitation, frustration
# 憤怒、愉悅、悲傷、驚訝、困惑、專注、興奮、沮喪
GAVE_LABELS = ('an', 'de', 'sa', 'su', 'co', 'fl', 'ex', 'fr')

EMO_LABELS = ('an', 'di', 'fe', 'ha', 'sa', 'bo', 'ne')
CK_LABELS = ('an', 'di', 'fe', 'ha', 'sa', 'su')

DATA_PATH = 'D:/Adair/emotion_data/preprocessedData/'
AUDIO_DATA_PATH = DATA_PATH + 'audio/'
VISUAL_DATA_PATH = DATA_PATH + 'visual/'
AUDIO_VISUAL_DATA_PATH = DATA_PATH + 'audioVisual/'

VISUAL_SIZE_60x60_DATA_PATH = '60x60/'
VISUAL_SIZE_64x64_DATA_PATH = '64x64/'
VISUAL_SIZE_224x224_DATA_PATH = '224x224/'

VISUAL_NPY_DATA_PATH = 'npy/'
VISUAL_NPY_RGB_DATA_PATH = 'npy_rgb/'
VISUAL_NPY_HISTOGRAM_DATA_PATH = 'npy_histogram/'
VISUAL_NPY_FSIM_DATA_PATH = 'npy_fsim/'
VISUAL_NPY_SSIM_DATA_PATH = 'npy_ssim/'
VISUAL_NPY_FSM_DATA_PATH = 'npy_fsm/'
VISUAL_NPY_HISTOGRAM_FRAMES_COUNT_DATA_PATH = 'npy_histogram_frames_count/'
VISUAL_NPY_FSIM_FRAMES_COUNT_DATA_PATH = 'npy_fsim_frames_count/'
VISUAL_NPY_SSIM_FRAMES_COUNT_DATA_PATH = 'npy_ssim_frames_count/'
VISUAL_NPY_FSM_FRAMES_COUNT_DATA_PATH = 'npy_fsm_frames_count/'
VISUAL_NPY_HISTOGRAM_REVERSE_DATA_PATH = 'npy_histogram_reverse/'
VISUAL_NPY_FSIM_REVERSE_DATA_PATH = 'npy_fsim_reverse/'
VISUAL_NPY_SSIM_REVERSE_DATA_PATH = 'npy_ssim_reverse/'
VISUAL_NPY_FSM_REVERSE_DATA_PATH = 'npy_fsm_reverse/'
VISUAL_IMAGE_DATA_PATH = 'image/'
VISUAL_AU_DATA_PATH = 'au/'

ENTERFACE_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'enterface/'
ENTERFACE_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'enterface/'
ENTERFACE_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'enterface/'

BAUM1S_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'BAUM1s/'
BAUM1S_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'BAUM1s/'
BAUM1S_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'BAUM1s/'

BAUM1A_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'BAUM1a/'
BAUM1A_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'BAUM1a/'
BAUM1A_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'BAUM1a/'

SAVEE_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'savee/'
SAVEE_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'savee/'
SAVEE_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'savee/'

RAVDESS_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'RAVDESS/'
RAVDESS_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'RAVDESS/'
RAVDESS_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'RAVDESS/'

GAVE_DATA_PATH = AUDIO_VISUAL_DATA_PATH + 'GAVE/'
GAVE_AUDIO_DATA_PATH = AUDIO_DATA_PATH + 'GAVE/'
GAVE_VISUAL_DATA_PATH = VISUAL_DATA_PATH + 'GAVE/'

EMO_DATA_PATH = AUDIO_DATA_PATH + 'EMO_DB/'
CK_DATA_PATH = VISUAL_DATA_PATH + 'CK+/'

ENTERFACE_VISUAL_NPY_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_DATA_PATH
ENTERFACE_VISUAL_NPY_HISTOGRAM_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_HISTOGRAM_DATA_PATH
ENTERFACE_VISUAL_NPY_FSIM_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_FSIM_DATA_PATH
ENTERFACE_VISUAL_NPY_SSIM_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_SSIM_DATA_PATH
ENTERFACE_VISUAL_NPY_FSM_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_FSM_DATA_PATH
ENTERFACE_VISUAL_NPY_HISTOGRAM_REVERSE_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_HISTOGRAM_REVERSE_DATA_PATH
ENTERFACE_VISUAL_NPY_FSIM_REVERSE_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_FSIM_REVERSE_DATA_PATH
ENTERFACE_VISUAL_NPY_SSIM_REVERSE_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_SSIM_REVERSE_DATA_PATH
ENTERFACE_VISUAL_NPY_FSM_REVERSE_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_NPY_FSM_REVERSE_DATA_PATH
ENTERFACE_VISUAL_IMAGE_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_IMAGE_DATA_PATH
ENTERFACE_VISUAL_AU_DATA_PATH = ENTERFACE_VISUAL_DATA_PATH + VISUAL_AU_DATA_PATH

def plot_confusion_matrix(cm, class_labels=ENTERFACE_LABELS, normalize=False, title='Confusion matrix', index='', cmap=plt.cm.Blues, path=None):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] == thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path==None:
        plt.show()
    else:
        remove_file(path + title + '_' + index + '_confusion_matrix.png')
        plt.savefig(path + title + '_' + index + '_confusion_matrix.png')
    plt.close()

def plot_history_table(train_acc, val_acc, train_loss, val_loss, title='', index='', path=None):
    # summarize history for accuracy
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.title(title + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if path==None:
        plt.show()
    else:
        remove_file(path + title + '_' + index + '_acc.png')
        plt.savefig(path + title + '_' + index + '_acc.png')
        plt.close()

    # summarize history for loss 
    plt.plot(train_loss, label='train') 
    plt.plot(val_loss, label='val') 
    plt.title(title + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if path==None:
        plt.show()
    else:
        remove_file(path + title + '_' + index + '_loss.png')
        plt.savefig(path + title + '_' + index + '_loss.png')
    plt.close()

def plot_melspec(mel, sr, title='Mel-frequency spectrogram', path=None):
    # 繪製梅爾頻譜圖
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    if path==None:
        plt.show()
    else:
        remove_file(path)
        plt.savefig(path)
    plt.close()

def get_data_labels(Datasets):
    labels=[]
    if 'ENTERFACE' in Datasets:
        labels = ENTERFACE_LABELS
    elif 'BAUM1S' in Datasets:
        labels = BAUM1S_LABELS
        #labels = SAVEE_LABELS
        #labels = ENTERFACE_LABELS 
    elif 'BAUM1A' in Datasets:
        labels = BAUM1A_LABELS
    elif 'SAVEE' in Datasets:
        labels = SAVEE_LABELS
        #labels = ENTERFACE_LABELS
    elif 'RAVDESS' in Datasets:
        labels = RAVDESS_LABELS
        #labels = ENTERFACE_LABELS 
    elif 'GAVE' in Datasets:
        labels = GAVE_LABELS
    elif Datasets == 'EMO':
        labels = EMO_LABELS
    elif Datasets == 'CK+':
        labels = CK_LABELS
    return labels

def get_dataset_name(Datasets):
    dataset_name = ''
    if 'ENTERFACE' in Datasets:
        dataset_name = 'ENTERFACE'
    elif 'BAUM1S' in Datasets:
        dataset_name = 'BAUM1S'
    elif 'BAUM1A' in Datasets:
        dataset_name = 'BAUM1A'
    elif 'SAVEE' in Datasets:
        dataset_name = 'SAVEE'
    elif 'RAVDESS' in Datasets:
        dataset_name = 'RAVDESS'
    elif 'GAVE' in Datasets:
        dataset_name = 'GAVE'
    return dataset_name

def get_data_path(Datasets):
    path=''
    if 'ENTERFACE' in Datasets:
        path = ENTERFACE_DATA_PATH
    elif 'BAUM1S' in Datasets:
        path = BAUM1S_DATA_PATH
    elif 'BAUM1A' in Datasets:
        path = BAUM1A_DATA_PATH
    elif 'SAVEE' in Datasets:
        path = SAVEE_DATA_PATH
    elif 'RAVDESS' in Datasets:
        path = RAVDESS_DATA_PATH
    elif 'GAVE' in Datasets:
        path = GAVE_DATA_PATH
    return path

def get_audio_data_path(Datasets):
    path=''
    if 'ENTERFACE' in Datasets:
        path = ENTERFACE_AUDIO_DATA_PATH
    elif 'BAUM1S' in Datasets:
        path = BAUM1S_AUDIO_DATA_PATH
    elif 'BAUM1A' in Datasets:
        path = BAUM1A_AUDIO_DATA_PATH
    elif 'SAVEE' in Datasets:
        path = SAVEE_AUDIO_DATA_PATH
    elif 'RAVDESS' in Datasets:
        path = RAVDESS_AUDIO_DATA_PATH
    elif 'GAVE' in Datasets:
        path = GAVE_AUDIO_DATA_PATH
    elif Datasets == 'EMO':
        path = EMO_DATA_PATH
    return path

def get_visual_data_path(Datasets):
    path=''
    if 'ENTERFACE' in Datasets:
        path = ENTERFACE_VISUAL_DATA_PATH
    elif 'BAUM1S' in Datasets:
        path = BAUM1S_VISUAL_DATA_PATH
    elif 'BAUM1A' in Datasets:
        path = BAUM1A_VISUAL_DATA_PATH
    elif 'SAVEE' in Datasets:
        path = SAVEE_VISUAL_DATA_PATH
    elif 'RAVDESS' in Datasets:
        path = RAVDESS_VISUAL_DATA_PATH
    elif 'GAVE' in Datasets:
        path = GAVE_VISUAL_DATA_PATH
    else:
        return ''

    if '60x60' in Datasets:
        path = path + VISUAL_SIZE_60x60_DATA_PATH
    elif '64x64' in Datasets:
        path = path + VISUAL_SIZE_64x64_DATA_PATH

        
    if '_HISTOGRAM_REVERSE' in Datasets:
        path = path + VISUAL_NPY_HISTOGRAM_REVERSE_DATA_PATH
    elif '_FSIM_REVERSE' in Datasets:
        path = path + VISUAL_NPY_FSIM_REVERSE_DATA_PATH
    elif '_SSIM_REVERSE' in Datasets:
        path = path + VISUAL_NPY_SSIM_REVERSE_DATA_PATH
    elif '_FSM_REVERSE' in Datasets:
        path = path + VISUAL_NPY_FSM_REVERSE_DATA_PATH
    elif '_HISTOGRAM_FRAMES_COUNT' in Datasets:
        path = path + VISUAL_NPY_HISTOGRAM_FRAMES_COUNT_DATA_PATH
    elif '_FSIM_FRAMES_COUNT' in Datasets:
        path = path + VISUAL_NPY_FSIM_FRAMES_COUNT_DATA_PATH
    elif '_SSIM_FRAMES_COUNT' in Datasets:
        path = path + VISUAL_NPY_SSIM_FRAMES_COUNT_DATA_PATH
    elif '_FSM_FRAMES_COUNT' in Datasets:
        path = path + VISUAL_NPY_FSM_FRAMES_COUNT_DATA_PATH
    elif '_HISTOGRAM' in Datasets:
        path = path + VISUAL_NPY_HISTOGRAM_DATA_PATH
    elif '_FSIM' in Datasets:
        path = path + VISUAL_NPY_FSIM_DATA_PATH
    elif '_SSIM' in Datasets:
        path = path + VISUAL_NPY_SSIM_DATA_PATH
    elif '_FSM' in Datasets:
        path = path + VISUAL_NPY_FSM_DATA_PATH
    elif '_AU' in Datasets:
        path = path + VISUAL_AU_DATA_PATH
    elif '_RGB' in Datasets:
        path = path + VISUAL_NPY_RGB_DATA_PATH
    else:
        path = path + VISUAL_NPY_DATA_PATH

    if Datasets == 'CK+':
        path = CK_DATA_PATH
    return path

def get_labels_name(class_labels=ENTERFACE_LABELS):
    file_name = ''
    for i, directory in enumerate(class_labels):
        file_name = file_name + directory + '_'

    return file_name

def get_cnn_visual_data_path(Datasets):
    paths=[]
    if 'ENTERFACE' in Datasets:
        paths.append(get_visual_data_path(Datasets.replace('ENTERFACE', 'BAUM1S')))
        paths.append(get_visual_data_path(Datasets.replace('ENTERFACE', 'SAVEE')))
    elif 'BAUM1S' in Datasets:
        paths.append(get_visual_data_path(Datasets.replace('BAUM1S', 'ENTERFACE')))
        paths.append(get_visual_data_path(Datasets.replace('BAUM1S', 'SAVEE')))
    elif 'SAVEE' in Datasets:
        paths.append(get_visual_data_path(Datasets.replace('SAVEE', 'ENTERFACE')))
        paths.append(get_visual_data_path(Datasets.replace('SAVEE', 'BAUM1S')))
        
    return paths

def remove_file(path):
    if os.path.isfile(path) == True:
        os.remove(path)

def make_dirs(path):
    if os.path.isdir(path) != True:
        os.makedirs(path)

def write_result(path, content):
    f = open(path, 'w')
    f.write(str(content))
    f.close()
    print(content)

def read_result(path):
    content = ''
    if os.path.isfile(path) == True:
        f = open(path, 'r')
        content = f.read()
        f.close()
        print(content)
    return content

def get_timepoint():
    timepoint = time.time()
    print('Timepoint: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timepoint)))
    return timepoint, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timepoint))

def get_time_cost(timepoint_start, timepoint_end):
    time_cost = timepoint_end - timepoint_start
    ss = time_cost % 60
    mm = time_cost // 60 % 60
    hh = time_cost // 3600

    msg = str(hh) + 'hr ' + str(mm) + 'm ' + str(ss) + 's'
    print('Time Cost: ' + msg)
    return time_cost, msg