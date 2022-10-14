from src.common.utilities import *
from src.dataPreprocess.preprocess import *

datasets = ['ENTERFACE_AU', 'SAVEE_AU', 'RAVDESS_AU', 'BAUM1A_AU', 'BAUM1S_AU']
_datasets = []

def getName(dataset):
    if 'ENTERFACE' in dataset:
        return 'en'
    if 'SAVEE' in dataset:
        return 'sa'
    if 'RAVDESS' in dataset:
        return 'ra'
    if 'BAUM1A' in dataset:
        return 'ba'
    if 'BAUM1S' in dataset:
        return 'bs'



if __name__ == '__main__': 
    csv = 'DATASET,TRAIN,VAL,TEST,TOTAL\n'
    for d1 in datasets:
        data_labels = get_data_labels(d1)
        visual_data_path = get_visual_data_path(d1)
        audio_data_path = get_audio_data_path(d1)
        x_train, x_val, x_test, y_train, y_val, y_test, num_classes = extract_visual_audio_data(visual_data_path=visual_data_path, 
                                                                                                audio_data_path=audio_data_path, 
                                                                                                class_labels=data_labels, random=2020)
        
        csv += '{},{:d},{:d},{:d},{:d}\n'.format(d1, len(y_train), len(y_val), len(y_test), len(y_train)+len(y_val)+len(y_test))
    write_result('./dataset.csv', csv)

#    for d1 in datasets:
#        _datasets.append(d1)
#        for d2 in datasets:
#            if d2 == d1:
#                continue
#            _datasets.append(d2)
#            for d3 in datasets:
#                if d3 == d1 or d3 == d2:
#                    continue
#                _datasets.append(d3)
#                for d4 in datasets:
#                    if d4 == d1 or d4 == d2 or d4 == d3:
#                        continue
#                    _datasets.append(d4)
#                    #print(_datasets)
#                    dirName = getName(_datasets[0]) + '_' + getName(_datasets[1]) + '_' + getName(_datasets[2]) + '_' + getName(_datasets[3])
#                    print('for %%i in ('+_datasets[0]+', '+_datasets[1]+', '+_datasets[2]+', '+_datasets[3]+') do (')
#                    print('    python ./src/train/training.py --model Lstm --results_path ./results/20221005_' + dirName + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True')
#                    print('    python ./src/train/training.py --model Wav2vec2 --results_path ./results/20221005_' + dirName + ' --dataset %%i --epochs 30 --batch_size 1 --learn_rate 1e-4 --accumulation_steps 8 --is_transfer True')
#                    print('    python ./src/train/trainingFusion.py --visual_model Lstm --audio_model Wav2vec2 --results_path ./results/20221005_' + dirName + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True')
#                    print(')')
#                    _datasets.pop(-1)
#                _datasets.pop(-1)
#            _datasets.pop(-1)
#        _datasets.pop(-1)


#[ENTERFACE_AU, SAVEE_AU, RAVDESS_AU, BAUM1A_AU]
#[ENTERFACE_AU, SAVEE_AU, BAUM1A_AU, RAVDESS_AU]
#[ENTERFACE_AU, RAVDESS_AU, SAVEE_AU, BAUM1A_AU]
#[ENTERFACE_AU, RAVDESS_AU, BAUM1A_AU, SAVEE_AU]
#[ENTERFACE_AU, BAUM1A_AU, SAVEE_AU, RAVDESS_AU]
#[ENTERFACE_AU, BAUM1A_AU, RAVDESS_AU, SAVEE_AU]
#[SAVEE_AU, ENTERFACE_AU, RAVDESS_AU, BAUM1A_AU]
#[SAVEE_AU, ENTERFACE_AU, BAUM1A_AU, RAVDESS_AU]
#[SAVEE_AU, RAVDESS_AU, ENTERFACE_AU, BAUM1A_AU]
#[SAVEE_AU, RAVDESS_AU, BAUM1A_AU, ENTERFACE_AU]
#[SAVEE_AU, BAUM1A_AU, ENTERFACE_AU, RAVDESS_AU]
#[SAVEE_AU, BAUM1A_AU, RAVDESS_AU, ENTERFACE_AU]
#[RAVDESS_AU, ENTERFACE_AU, SAVEE_AU, BAUM1A_AU]
#[RAVDESS_AU, ENTERFACE_AU, BAUM1A_AU, SAVEE_AU]
#[RAVDESS_AU, SAVEE_AU, ENTERFACE_AU, BAUM1A_AU]
#[RAVDESS_AU, SAVEE_AU, BAUM1A_AU, ENTERFACE_AU]
#[RAVDESS_AU, BAUM1A_AU, ENTERFACE_AU, SAVEE_AU]
#[RAVDESS_AU, BAUM1A_AU, SAVEE_AU, ENTERFACE_AU]
#[BAUM1A_AU, ENTERFACE_AU, SAVEE_AU, RAVDESS_AU]
#[BAUM1A_AU, ENTERFACE_AU, RAVDESS_AU, SAVEE_AU]
#[BAUM1A_AU, SAVEE_AU, ENTERFACE_AU, RAVDESS_AU]
#[BAUM1A_AU, SAVEE_AU, RAVDESS_AU, ENTERFACE_AU]
#[BAUM1A_AU, RAVDESS_AU, ENTERFACE_AU, SAVEE_AU]
#[BAUM1A_AU, RAVDESS_AU, SAVEE_AU, ENTERFACE_AU]