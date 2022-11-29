import sys
sys.path.append('./')
from src.common.utilities import *

def train_transfer_str(dataset_au_str, dataset_fsim_str, path):
    _str = train_transfer_au_str(dataset_au_str, path)
    _str = _str + train_transfer_au_fusion_str(dataset_au_str, path)
    _str = _str + train_transfer_fsim_str(dataset_fsim_str, path)
    _str = _str + train_transfer_fsim_fusion_str(dataset_fsim_str, path)
    _str = _str + train_transfer_remove_str(path)
    return _str

def train_transfer_au_str(dataset_au_str, path):
    _str = 'for %%i in (' + dataset_au_str + ') do (\n'
    _str = _str + '\tpython ./src/train/training.py --model Lstm --results_path '+ path + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/training.py --model Gru --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/training.py --model Wav2vec2 --results_path '+ path  + ' --dataset %%i --epochs 30 --batch_size 1 --learn_rate 1e-4 --accumulation_steps 8 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/training.py --model UniSpeech --results_path '+ path  + ' --dataset %%i --epochs 30 --batch_size 1 --learn_rate 1e-4 --accumulation_steps 8 --is_transfer True\n'
    _str = _str + ')\n\n'
    return _str

def train_transfer_au_fusion_str(dataset_au_str, path):
    _str = 'for %%i in (' + dataset_au_str + ') do (\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model Lstm --audio_model Wav2vec2 --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model Gru --audio_model Wav2vec2 --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model Lstm --audio_model UniSpeech --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model Gru --audio_model UniSpeech --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + ')\n\n'
    return _str

def train_transfer_fsim_str(dataset_fsim_str, path):    
    _str = 'for %%i in (' + dataset_fsim_str + ') do (\n'
    _str = _str + '\tpython ./src/train/training.py --model DFLN_BiLstm --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 16 --learn_rate 1e-4 --accumulation_steps 2 --is_transfer True\n'
    _str = _str + ')\n\n'
    return _str

def train_transfer_fsim_fusion_str(dataset_fsim_str, path):
    _str = 'for %%i in (' + dataset_fsim_str + ') do (\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model DFLN_BiLstm --audio_model Wav2vec2 --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + '\tpython ./src/train/trainingFusion.py --visual_model DFLN_BiLstm --audio_model UniSpeech --results_path '+ path  + ' --dataset %%i --epochs 100 --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer True\n'
    _str = _str + ')\n\n'
    return _str

def train_transfer_remove_str(path):
    _str = 'python ./src/train/training.py --model Remove --results_path ' + path  + '\n\n'
    return _str

def get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end):
    cmd_list = []
    for i in range(len(bat_dataset_au_strs)):
        bat_str = bat_start
        bat_str = bat_str + train_transfer_au_str(bat_dataset_au_strs[i], paths[i])
        cmd_list.append(bat_str + bat_end)

        bat_str = bat_start
        bat_str = bat_str + train_transfer_au_fusion_str(bat_dataset_au_strs[i], paths[i])
        cmd_list.append(bat_str + bat_end)

        bat_str = bat_start
        bat_str = bat_str + train_transfer_fsim_str(bat_dataset_fsim_strs[i], paths[i])
        cmd_list.append(bat_str + bat_end)

        bat_str = bat_start
        bat_str = bat_str + train_transfer_fsim_fusion_str(bat_dataset_fsim_strs[i], paths[i])
        cmd_list.append(bat_str + bat_end)

        bat_str = bat_start
        bat_str = bat_str + train_transfer_remove_str(paths[i])
        cmd_list.append(bat_str + bat_end)

    return cmd_list

def train_transfer_no(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []

    for dataset in datasets:
        bat_str = bat_start
        dataset_au_str = dataset + '_AU'
        dataset_fsim_str = dataset + '_FSIM_FRAMES_COUNT'
        path = results_path + '_no_transfer/'
        bat_str = bat_str + train_transfer_str(dataset_au_str, dataset_fsim_str, path)
        bat_str = bat_str.replace(' --is_transfer True', '')
        cmd_list.append(bat_str + bat_end)

    bat_str = bat_start
    dataset_au_str = target_dataset + '_AU'
    dataset_fsim_str = target_dataset + '_FSIM_FRAMES_COUNT'
    path = results_path + '_no_transfer/'
    bat_str = bat_str + train_transfer_str(dataset_au_str, dataset_fsim_str, path)
    bat_str = bat_str.replace(' --is_transfer True', '')
    cmd_list.append(bat_str + bat_end)

    return cmd_list

def train_transfer_1(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []
    bat_dataset_au_strs = []
    bat_dataset_fsim_strs = []
    paths = []

    for d1 in datasets:
        bat_dataset_au_strs.append(d1 + '_AU, ' + target_dataset + '_AU')
        bat_dataset_fsim_strs.append(d1 + '_FSIM_FRAMES_COUNT, ' + target_dataset + '_FSIM_FRAMES_COUNT')
        paths.append(results_path + '_' + dataset_to_labal(d1) + '_' + dataset_to_labal(target_dataset) + '/')

    cmd_list = get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end)

    return cmd_list

def train_transfer_2(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []
    bat_dataset_au_strs = []
    bat_dataset_fsim_strs = []
    paths = []

    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            bat_dataset_au_strs.append(d1 + '_AU, ' + d2 + '_AU, ' + target_dataset + '_AU')
            bat_dataset_fsim_strs.append(d1 + '_FSIM_FRAMES_COUNT, ' + d2 + '_FSIM_FRAMES_COUNT, ' + target_dataset + '_FSIM_FRAMES_COUNT')
            paths.append(results_path + '_' + dataset_to_labal(d1) + '_' + dataset_to_labal(d2) + '_' + dataset_to_labal(target_dataset) + '/')

    cmd_list = get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end)

    return cmd_list

def train_transfer_3(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []
    bat_dataset_au_strs = []
    bat_dataset_fsim_strs = []
    paths = []

    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            for d3 in datasets:
                if d3 == d1 or d3 == d2:
                    continue
                bat_dataset_au_strs.append(d1 + '_AU, ' + d2 + '_AU, ' + d3 + '_AU, ' + target_dataset + '_AU')
                bat_dataset_fsim_strs.append(d1 + '_FSIM_FRAMES_COUNT, ' + d2 + '_FSIM_FRAMES_COUNT, ' + d3 + '_FSIM_FRAMES_COUNT, ' + target_dataset + '_FSIM_FRAMES_COUNT')
                paths.append(results_path + '_' + dataset_to_labal(d1) + '_' + dataset_to_labal(d2) + '_' + dataset_to_labal(d3) + '_' + dataset_to_labal(target_dataset) + '/')

    cmd_list = get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end)

    return cmd_list

def train_transfer_4(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []
    bat_dataset_au_strs = []
    bat_dataset_fsim_strs = []
    paths = []

    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            for d3 in datasets:
                if d3 == d1 or d3 == d2:
                    continue
                for d4 in datasets:
                    if d4 == d1 or d4 == d2 or d4 == d3:
                        continue
                    bat_dataset_au_strs.append(d1 + '_AU, ' + d2 + '_AU, ' + d3 + '_AU, ' + d4 + '_AU, ' + target_dataset + '_AU')
                    bat_dataset_fsim_strs.append(d1 + '_FSIM_FRAMES_COUNT, ' + d2 + '_FSIM_FRAMES_COUNT, ' + d3 + '_FSIM_FRAMES_COUNT, ' + d4 + '_FSIM_FRAMES_COUNT, ' + target_dataset + '_FSIM_FRAMES_COUNT')
                    paths.append(results_path + '_' + dataset_to_labal(d1) + '_' + dataset_to_labal(d2) + '_' + dataset_to_labal(d3) + '_' + dataset_to_labal(d4) + '_' + dataset_to_labal(target_dataset) + '/')

    cmd_list = get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end)

    return cmd_list

def train_transfer_5(datasets, target_dataset, results_path, bat_start, bat_end):
    cmd_list = []
    bat_dataset_au_strs = []
    bat_dataset_fsim_strs = []
    paths = []

    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            for d3 in datasets:
                if d3 == d1 or d3 == d2:
                    continue
                for d4 in datasets:
                    if d4 == d1 or d4 == d2 or d4 == d3:
                        continue
                    for d5 in datasets:
                        if d5 == d1 or d5 == d2 or d5 == d3 or d5 == d4:
                            continue
                    bat_dataset_au_strs.append(d1 + '_AU, ' + d2 + '_AU, ' + d3 + '_AU, ' + d4 + '_AU, ' + d5 + '_AU, ' + target_dataset + '_AU')
                    bat_dataset_fsim_strs.append(d1 + '_FSIM_FRAMES_COUNT, ' + d2 + '_FSIM_FRAMES_COUNT, ' + d3 + '_FSIM_FRAMES_COUNT, ' + d4 + '_FSIM_FRAMES_COUNT, ' + d5 + '_FSIM_FRAMES_COUNT, ' + target_dataset + '_FSIM_FRAMES_COUNT')
                    paths.append(results_path + '_' + dataset_to_labal(d1) + '_' + dataset_to_labal(d2) + '_' + dataset_to_labal(d3) + '_' + dataset_to_labal(d4)  + '_' + dataset_to_labal(d5) + '_' + dataset_to_labal(target_dataset) + '/')

    cmd_list = get_cmd_list(bat_dataset_au_strs, bat_dataset_fsim_strs, paths, bat_start, bat_end)

    return cmd_list

def dataset_to_labal(x): 
    return{ 
        'BAUM1S':'bs', 
        'BAUM1A':'ba', 
        'ENTERFACE':'en', 
        'RAVDESS':'ra', 
        'SAVEE':'sa',        
        'GAVE':'ga'
    }.get(x, 'not')

if __name__ == '__main__': 

    datasets = ['BAUM1S','BAUM1A','ENTERFACE','RAVDESS','SAVEE']
    target_dataset = 'GAVE'
    results_path = './results/20221122/20221122'
    bat_path = './bat/'
    bat_start = 'call C:/Users/User/Documents/GamingAVEM/GamingAVEM/env/Scripts/activate.bat\ncd C:/Users/User/Documents/GamingAVEM/GamingAVEM\n\n'
    bat_end = '\n@pause'
    bat_end = ''
    bat_list = []

    cmd_list = train_transfer_no(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_no_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)

    cmd_list = train_transfer_1(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_1_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)

    cmd_list = train_transfer_2(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_2_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)

    cmd_list = train_transfer_3(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_3_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)

    cmd_list = train_transfer_4(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_4_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)

    cmd_list = train_transfer_5(datasets, target_dataset, results_path, bat_start, bat_end)
    for i in range(len(cmd_list)):
        bat_name = 'train_transfer_5_' + str(i+1) + '.bat'
        remove_file(bat_path + bat_name)
        write_result(bat_path + bat_name, cmd_list[i])
        bat_list.append(bat_name)
    
    str1 = ''
    for i in range(len(bat_list)):
        str1 = str1 + 'call C:/Users/User/Documents/GamingAVEM/GamingAVEM/bat/' + bat_list[i] + '\n'
    str1 = str1 + '\n@pause'
    remove_file(bat_path + 'train.bat')
    write_result(bat_path + 'train.bat', str1)