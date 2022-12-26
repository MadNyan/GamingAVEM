import sys

from numpy import True_
sys.path.append('./')
from src.common.utilities import *

def create_train_cmd_str(model, path, dataset, epochs, batch_size, accumulation_steps, is_transfer=False, load_path='', save_path='', img_path='', nickname=''):
    _str = 'python ./src/train/training.py --model ' + model + ' '
    _str = _str + '--results_path ' + path + ' '
    _str = _str + '--dataset ' + dataset + ' '
    _str = _str + '--epochs ' + str(epochs) + ' --batch_size ' + str(batch_size) + ' --learn_rate 1e-4 --accumulation_steps ' + str(accumulation_steps) + ' '
    _str = _str + '--is_transfer ' + str(is_transfer) + ' '
    _str = _str + '--load_path ' + load_path + ' --save_path ' + save_path + ' --img_path ' + img_path + ' --nickname ' + nickname + '\n'
    return _str

def create_train_fusion_cmd_str(visual_model, audio_model, visual_load_path, audio_load_path, path, dataset, epochs, is_transfer=False, load_path='', save_path='', img_path='', nickname=''):
    _str = 'python ./src/train/trainingFusion.py --visual_model ' + visual_model + ' --audio_model ' + audio_model + ' '
    _str = _str + '--visual_load_path ' + visual_load_path + ' --audio_load_path ' + audio_load_path + ' '
    _str = _str + '--results_path '+ path + ' '
    _str = _str + '--dataset ' + dataset + ' '
    _str = _str + '--epochs ' + str(epochs) + ' --batch_size 64 --learn_rate 1e-4 --accumulation_steps 1 --is_transfer ' + str(is_transfer) + ' '
    _str = _str + '--load_path ' + load_path + ' --save_path ' + save_path + ' --img_path ' + img_path + ' --nickname ' + nickname + '\n'    
    return _str

def create_remove_cmd_str(path):
    _str = 'python ./src/train/training.py --model Remove --results_path ' + path + '\n'
    return _str

def create_train_bat_list(bat_list, model, path, datasets, dataset_type, epochs, batch_size, accumulation_steps):
    bat_start = 'call C:/Users/User/Documents/GamingAVEM/GamingAVEM/env/Scripts/activate.bat\ncd C:/Users/User/Documents/GamingAVEM/GamingAVEM\n\n'

    _str = ''
    for d1 in datasets:
        dataset_nickname = get_dataset_nickname(d1)
        _str = _str + create_train_cmd_str(model, path, d1, epochs, batch_size, accumulation_steps, is_transfer=False, 
                                    load_path='', 
                                    save_path=path + model + '_' + dataset_nickname + '.pth', 
                                    img_path=path + dataset_nickname + '/',
                                    nickname=model + '_' + dataset_nickname)
    bat_path = './bat/' + dataset_type + '/' + '1' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    _str = ''
    for d1 in datasets:
        load_dataset_nickname = get_dataset_nickname(d1)
        for d2 in datasets:
            dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2)
            if d2 == d1:
                continue
            _str = _str + create_train_cmd_str(model, path, d2, epochs, batch_size, accumulation_steps, is_transfer=True, 
                                        load_path=path + model + '_' + load_dataset_nickname + '.pth', 
                                        save_path=path + model + '_' + dataset_nickname + '.pth', 
                                        img_path=path + dataset_nickname + '/',
                                        nickname=model + '_' + dataset_nickname)
    bat_path = './bat/' + dataset_type + '/' + '2' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)
    
    _str = ''
    for d1 in datasets:
        for d2 in datasets:
            load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2)
            if d2 == d1:
                continue
            for d3 in datasets:
                dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3)
                if d3 == d1 or d3 == d2:
                    continue
                _str = _str + create_train_cmd_str(model, path, d3, epochs, batch_size, accumulation_steps, is_transfer=True, 
                                            load_path=path + model + '_' + load_dataset_nickname + '.pth', 
                                            save_path=path + model + '_' + dataset_nickname + '.pth', 
                                            img_path=path + dataset_nickname + '/',
                                            nickname=model + '_' + dataset_nickname)
    bat_path = './bat/' + dataset_type + '/' + '3' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    _str = ''
    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            for d3 in datasets:                
                load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3)
                if d3 == d1 or d3 == d2:
                    continue
                for d4 in datasets:
                    dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4)
                    if d4 == d1 or d4 == d2 or d4 == d3:
                        continue
                    _str = _str + create_train_cmd_str(model, path, d4, epochs, batch_size, accumulation_steps, is_transfer=True, 
                                                load_path=path + model + '_' + load_dataset_nickname + '.pth', 
                                                save_path=path + model + '_' + dataset_nickname + '.pth', 
                                                img_path=path + dataset_nickname + '/',
                                                nickname=model + '_' + dataset_nickname)
    bat_path = './bat/' + dataset_type + '/' + '4' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    #_str = ''
    #for d1 in datasets:
    #    for d2 in datasets:
    #        if d2 == d1:
    #            continue
    #        for d3 in datasets:
    #            if d3 == d1 or d3 == d2:
    #                continue
    #            for d4 in datasets:
    #                load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4)
    #                if d4 == d1 or d4 == d2 or d4 == d3:
    #                    continue
    #                for d5 in datasets:
    #                    dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4) + '_' + get_dataset_nickname(d5)
    #                    if d5 == d1 or d5 == d2 or d5 == d3 or d5 == d4:
    #                        continue
    #                    _str = _str + create_train_cmd_str(model, path, d5, epochs, batch_size, accumulation_steps, is_transfer=True, 
    #                                                load_path=path + model + '_' + load_dataset_nickname + '.pth', 
    #                                                save_path=path + model + '_' + dataset_nickname + '.pth', 
    #                                                img_path=path + dataset_nickname + '/',
    #                                                nickname=model + '_' + dataset_nickname)
    #bat_path = './bat/' + dataset_type + '/' + '5' + '.bat'
    #make_dirs('./bat/' + dataset_type + '/')
    #bat_list.append(bat_path)
    #remove_file(bat_path)
    #write_result(bat_path, bat_start + _str)

    return bat_list

def create_train_fusion_bat_list(bat_list, visual_model, audio_model, visual_load_path, audio_load_path, path, datasets, dataset_type, epochs):
    bat_start = 'call C:/Users/User/Documents/GamingAVEM/GamingAVEM/env/Scripts/activate.bat\ncd C:/Users/User/Documents/GamingAVEM/GamingAVEM\n\n'

    _str = ''
    for d1 in datasets:
        dataset_nickname = get_dataset_nickname(d1)
        nickname = visual_model + '_' + audio_model + '_' + dataset_nickname
        _str = _str + create_train_fusion_cmd_str(visual_model, audio_model, 
                                           visual_load_path=visual_load_path + '/' + visual_model + '_' + dataset_nickname + '.pth', 
                                           audio_load_path=audio_load_path + '/' + audio_model + '_' + dataset_nickname + '.pth', 
                                           path=path, dataset=d1, epochs=epochs, is_transfer=False, 
                                           load_path='', 
                                           save_path=path + nickname + '.pth', 
                                           img_path=path + dataset_nickname + '/',
                                           nickname=nickname)
    bat_path = './bat/' + dataset_type + '/' + 'fusion_1' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    _str = ''
    for d1 in datasets:
        load_dataset_nickname = get_dataset_nickname(d1)
        for d2 in datasets:
            dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2)
            if d2 == d1:
                continue
            nickname = visual_model + '_' + audio_model + '_' + dataset_nickname
            _str = _str + create_train_fusion_cmd_str(visual_model, audio_model, 
                                               visual_load_path=visual_load_path + '/' + visual_model + '_' + dataset_nickname + '.pth', 
                                               audio_load_path=audio_load_path + '/' + audio_model + '_' + dataset_nickname + '.pth', 
                                               path=path, dataset=d2, epochs=epochs, is_transfer=False, 
                                               load_path='', 
                                               save_path=path + nickname + '.pth', 
                                               img_path=path + dataset_nickname + '/',
                                               nickname=nickname)
    bat_path = './bat/' + dataset_type + '/' + 'fusion_2' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    _str = ''  
    for d1 in datasets:
        for d2 in datasets:
            load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2)
            if d2 == d1:
                continue
            for d3 in datasets:
                dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3)
                if d3 == d1 or d3 == d2:
                    continue
                nickname = visual_model + '_' + audio_model + '_' + dataset_nickname
                _str = _str + create_train_fusion_cmd_str(visual_model, audio_model, 
                                                   visual_load_path=visual_load_path + '/' + visual_model + '_' + dataset_nickname + '.pth', 
                                                   audio_load_path=audio_load_path + '/' + audio_model + '_' + dataset_nickname + '.pth', 
                                                   path=path, dataset=d3, epochs=epochs, is_transfer=False, 
                                                   load_path='', 
                                                   save_path=path + nickname + '.pth', 
                                                   img_path=path + dataset_nickname + '/',
                                                   nickname=nickname)
    bat_path = './bat/' + dataset_type + '/' + 'fusion_3' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)
    
    _str = ''
    for d1 in datasets:
        for d2 in datasets:
            if d2 == d1:
                continue
            for d3 in datasets:                
                load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3)
                if d3 == d1 or d3 == d2:
                    continue
                for d4 in datasets:
                    dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4)
                    if d4 == d1 or d4 == d2 or d4 == d3:
                        continue
                    nickname = visual_model + '_' + audio_model + '_' + dataset_nickname
                    _str = _str + create_train_fusion_cmd_str(visual_model, audio_model, 
                                                       visual_load_path=visual_load_path + '/' + visual_model + '_' + dataset_nickname + '.pth', 
                                                       audio_load_path=audio_load_path + '/' + audio_model + '_' + dataset_nickname + '.pth', 
                                                       path=path, dataset=d4, epochs=epochs, is_transfer=False, 
                                                       load_path='', 
                                                       save_path=path + nickname + '.pth', 
                                                       img_path=path + dataset_nickname + '/',
                                                       nickname=nickname)
    bat_path = './bat/' + dataset_type + '/' + 'fusion_4' + '.bat'
    make_dirs('./bat/' + dataset_type + '/')
    bat_list.append(bat_path)
    remove_file(bat_path)
    write_result(bat_path, bat_start + _str)

    #_str = ''
    #for d1 in datasets:
    #    for d2 in datasets:
    #        if d2 == d1:
    #            continue
    #        for d3 in datasets:
    #            if d3 == d1 or d3 == d2:
    #                continue
    #            for d4 in datasets:
    #                load_dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4)
    #                if d4 == d1 or d4 == d2 or d4 == d3:
    #                    continue
    #                for d5 in datasets:
    #                    dataset_nickname = get_dataset_nickname(d1) + '_' + get_dataset_nickname(d2) + '_' + get_dataset_nickname(d3) + '_' + get_dataset_nickname(d4) + '_' + get_dataset_nickname(d5)
    #                    if d5 == d1 or d5 == d2 or d5 == d3 or d5 == d4:
    #                        continue
    #                    nickname = visual_model + '_' + audio_model + '_' + dataset_nickname
    #                    _str = _str + create_train_fusion_cmd_str(visual_model, audio_model, 
    #                                                       visual_load_path=visual_load_path + '/' + visual_model + '_' + dataset_nickname + '.pth', 
    #                                                       audio_load_path=audio_load_path + '/' + audio_model + '_' + dataset_nickname + '.pth', 
    #                                                       path=path, dataset=d5, epochs=epochs, is_transfer=False, 
    #                                                       load_path='', 
    #                                                       save_path=path + nickname + '.pth', 
    #                                                       img_path=path + dataset_nickname + '/',
    #                                                       nickname=nickname)
    #bat_path = './bat/' + dataset_type + '/' + 'fusion_5' + '.bat'
    #make_dirs('./bat/' + dataset_type + '/')
    #bat_list.append(bat_path)
    #remove_file(bat_path)
    #write_result(bat_path, bat_start + _str)

    return bat_list

if __name__ == '__main__': 
    
    target_dataset = 'GAVE'
    datasets = ['BAUM1S','BAUM1A','ENTERFACE','RAVDESS','SAVEE']
    au_datasets = ['_AU','_AU_HISTOGRAM_FRAMES_COUNT','_AU_FSIM_FRAMES_COUNT','_AU_SSIM_FRAMES_COUNT','_AU_FSM_FRAMES_COUNT','_AU_HISTOGRAM','_AU_FSIM','_AU_SSIM','_AU_FSM']
    npy_datasets = ['_HISTOGRAM_FRAMES_COUNT','_FSIM_FRAMES_COUNT','_SSIM_FRAMES_COUNT','_FSM_FRAMES_COUNT','_HISTOGRAM','_FSIM','_SSIM','_FSM']

    datasets = ['BAUM1S','ENTERFACE','RAVDESS','SAVEE']
    au_datasets = ['_AU','_AU_HISTOGRAM_FRAMES_COUNT','_AU_FSIM_FRAMES_COUNT','_AU_SSIM_FRAMES_COUNT','_AU_FSM_FRAMES_COUNT']
    npy_datasets = ['_HISTOGRAM_FRAMES_COUNT','_FSIM_FRAMES_COUNT','_SSIM_FRAMES_COUNT','_FSM_FRAMES_COUNT']
    
    au_models = ['Lstm','Gru']
    npy_models = ['DFLN_BiLstm']
    audio_models = ['Wav2vec2','UniSpeech']
    
    results_path = './results/test/'

    bat_path = './bat/'
    bat_start = 'call C:/Users/User/Documents/GamingAVEM/GamingAVEM/env/Scripts/activate.bat\ncd C:/Users/User/Documents/GamingAVEM/GamingAVEM\n\n'
    bat_list = []

    # Lstm, Gru
    _datasets = []
    for d1 in au_datasets:
        for d2 in datasets:
            _datasets.append(d2 + d1)
        for m in au_models:
            path = results_path + m + d1 + '/'
            bat_list = create_train_bat_list(bat_list, m, path, _datasets, d1, epochs=100, batch_size=64, accumulation_steps=1)

    # DFLN_BiLstm
    _datasets = []
    for d1 in npy_datasets:
        for d2 in datasets:
            _datasets.append(d2 + d1)
        for m in npy_models:
            path = results_path + m + d1 + '/'
            bat_list = create_train_bat_list(bat_list, m, path, _datasets, d1, epochs=100, batch_size=16, accumulation_steps=2)

    # Wav2vec2, UniSpeech
    _datasets = []
    for m in audio_models:
        path = results_path + m + '/'
        bat_list = create_train_bat_list(bat_list, m, path, datasets, m, epochs=30, batch_size=1, accumulation_steps=8)

    # Fusion
    _datasets = []
    for d1 in au_datasets:
        for d2 in datasets:
            _datasets.append(d2 + d1)
            for v_m in au_models:
                for a_m in audio_models:
                    path = results_path + v_m + '_' + a_m + d1 + '/'
                    v_load_path = results_path + m + d1 + '/'
                    a_load_path = results_path + m + '/'
                    bat_list = create_train_fusion_bat_list(bat_list, v_m, a_m, v_load_path, a_load_path, path, _datasets, d1, epochs=100)
    _datasets = []
    for d1 in npy_datasets:
        for d2 in datasets:
            _datasets.append(d2 + d1)
            for v_m in npy_models:
                for a_m in audio_models:
                    path = results_path + v_m + '_' + a_m + d1 + '/'
                    v_load_path = results_path + m + d1 + '/'
                    a_load_path = results_path + m + '/'
                    bat_list = create_train_fusion_bat_list(bat_list, v_m, a_m, v_load_path, a_load_path, path, _datasets, d1, epochs=100)
                
    str1 = 'cd C:/Users/User/Documents/GamingAVEM/GamingAVEM\n'
    for i in range(len(bat_list)):
        str1 = str1 + 'call ' + bat_list[i] + '\n'
    str1 = str1 + '\n@pause'
    remove_file(bat_path + 'train.bat')
    write_result(bat_path + 'train.bat', str1)