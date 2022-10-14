import os
import sys
import argparse

sys.path.append('./')

from src.common.utilities import *
from src.dataPreprocess.preprocess import *
from src.train.trainerRnn import trainerLstm, trainerGru
from src.train.trainerWav2vec2 import trainerWav2vec2
from src.train.trainerDFLN_BiLstm import trainerDFLN_BiLstm
from src.train.trainerFAN import trainerFAN
from src.train.trainerUniSpeech import trainerUniSpeech
from src.train.trainerFusion import trainerFusion

if __name__ == '__main__': 
    
    model = 'FAN'
    results_path = './results/test'
    dataset_name = 'ENTERFACE_FSIM_FRAMES_COUNT'
    epochs = 1
    batch_size = 64
    learn_rate = 1e-4
    gradient_accumulation_steps = 1
    is_transfer = True
    is_fold = False
    fold = 0
    fold_name = ''

    #Read input parameters
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-model', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-results', '--results_path', type=str, required=True,
                        help='Path with the results')
    parser.add_argument('-dataset', '--dataset', type=str,
                        help='dataset',
                        default='RAVDESS_AU')
    parser.add_argument('-epochs', '--epochs', type=int,
                        help='Epochs',
                        default=30)
    parser.add_argument('-batch_size', '--batch_size', type=int, 
                        help='Batch size',
                        default=16)
    parser.add_argument('-learn_rate', '--learn_rate', type=float, 
                        help='Learn rate',
                        default=1e-3)
    parser.add_argument('-accumulation_steps', '--accumulation_steps', type=int, 
                        help='Gradient accumulation steps',
                        default=2)
    parser.add_argument('-transfer', '--is_transfer', type=bool, 
                        help='Transfer',
                        default=False)
    parser.add_argument('-is_fold', '--is_fold', type=bool, 
                        help='Is fold',
                        default=False)
    parser.add_argument('-fold', '--fold', type=int, 
                        help='Fold',
                        default=0)
    args = parser.parse_args()

    model = args.model
    results_path = args.results_path
    dataset_name = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    gradient_accumulation_steps = args.accumulation_steps
    is_transfer = args.is_transfer
    is_fold = args.is_fold
    fold = args.fold

    if model == 'Remove':
        for filename in os.listdir(results_path):
            if '.npy' in filename or '.pth' in filename:
                filepath = results_path + '/' + filename
                remove_file(filepath)
                print('Remove %s' % filepath)
    else:
        data_labels = get_data_labels(dataset_name)
        visual_data_path = get_visual_data_path(dataset_name)
        audio_data_path = get_audio_data_path(dataset_name)
        
        if is_fold:
            x_train, x_test, y_train, y_test, num_classes = get_visual_audio_data_fold(visual_data_path=visual_data_path, 
                                                                                       audio_data_path=audio_data_path, 
                                                                                       class_labels=data_labels, fold=fold)
            x_visual_train, x_audio_train = split_visual_audio_data(x_train)
            x_visual_test, x_audio_test = split_visual_audio_data(x_test)
            x_visual_val, x_audio_val = x_visual_test, x_audio_test
            y_val = y_test
            fold_name = str(fold)
        else:
            x_train, x_val, x_test, y_train, y_val, y_test, num_classes = extract_visual_audio_data(visual_data_path=visual_data_path, 
                                                                                                    audio_data_path=audio_data_path, 
                                                                                                    class_labels=data_labels, random=2020)
            x_visual_train, x_audio_train = split_visual_audio_data(x_train)
            x_visual_val, x_audio_val = split_visual_audio_data(x_val)
            x_visual_test, x_audio_test = split_visual_audio_data(x_test)

        if model == 'Lstm':
            trainer = trainerLstm(results_path, dataset_name, x_visual_train, x_visual_val, x_visual_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name)
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_visual_train, x_visual_val, x_visual_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        elif model == 'Gru':
            trainer = trainerGru(results_path, dataset_name, x_visual_train, x_visual_val, x_visual_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name)
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_visual_train, x_visual_val, x_visual_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        elif model == 'DFLN_BiLstm':
            trainer = trainerDFLN_BiLstm(results_path, dataset_name, x_visual_train, x_visual_val, x_visual_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name)
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_visual_train, x_visual_val, x_visual_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        elif model == 'FAN':
            trainer = trainerFAN(results_path, dataset_name, x_visual_train, x_visual_val, x_visual_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name)
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_visual_train, x_visual_val, x_visual_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        elif model == 'Wav2vec2':
            trainer = trainerWav2vec2(results_path, dataset_name, x_audio_train, x_audio_val, x_audio_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name)
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_audio_train, x_audio_val, x_audio_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        elif model == 'UniSpeech':
            trainer = trainerUniSpeech(results_path, dataset_name, x_audio_train, x_audio_val, x_audio_test, y_train, y_val, y_test, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, fold_name, 'microsoft/unispeech-1350-en-353-fr-ft-1h')
            trainer.train_model()
            train_feature, val_feature, test_feature = trainer.get_feature(trainer.model, x_audio_train, x_audio_val, x_audio_test)
            trainer.save_feature(train_feature, val_feature, test_feature)
        else:
            print('not found model')