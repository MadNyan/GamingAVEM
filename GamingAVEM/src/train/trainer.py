from pickle import TRUE
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix

from src.common.utilities import *
from src.dataPreprocess.preprocess import *
from src.dataPreprocess.preprocessAudio import *

class trainer(object):
    def __init__(self, results_path: str, dataset_name: str,
                 epochs: int, batch_size: int, learn_rate: float, gradient_accumulation_steps: int, is_transfer: bool, fold: str):
        super(trainer, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.results_path = results_path
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.is_transfer = is_transfer
        self.fold = fold        
        
        self.data_labels = get_data_labels(self.dataset_name)
        self.num_classes = len(self.data_labels)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = None
        self.model_name = ''
        self.model_type = ''

        self.loss_func = None
        self.optimizer = None
        self.scheduler = None
    
    def train_model(self):
        title = self.get_title()
        if self.is_transfer:
            self.load_model()

        make_dirs(self.results_path)
        csv = 'TYPE,FOLD,IS TRANSFER,MODEL,DATASET,NUM CLASSES,MAX VAL ACCURACY,MIN VAL ACCURACY,TEST ACCURACY,TIME COST\n'
    
        training_timepoint_start, _ = get_timepoint()
        history = self.training(self.model, self.train_loader, self.val_loader, self.epochs, self.gradient_accumulation_steps, self.loss_func, self.optimizer, self.scheduler)
        training_timepoint_end, _ = get_timepoint()
        _, time_cost_msg = get_time_cost(training_timepoint_start, training_timepoint_end)

        img_path = self.results_path + '/' + self.get_title() + '_' + str(self.is_transfer) + '/'
        make_dirs(img_path)
        plot_history_table(history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'], title=title, index=self.fold, path=img_path)

        max_val_accuracy = max(history['val_accuracy'])
        min_val_accuracy = min(history['val_accuracy'])
        
        test_accuracy, y_pred, y_true = self.testing(self.model, self.test_loader)

        if os.path.isfile(self.results_path + '/acc.csv') == True:
            csv = read_result(self.results_path + '/acc.csv')
            
        print('Training_time_cost: {}'.format(time_cost_msg))
        csv += '{},{},{},{},{},{:d},{:.4f},{:.4f},{:.4f},{}\n'.format(self.model_type, self.fold, str(self.is_transfer), self.model_name, self.dataset_name, self.num_classes, max_val_accuracy, min_val_accuracy, test_accuracy, time_cost_msg)

        matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
        plot_confusion_matrix(matrix, class_labels=self.data_labels, normalize=True, title=title, index=self.fold, path=img_path)
        
        write_result(self.results_path + '/acc.csv', csv)
        self.save_model()

    def training(self, model: nn.Module, train_loader: TensorDataset, val_loader: TensorDataset, epochs: int, gradient_accumulation_steps: int, loss_func: torch.nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler):
        model = model.to(self.device)

        # Train the model
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        total_step = len(train_loader)
    
        for epoch in range(epochs):
            model.train()        
            correct = 0
            total = 0
            for i, inputs in enumerate(train_loader):
                # Forward pass
                outputs, labels_one_hot = self.training_step(model, inputs)
                loss = loss_func(outputs, labels_one_hot)
        
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                loss.detach()

                if ((i + 1) % gradient_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler != None:
                        scheduler.step()

                train_loss = loss.item()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Learn rate: {:e}'.format(epoch+1, epochs, i+1, total_step, train_loss, scheduler.get_last_lr()[0]), end='\r', flush=True)

                # Calculating Accuracy
                _, labels = torch.max(labels_one_hot, 1)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print('')
            print('accuracy: {:.4f} %'.format(train_accuracy))
            print('{} / {}'.format(correct, total))
            history['accuracy'].append(train_accuracy)
            history['loss'].append(train_loss)
    
            # Val the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs in val_loader:
                    # Forward pass
                    outputs, labels_one_hot = self.training_step(model, inputs)
                    loss = loss_func(outputs, labels_one_hot)

                    # Calculating Accuracy
                    _, labels = torch.max(labels_one_hot, 1)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss = loss.item()

                val_accuracy = 100 * correct / total                    
                print('val_accuracy: {:.4f} %'.format(val_accuracy))
                print('{} / {}'.format(correct, total))
                history['val_accuracy'].append(val_accuracy)
                history['val_loss'].append(val_loss)

        return history

    def testing(self, model: nn.Module, test_loader: TensorDataset):
        model = model.to(self.device)

        # Test the model
        y_pred = []   #保存預測label
        y_true = []   #保存實際label
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0

            for inputs in test_loader:
                # Forward pass
                outputs, labels_one_hot = self.training_step(model, inputs)

                # Calculating Accuracy
                _, labels = torch.max(labels_one_hot, 1)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred.extend(predicted.view(-1).detach().cpu().numpy())     
                y_true.extend(labels.view(-1).detach().cpu().numpy())

            test_accuracy = 100 * correct / total
            print('test_accuracy: {:.4f} %'.format(test_accuracy))
            print('{} / {}'.format(correct, total))

            return test_accuracy, y_pred, y_true
        
    def save_model(self, path: str = ''):
        if path == '':
            path = self.get_save_path() + '.pth'
        remove_file(path)
        torch.save(self.model, path)

    def load_model(self, path: str = ''):
        is_load = False        
        if path == '':
            path = self.get_save_path() + '.pth'
        if os.path.isfile(path) == True:
            self.model = torch.load(path)
            is_load = True
        return is_load

    def save_feature(self, train_feature: np.ndarray, val_feature: np.ndarray, test_feature: np.ndarray):
        path = self.results_path + '/' + self.get_title() + '_' + self.fold + '_train_feature.npy'
        remove_file(path)
        np.save(path, train_feature)
        path = self.results_path + '/' + self.get_title() + '_' + self.fold + '_val_feature.npy'
        remove_file(path)
        np.save(path, val_feature)
        path = self.results_path + '/' + self.get_title() + '_' + self.fold + '_test_feature.npy'
        remove_file(path)
        np.save(path, test_feature)

    def load_feature(self, model_name: str):
        raise NotImplementedError()

    def get_title(self):
        return self.model_name + '_' + self.dataset_name

    def get_save_path(self):
        title = self.get_title()
        if self.is_transfer:
            title = self.model_name + '_transfer'
        if self.fold != '':
            title = title + '_' + self.fold
        
        return self.results_path + '/' + title

    def init_model(self):
        raise NotImplementedError()

    def init_loss(self):
        raise NotImplementedError()

    def init_optimizer_and_scheduler(self):
        raise NotImplementedError()

    def training_step(self, model: nn.Module, inputs: list):
        raise NotImplementedError()

    def get_dataset(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, num_classes: int, batch_size: int):
        raise NotImplementedError()
    
    def get_feature(self, model: nn.Module, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
        raise NotImplementedError()