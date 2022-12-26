import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from src.common.utilities import *
from src.model.dense_FaceLiveNet_BiLstm import *
from src.train.trainer import trainer

class trainerDFLN_BiLstm(trainer):
    def __init__(self, results_path: str, dataset_name: str, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, 
                 epochs: int = 100, batch_size: int = 16, learn_rate: float = 1e-4, gradient_accumulation_steps: int = 2, is_transfer: bool = False, load_path: str = '', save_path: str = '', img_path: str = '', nickname:str = '', fold: str = ''):
        super(trainerDFLN_BiLstm, self).__init__(results_path, dataset_name, epochs, batch_size, learn_rate, gradient_accumulation_steps, is_transfer, load_path, save_path, img_path, nickname, fold)
        
        # 資料集
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset(x_train, x_val, x_test, y_train, y_val, y_test, self.num_classes, self.batch_size)

        self.model_name = 'DFLN_BiLstm'
        self.model_type = 'Visual'

        self.init_model()
        self.init_loss()
        self.init_optimizer_and_scheduler()

    def training_step(self, model: nn.Module, inputs: list):
        datas, labels_one_hot = inputs

        datas = datas.to(self.device)
        labels_one_hot = labels_one_hot.to(self.device)
            
        # Forward pass
        outputs = model(datas)

        return outputs, labels_one_hot
    
    def get_dataset(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, num_classes: int, batch_size: int = 64):
        x_train, x_val, x_test = self.get_dataset_x(x_train, x_val, x_test)

        y_train = torch.from_numpy(y_train.astype(np.int64))
        y_val = torch.from_numpy(y_val.astype(np.int64))
        y_test = torch.from_numpy(y_test.astype(np.int64))
        
        y_train_one_hot = nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32)
        y_val_one_hot = nn.functional.one_hot(y_val, num_classes=num_classes).to(torch.float32)
        y_test_one_hot = nn.functional.one_hot(y_test, num_classes=num_classes).to(torch.float32)

        train_dataset = TensorDataset(x_train, y_train_one_hot)
        val_dataset = TensorDataset(x_val, y_val_one_hot)
        test_dataset = TensorDataset(x_test, y_test_one_hot)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def get_feature(self, model: nn.Module, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
        x_train, x_val, x_test = self.get_dataset_x(x_train, x_val, x_test)

        train_dataset = TensorDataset(x_train)
        val_dataset = TensorDataset(x_val)
        test_dataset = TensorDataset(x_test)

        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
        train_feature = []
        val_feature = []
        test_feature = []

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data in train_loader:
                outputs = model(data[0].to(self.device))
                train_feature.append(outputs.detach().cpu().numpy())
            for data in val_loader:
                outputs = model(data[0].to(self.device))
                val_feature.append(outputs.detach().cpu().numpy())
            for data in test_loader:
                outputs = model(data[0].to(self.device))
                test_feature.append(outputs.detach().cpu().numpy())

        train_feature = torch.from_numpy(np.array(train_feature))
        val_feature = torch.from_numpy(np.array(val_feature))
        test_feature = torch.from_numpy(np.array(test_feature))

        train_feature = torch.reshape(train_feature, (train_feature.size()[0], train_feature.size()[-1]))
        val_feature = torch.reshape(val_feature, (val_feature.size()[0], val_feature.size()[-1]))
        test_feature = torch.reshape(test_feature, (test_feature.size()[0], test_feature.size()[-1]))

        return train_feature, val_feature, test_feature

    def get_dataset_x(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
        x_train = torch.from_numpy(np.moveaxis(x_train.astype(np.float32), -1, 2))
        x_val = torch.from_numpy(np.moveaxis(x_val.astype(np.float32), -1, 2))
        x_test = torch.from_numpy(np.moveaxis(x_test.astype(np.float32), -1, 2))
        return x_train, x_val, x_test

    def load_model(self, path: str = ''):
        is_load = super().load_model(path)
        if is_load:
            self.model.classifier = Dense_FaceLiveNet_BiLSTM_Classification_Head(num_classes=self.num_classes)
            self.init_loss()
            self.init_optimizer_and_scheduler()
            
    def init_model(self):
        self.model = Dense_FaceLiveNet_BiLSTM(self.num_classes)

    def init_loss(self):
        self.loss_func = nn.CrossEntropyLoss()

    def init_optimizer_and_scheduler(self):
        total_step = self.epochs * (len(self.train_loader) // self.gradient_accumulation_steps)
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            total_step += self.epochs
        
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.learn_rate)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_step)